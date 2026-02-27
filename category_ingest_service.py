
"""
category_ingest_service.py

Features:
- POST /ingest returns immediately (202 Accepted) with job_id
- Background thread executes ingest_from_payload
- GET /ingest/status/{job_id} to check received -> processing -> success/error
- Date-wise IST log folders:
    logs/YYYY-MM-DD/ingest.log
  Folder is created ONLY when the first ingest log is written that day.
  If no ingests happen on a date, no folder is created.

Run:
  uvicorn category_ingest_service:app --host 0.0.0.0 --port 8088 --reload
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from llm_category_ingest_v2 import ingest_from_payload  # same folder import

app = FastAPI(title="Category Ingest Service")

API_KEY = os.environ.get("INGEST_API_KEY", "").strip()

# -------- Logging (IST date-wise folders) --------

IST = ZoneInfo("Asia/Kolkata")
LOGS_ROOT = os.environ.get("INGEST_LOGS_DIR", "logs")  # change if you want


class ISTFormatter(logging.Formatter):
    """Formatter that prints time in IST."""
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


class ISTDailyFolderFileHandler(logging.Handler):
    """
    Writes logs to logs/YYYY-MM-DD/ingest.log (IST).
    Folder and file are created lazily on first emit.
    """

    def __init__(self, base_dir: str, filename: str = "ingest.log"):
        super().__init__()
        self.base_dir = base_dir
        self.filename = filename
        self._lock = threading.RLock()
        self._current_date = None  # YYYY-MM-DD
        self._stream = None

    def _today_str(self) -> str:
        return datetime.now(IST).strftime("%Y-%m-%d")

    def _ensure_stream(self) -> None:
        today = self._today_str()
        if self._stream is None or self._current_date != today:
            # rotate by date folder
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass

            day_dir = os.path.join(self.base_dir, today)
            os.makedirs(day_dir, exist_ok=True)  # folder created ONLY when we log
            path = os.path.join(day_dir, self.filename)
            self._stream = open(path, "a", encoding="utf-8")
            self._current_date = today

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._ensure_stream()
                assert self._stream is not None
                self._stream.write(msg + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        super().close()


def setup_ingest_logger() -> logging.Logger:
    logger = logging.getLogger("ingest")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate into root logger

    # avoid duplicate handlers on --reload
    has_daily = any(isinstance(h, ISTDailyFolderFileHandler) for h in logger.handlers)
    if not has_daily:
        logger.handlers.clear()

        fmt = ISTFormatter("%(asctime)s IST | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # file handler (date-wise folders)
        fh = ISTDailyFolderFileHandler(LOGS_ROOT, "ingest.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # also keep console logs (optional)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


logger = setup_ingest_logger()

# -------- Job store (in-memory) --------
# NOTE: resets on restart/reload; good enough for local.
JOBS: dict[str, dict[str, Any]] = {}


# -------- Helpers --------

def _redact_mongo_uri(uri: str) -> str:
    # Avoid logging credentials
    try:
        p = urlparse(uri)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        path = p.path or ""
        return f"{p.scheme}://{host}{port}{path}"
    except Exception:
        return "<redacted>"


def _first_present(payload: dict, keys: list[str]):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return None



def _pick_mongo_uri_from_payload(payload: dict) -> tuple[Optional[int], Optional[str]]:
    direct = payload.get("mongo_uri") or payload.get("mongoUri") or payload.get("mongodb_url") or payload.get("mongo_url")
    if isinstance(direct, str) and direct.strip():
        return None, direct.strip()

    flag = _first_present(payload, ["mongo_target", "mongoTarget", "mongo_server", "db_target"])
    if flag is None:
        return None, None

    s = str(flag).strip().lower()
    if s in ("0", "false"):
        flag_int = 0
    elif s in ("1", "true"):
        flag_int = 1
    else:
        flag_int = int(s)

    if flag_int == 0:
        return 0, (os.environ.get("MONGO_URI_X") or os.environ.get("MONGO_URI_0"))
    if flag_int == 1:
        return 1, (os.environ.get("MONGO_URI_Y") or os.environ.get("MONGO_URI_1"))

    raise HTTPException(status_code=400, detail="mongo_target must be 0 or 1")



def _require_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (not x_api_key or x_api_key.strip() != API_KEY):
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")


def _parse_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _parse_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _json_maybe(v: Any) -> Any:
    """If form-data field contains JSON string, parse it."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _rows_from_csv_bytes(csv_bytes: bytes) -> list[dict[str, Any]]:
    """
    CSV -> manager-style categories rows.
    Accepts headers like:
      category_name/sub_category_name/max_score/description/prompt/category_slug/sub_category_slug
    or:
      Category/Subcategory/Max Score/Description/Prompt/Category Slug/Subcategory Slug
    """
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if len(df.columns) == 1:
            df = pd.read_csv(io.BytesIO(csv_bytes), sep="\t")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def col(*names: str) -> Optional[str]:
        for n in names:
            k = n.lower()
            if k in lower:
                return lower[k]
        return None

    c_cat = col("category", "category_name", "Category", "Category Name")
    c_cat_slug = col("category_slug", "Category Slug", "category key", "category_key")
    c_sub = col("subcategory", "sub_category_name", "sub category", "Subcategory", "Sub Category Name")
    c_sub_slug = col("sub_category_slug", "Subcategory Slug", "sub_category_key", "subcat_slug")
    c_score = col("max_score", "max score", "score", "Max Score")
    c_desc = col("description", "Description")
    c_prompt = col("prompt", "Prompt")

    if not c_cat or not c_sub or not c_score:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain category + subcategory + max_score columns. Found: {list(df.columns)}",
        )

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        cat = r.get(c_cat)
        sub = r.get(c_sub)
        sc = r.get(c_score)
        if pd.isna(cat) or pd.isna(sub) or pd.isna(sc):
            continue

        row: dict[str, Any] = {
            "category_name": str(cat).strip(),
            "sub_category_name": str(sub).strip(),
            "max_score": int(float(sc)),
        }
        if c_cat_slug and r.get(c_cat_slug) is not None and not (isinstance(r.get(c_cat_slug), float) and pd.isna(r.get(c_cat_slug))):
            row["category_slug"] = str(r.get(c_cat_slug)).strip()
        if c_sub_slug and r.get(c_sub_slug) is not None and not (isinstance(r.get(c_sub_slug), float) and pd.isna(r.get(c_sub_slug))):
            row["sub_category_slug"] = str(r.get(c_sub_slug)).strip()
        if c_desc and r.get(c_desc) is not None and not (isinstance(r.get(c_desc), float) and pd.isna(r.get(c_desc))):
            row["description"] = str(r.get(c_desc)).strip()
        if c_prompt and r.get(c_prompt) is not None and not (isinstance(r.get(c_prompt), float) and pd.isna(r.get(c_prompt))):
            row["prompt"] = str(r.get(c_prompt)).strip()

        rows.append(row)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV parsed but produced 0 valid rows")
    return rows


# -------- Background job runner --------

def _run_ingest_job(job_id: str, payload: Dict[str, Any], opts: Dict[str, Any]) -> None:
    t0 = time.time()
    try:
        JOBS[job_id].update({"status": "processing", "stage": "ingest_from_payload", "started_at": time.time()})
        logger.info("JOB=%s ⏳ processing started (execute=%s use_llm=%s)", job_id, opts["execute"], opts["use_llm"])

        res = ingest_from_payload(payload, **opts)

        JOBS[job_id].update(
            {
                "status": "success",
                "stage": "done",
                "company_id": res.company_id,
                "team_id": res.team_id,
                "out_file": res.out_file,
                "executed": bool(opts.get("execute", False)),
                "elapsed_s": round(time.time() - t0, 2),
                "finished_at": time.time(),
            }
        )
        logger.info(
            "JOB=%s ✅ completed company_id=%s team_id=%s out_file=%s elapsed=%.2fs",
            job_id,
            res.company_id,
            res.team_id,
            res.out_file,
            (time.time() - t0),
        )
    except Exception as e:
        JOBS[job_id].update(
            {"status": "error", "stage": "failed", "error": str(e), "elapsed_s": round(time.time() - t0, 2), "finished_at": time.time()}
        )
        logger.exception("JOB=%s ❌ failed: %s", job_id, str(e))





# import time
# import uuid

@app.middleware("http")
async def log_ingest_requests(request: Request, call_next):
    """
    Logs every request to /ingest* including:
    - 405 (method not allowed)
    - 401 (unauthorized)
    - 500 (unhandled errors)
    Logs go to your IST date-wise folder via the same `logger`.
    """
    t0 = time.time()
    rid = uuid.uuid4().hex[:12]  # short request id for tracing

    response = None
    try:
        response = await call_next(request)
        return response
    except Exception:
        # If something truly crashes, log it (will show as 500)
        logger.exception(
            "REQ=%s ❌ unhandled exception method=%s path=%s ip=%s",
            rid,
            request.method,
            request.url.path,
            getattr(request.client, "host", None),
        )
        raise
    finally:
        # This will run for 401/405 too (as long as FastAPI returns a response)
        if request.url.path.startswith("/ingest"):
            status = response.status_code if response is not None else 500
            ms = (time.time() - t0) * 1000.0
            logger.info(
                "REQ=%s %s %s -> %s (%.1f ms) ip=%s",
                rid,
                request.method,
                request.url.path,
                status,
                ms,
                getattr(request.client, "host", None),
            )















# -------- Routes --------



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found (maybe server restarted/reloaded)")
    return job


@app.post("/ingest")
async def ingest(request: Request, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    ct = (request.headers.get("content-type") or "").lower()

    payload: Dict[str, Any] = {}

    if "application/json" in ct:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")

    elif "multipart/form-data" in ct:
        form = await request.form()
        payload = {k: _json_maybe(form.get(k)) for k in form.keys()}

        upload = form.get("file") or form.get("csv")
        if upload is not None:
            file_bytes = await upload.read()
            payload["categories"] = _rows_from_csv_bytes(file_bytes)

        if isinstance(payload.get("payload"), dict):
            for k, v in payload["payload"].items():
                payload.setdefault(k, v)

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {ct}")

    # create a job_id (reusing request_id if user sent it)
    job_id = str(payload.get("request_id") or uuid.uuid4().hex)
    payload["request_id"] = job_id  # helps your downstream filename suffix too

    # normalize runtime options
    opts: Dict[str, Any] = {
        "env": str(payload.get("env") or "dev"),
        "out_dir": str(payload.get("out_dir") or "out"),
        "chunk_size": _parse_int(payload.get("chunk_size"), 25),
        "model": str(payload.get("model") or os.environ.get("OPENAI_MODEL", "gpt-5")),
        "execute": _parse_bool(payload.get("execute"), default=False),
        "backup": _parse_bool(payload.get("backup"), default=True),
        "use_llm": _parse_bool(payload.get("use_llm"), default=True),
    }

    rows = payload.get("categories") or payload.get("grid")
    row_count = len(rows) if isinstance(rows, list) else None

    # record job immediately
    JOBS[job_id] = {
        "status": "received",
        "stage": "queued",
        "content_type": ct,
        "rows": row_count,
        "created_at": time.time(),
    }

    # FIRST log line for the day creates: logs/YYYY-MM-DD/ingest.log (IST)
    logger.info("JOB=%s ✅ payload received; queued (rows=%s)", job_id, row_count)






    mongo_flag, mongo_uri = _pick_mongo_uri_from_payload(payload)

    # If they provided mongo_target but env var missing:
    if mongo_flag is not None and not mongo_uri:
        raise HTTPException(
            status_code=400,
            detail=f"mongo_target={mongo_flag} provided but no matching env var set. "
                f"Set MONGO_URI_X/MONGO_URI_Y (or MONGO_URI_0/MONGO_URI_1)."
        )

    opts = {
        "env": str(payload.get("env") or "dev"),
        "out_dir": str(payload.get("out_dir") or "out"),
        "chunk_size": _parse_int(payload.get("chunk_size"), 25),
        "model": str(payload.get("model") or os.environ.get("OPENAI_MODEL", "gpt-5")),
        "execute": _parse_bool(payload.get("execute"), default=False),
        "backup": _parse_bool(payload.get("backup"), default=True),
        "use_llm": _parse_bool(payload.get("use_llm"), default=True),
        "mongo_uri": mongo_uri,  # ✅ NEW
    }

    # optionally store in JOBS + logs
    JOBS[job_id]["mongo_target"] = mongo_flag
    JOBS[job_id]["mongo"] = _redact_mongo_uri(mongo_uri) if mongo_uri else None
    logger.info("JOB=%s mongo_target=%s mongo=%s", job_id, mongo_flag, _redact_mongo_uri(mongo_uri) if mongo_uri else None)




    # run job in a separate thread so status endpoint stays responsive
    t = threading.Thread(target=_run_ingest_job, args=(job_id, payload, opts), daemon=True)
    t.start()

    return JSONResponse(
        {
            "status": "accepted",
            "job_id": job_id,
            "message": "payload received; processing started",
            "status_endpoint": f"/ingest/status/{job_id}",
        },
        status_code=202,
    )