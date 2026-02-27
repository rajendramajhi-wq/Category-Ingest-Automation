#!/usr/bin/env python3
"""
llm_category_ingest_v2.py (slug-aware)

What this version supports (based on your manager payload like Untitled (4)):
- Accepts manager payload keys:
    category_name, category_slug, sub_category_name, sub_category_slug,
    description, max_score, prompt
- Uses category_slug / sub_category_slug *if provided* (Option 2)
  so your DB keys stay underscore-based (e.g., greeting_professionalism)
- Falls back to the legacy generate_slug() when slugs are not provided
- Stores prompt separately as meta.prompt
- Also uses prompt as a fallback "winning_pitch_phrase" if no pitch phrases are provided
- team_categories.max_subcat_score is NESTED by category key

You can import ingest_from_payload() from your FastAPI service.

Env vars:
- OPENAI_API_KEY (required if use_llm=True)
- MONGO_URI_DEV / MONGO_URI_PROD (or MONGO_URI)
"""

import os
import re
import json
import time
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import typer
from pydantic import BaseModel, Field

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # allow running payload->mongosh without OpenAI installed


# --------------------------
# Helpers / constants
# --------------------------

REQUIRED_COLLECTIONS = ["live_calls", "sessions", "team_categories", "team_configs"]


def generate_slug(s: str) -> str:
    # Legacy slug: replace non [a-z0-9_] with "*", then strip "*"
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "*", s)
    return s.strip("*")


def sanitize_slug_underscore(s: str) -> str:
    """
    For manager-provided slugs, keep underscore style:
      - lowercase
      - replace any non [a-z0-9_] with "_"
      - collapse multiple underscores
      - strip underscores
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(v: Any) -> List[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    out: List[str] = []
    for line in s.splitlines():
        out.extend([p.strip() for p in line.split(";") if p.strip()])
    return [x for x in out if x]


def resolve_db_name(company_id: str) -> str:
    return f"companyId_{company_id}_RA"


# --------------------------
# Payload -> DataFrame
# --------------------------

_CANONICAL_COLS = {
    "category": ["category", "category_name", "categoryname", "category title", "category_title"],
    "category_slug": ["category_slug", "categoryslug", "category_key", "categorykey", "cat_slug", "catkey"],
    "subcategory": ["subcategory", "sub_category", "sub category", "sub_category_name", "subcategory_name", "subcat", "sub_cat"],
    "subcategory_slug": ["sub_category_slug", "subcategory_slug", "sub_slug", "subcat_slug", "sub_category_key", "subcat_key"],
    "max_score": ["max_score", "maxscore", "max score", "score", "max points", "points"],
    "description": ["description", "sub_description", "subcategory_description", "desc"],
    "pitch": ["winning_pitch_phrases", "winning pitch phrases", "winning pitch phrase", "pitch", "pitches", "pitch_phrases"],
    "prompt": ["prompt", "instruction_prompt", "agent_prompt"],
}


def _find_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    # returns actual key in dict matching candidate (case/space/underscore-insensitive)
    norm = {re.sub(r"[\s_]+", "", str(k).strip().lower()): k for k in d.keys()}
    for c in candidates:
        ck = re.sub(r"[\s_]+", "", c.strip().lower())
        if ck in norm:
            return norm[ck]
    return None


def grid_payload_to_df(grid: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of dict rows into a canonical DataFrame.
    If category_slug/sub_category_slug exist, they are preserved (Option 2).
    """
    if not isinstance(grid, list) or not grid:
        raise ValueError("payload.grid/categories must be a non-empty list of row objects")

    rows_out: List[Dict[str, Any]] = []
    for i, r in enumerate(grid):
        if not isinstance(r, dict):
            raise ValueError(f"payload.rows[{i}] must be an object")

        k_cat = _find_key(r, _CANONICAL_COLS["category"])
        k_cat_slug = _find_key(r, _CANONICAL_COLS["category_slug"])
        k_sub = _find_key(r, _CANONICAL_COLS["subcategory"])
        k_sub_slug = _find_key(r, _CANONICAL_COLS["subcategory_slug"])
        k_max = _find_key(r, _CANONICAL_COLS["max_score"])
        k_desc = _find_key(r, _CANONICAL_COLS["description"])
        k_pitch = _find_key(r, _CANONICAL_COLS["pitch"])
        k_prompt = _find_key(r, _CANONICAL_COLS["prompt"])

        if not k_cat or not k_sub:
            raise ValueError(f"payload.rows[{i}] missing category/subcategory fields")

        max_raw = r.get(k_max) if k_max else 0
        try:
            max_score = int(float(max_raw)) if str(max_raw).strip() else 0
        except Exception:
            max_score = 0

        rows_out.append(
            {
                "Category": str(r.get(k_cat, "")).strip(),
                "Category Slug": str(r.get(k_cat_slug, "")).strip() if k_cat_slug else "",
                "Subcategory": str(r.get(k_sub, "")).strip(),
                "Subcategory Slug": str(r.get(k_sub_slug, "")).strip() if k_sub_slug else "",
                "Max Score": max_score,
                "Description": str(r.get(k_desc, "")).strip() if k_desc else "",
                "Winning pitch phrases": r.get(k_pitch) if k_pitch else "",
                "Prompt": str(r.get(k_prompt, "")).strip() if k_prompt else "",
            }
        )

    df = pd.DataFrame(rows_out)
    df.replace({"": pd.NA}, inplace=True)
    return df


# --------------------------
# Excel/CSV loader
# --------------------------

def load_grid(path: str, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Loads Excel/CSV into DataFrame and returns column mapping.
    Supports optional columns: category_slug, sub_category_slug, prompt.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(candidates: List[str], required: bool = True) -> Optional[str]:
        for cand in candidates:
            key = cand.strip().lower()
            if key in cols:
                return cols[key]
        if required:
            raise ValueError(f"Missing required column. Tried={candidates}. Found={list(df.columns)}")
        return None

    mapping = {
        "category": pick(["category", "categories", "cat", "category name", "category_name"]),
        "category_slug": pick(["category_slug", "category key", "category_key", "cat_slug"], required=False),
        "subcategory": pick(["subcategory", "sub category", "sub_category", "subcat", "subcategory name", "subcategory_name", "sub_category_name"]),
        "subcategory_slug": pick(["sub_category_slug", "subcategory_slug", "sub_category_key", "subcat_slug"], required=False),
        "max_score": pick(["max score", "max_score", "score", "max points"], required=False),
        "description": pick(["description", "desc", "subcategory description", "subcategory_description"], required=False),
        "pitch": pick(["winning pitch phrases", "winning_pitch_phrases", "winning pitch phrase", "pitch", "pitches"], required=False),
        "prompt": pick(["prompt"], required=False),
    }
    return df, mapping


# --------------------------
# OpenAI Structured Outputs
# --------------------------

class SubcategoryOut(BaseModel):
    name: str
    evaluation_criteria: List[str] = Field(default_factory=list)


class CategoryEnrichmentOut(BaseModel):
    category_description: str = ""
    subcategories: List[SubcategoryOut] = Field(default_factory=list)


def get_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai SDK is not installed. Install openai or set --no-llm.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def chunk_list(items: List[Any], n: int) -> List[List[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def openai_parse_enrichment(
    client: "OpenAI",
    model: str,
    category_name: str,
    rows: List[Dict[str, Any]],
    max_retries: int = 4,
) -> CategoryEnrichmentOut:
    expected_names = [r["name"] for r in rows]
    system_prompt = (
        "You are building a call QA scorecard.\n"
        "Return ONLY data that matches the provided JSON schema.\n"
        "Rules:\n"
        "- Keep each subcategory name EXACTLY the same as provided in input_rows.\n"
        "- For each subcategory, return 3-5 short operational evaluation criteria strings.\n"
        "- If winning_pitch_phrases exist, include ONE criterion like: Uses phrase like: <one phrase>\n"
        "- Keep English concise.\n"
    )
    user_payload = {
        "category_name": category_name,
        "expected_names": expected_names,
        "input_rows": rows,
        "output_schema": {
            "category_description": "string",
            "subcategories": [{"name": "string", "evaluation_criteria": ["string"]}],
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format=CategoryEnrichmentOut,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI parse failed after retries: {last_err}")


# --------------------------
# Build team_configs
# --------------------------

def build_team_configs_with_openai(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    company_id: str,
    team_id: str,
    model: str,
    chunk_size: int = 25,
) -> Dict[str, Any]:
    ts = now_iso()
    client = get_openai_client()

    cat_col = cols["category"]
    sub_col = cols["subcategory"]
    if not cat_col or not sub_col:
        raise ValueError("Need category and subcategory columns.")

    cat_slug_col = cols.get("category_slug")
    sub_slug_col = cols.get("subcategory_slug")
    max_col = cols.get("max_score")
    desc_col = cols.get("description")
    pitch_col = cols.get("pitch")
    prompt_col = cols.get("prompt")

    team_configs: Dict[str, Any] = {
        "company_id": str(company_id),
        "team_id": str(team_id),
        "categories": {},
        "created_at": ts,
        "updated_at": ts,
    }

    df2 = df.copy()
    df2[cat_col] = df2[cat_col].astype(str).str.strip()
    df2[sub_col] = df2[sub_col].astype(str).str.strip()

    for cat_name, g in df2.groupby(cat_col, dropna=False):
        cat_name = str(cat_name).strip()
        if not cat_name:
            continue

        # Option 2: use category_slug if present (must be consistent within category group)
        cat_slug_override = ""
        if cat_slug_col and cat_slug_col in g.columns:
            uniq = {str(x).strip() for x in g[cat_slug_col].dropna().tolist() if str(x).strip()}
            if len(uniq) > 1:
                raise ValueError(f"Multiple category_slug values for category '{cat_name}': {sorted(list(uniq))}")
            if len(uniq) == 1:
                cat_slug_override = sanitize_slug_underscore(list(uniq)[0])

        cat_slug = cat_slug_override or generate_slug(cat_name)

        if cat_slug in team_configs["categories"]:
            raise ValueError(f"Category key collision: '{cat_name}' -> {cat_slug}")

        raw_rows_all: List[Dict[str, Any]] = []
        for _, row in g.iterrows():
            sub_name = str(row[sub_col]).strip()
            if not sub_name:
                continue

            # sub slug override
            sub_slug_override = ""
            if sub_slug_col and sub_slug_col in g.columns:
                v = row.get(sub_slug_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    sub_slug_override = sanitize_slug_underscore(str(v).strip())

            max_score = 0
            if max_col and row.get(max_col) is not None and not (isinstance(row.get(max_col), float) and pd.isna(row.get(max_col))):
                try:
                    max_score = int(float(row[max_col]))
                except Exception:
                    max_score = 0

            desc = ""
            if desc_col and row.get(desc_col) is not None and not (isinstance(row.get(desc_col), float) and pd.isna(row.get(desc_col))):
                desc = str(row[desc_col]).strip()

            prompt_txt = ""
            if prompt_col and row.get(prompt_col) is not None and not (isinstance(row.get(prompt_col), float) and pd.isna(row.get(prompt_col))):
                prompt_txt = str(row[prompt_col]).strip()

            pitches: List[str] = []
            if pitch_col:
                pitches = parse_list(row.get(pitch_col))

            # if no explicit pitches, use prompt as a fallback pitch phrase
            if not pitches and prompt_txt:
                pitches = [prompt_txt]

            raw_rows_all.append(
                {
                    "name": sub_name,
                    "sub_slug_override": sub_slug_override,
                    "max_score": max_score,
                    "description": desc,
                    "winning_pitch_phrases": pitches,
                    "prompt": prompt_txt,
                }
            )

        cat_obj: Dict[str, Any] = {
            "name": cat_name,
            "is_static": True,
            "entity_type": None,
            "description": "",
            "total_score": 0,
            "subcategories": {},
            "created_at": ts,
            "updated_at": ts,
        }

        # Enrich with OpenAI in chunks
        enriched_desc = ""
        enriched_map: Dict[str, List[str]] = {}
        for chunk in chunk_list(raw_rows_all, chunk_size):
            # Strip helper keys before sending to OpenAI
            chunk_for_llm = [
                {
                    "name": r["name"],
                    "max_score": r["max_score"],
                    "description": r["description"],
                    "winning_pitch_phrases": r["winning_pitch_phrases"],
                }
                for r in chunk
            ]
            parsed = openai_parse_enrichment(client=client, model=model, category_name=cat_name, rows=chunk_for_llm)
            if not enriched_desc and parsed.category_description:
                enriched_desc = parsed.category_description.strip()

            for item in parsed.subcategories:
                nm = (item.name or "").strip()
                if not nm:
                    continue
                crit = [str(x).strip() for x in (item.evaluation_criteria or []) if str(x).strip()]
                enriched_map[nm] = crit

        cat_obj["description"] = enriched_desc or ""

        # Fill subcategories deterministically
        for r in raw_rows_all:
            sub_name = r["name"]
            sub_slug = r["sub_slug_override"] or generate_slug(sub_name)

            if sub_slug in cat_obj["subcategories"]:
                raise ValueError(f"Subcategory key collision in '{cat_name}': '{sub_name}' -> {sub_slug}")

            criteria = enriched_map.get(sub_name, [])
            if not criteria:
                base = r["description"]
                criteria = [base] if base else []

            cat_obj["subcategories"][sub_slug] = {
                "name": sub_name,
                "max_score": int(r["max_score"]),
                "is_dynamic": False,
                "description": r["description"],
                "required": True,
                "evaluation_criteria": criteria,
                "category_name": cat_name,
                "meta": {
                    "winning_pitch_phrases": r["winning_pitch_phrases"],
                    "prompt": r["prompt"],
                },
                "created_at": ts,
                "updated_at": ts,
            }

        cat_obj["total_score"] = sum(sc["max_score"] for sc in cat_obj["subcategories"].values())
        team_configs["categories"][cat_slug] = cat_obj

    return team_configs


# --------------------------
# Build team_categories (NESTED max_subcat_score)
# --------------------------

def build_team_categories(team_configs: Dict[str, Any], manager_id: Union[str, int]) -> Dict[str, Any]:
    ts = now_iso()
    cats = team_configs["categories"]

    cat_keys = list(cats.keys())
    cat_titles = [cats[k]["name"] for k in cat_keys]

    subcat_keys: List[str] = []
    subcat_titles: List[str] = []
    map_cat_to_subcat: Dict[str, List[str]] = {}

    max_subcat_score: Dict[str, Dict[str, int]] = {}
    max_cat_score: Dict[str, int] = {}

    for ck in cat_keys:
        subcats = cats[ck]["subcategories"]
        sks = list(subcats.keys())

        map_cat_to_subcat[ck] = sks
        max_cat_score[ck] = int(cats[ck]["total_score"])

        max_subcat_score[ck] = {}
        for sk in sks:
            subcat_keys.append(sk)
            subcat_titles.append(subcats[sk]["name"])
            max_subcat_score[ck][sk] = int(subcats[sk]["max_score"])

    return {
        "company_id": team_configs["company_id"],
        "team_id": team_configs["team_id"],
        "manager_id": str(manager_id),
        "cat_keys": cat_keys,
        "cat_titles": cat_titles,
        "subcat_keys": subcat_keys,
        "subcat_titles": subcat_titles,
        "map_cat_to_subcat": map_cat_to_subcat,
        "max_subcat_score": max_subcat_score,
        "max_cat_score": max_cat_score,
        "created_at": ts,
        "updated_at": ts,
    }


# --------------------------
# .mongosh writer + execution
# --------------------------

def to_pretty_js(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def write_mongosh(
    out_path: str,
    company_id: str,
    team_id: str,
    team_configs: Dict[str, Any],
    team_categories: Dict[str, Any],
    backup: bool = True,
) -> None:
    db_name = resolve_db_name(company_id)
    filt = {"company_id": str(company_id), "team_id": str(team_id)}

    js: List[str] = []
    js.append(f"use {db_name};")
    js.append("")

    js.append("// ensure required collections exist")
    js.append(f'const required = {json.dumps(REQUIRED_COLLECTIONS)};')
    js.append("const existing = db.getCollectionNames();")
    js.append("for (const c of required) { if (!existing.includes(c)) db.createCollection(c); }")
    js.append("")

    js.append("// enforce uniqueness on (company_id, team_id)")
    js.append("db.team_configs.createIndex({company_id:1, team_id:1}, {unique:true});")
    js.append("db.team_categories.createIndex({company_id:1, team_id:1}, {unique:true});")
    js.append("")

    if backup:
        js.append("// backup existing docs before overwrite")
        js.append('if (!db.getCollectionNames().includes("team_configs_backup")) db.createCollection("team_configs_backup");')
        js.append('if (!db.getCollectionNames().includes("team_categories_backup")) db.createCollection("team_categories_backup");')
        js.append(f"const filt = {json.dumps(filt)};")
        js.append("const oldTC = db.team_configs.findOne(filt);")
        js.append("if (oldTC) { oldTC.backed_up_at = new Date(); db.team_configs_backup.insertOne(oldTC); }")
        js.append("const oldTCat = db.team_categories.findOne(filt);")
        js.append("if (oldTCat) { oldTCat.backed_up_at = new Date(); db.team_categories_backup.insertOne(oldTCat); }")
        js.append("")

    js.append("// upsert team_configs + team_categories")
    js.append(f"const teamConfigs = {to_pretty_js(team_configs)};")
    js.append(f"const teamCategories = {to_pretty_js(team_categories)};")
    js.append("")

    js.append("// convert *_at ISO strings to Date objects recursively")
    js.append("function fixDates(obj){")
    js.append('  if (!obj || typeof obj !== "object") return;')
    js.append("  for (const k of Object.keys(obj)) {")
    js.append("    const v = obj[k];")
    js.append('    if ((k.endsWith(\"_at\") || k === \"created_at\" || k === \"updated_at\") && typeof v === \"string\") {')
    js.append("      obj[k] = new Date(v);")
    js.append("    } else if (v && typeof v === 'object') {")
    js.append("      fixDates(v);")
    js.append("    }")
    js.append("  }")
    js.append("}")
    js.append("fixDates(teamConfigs);")
    js.append("fixDates(teamCategories);")
    js.append("")

    js.append(f"db.team_configs.updateOne({json.dumps(filt)}, {{$set: teamConfigs}}, {{upsert:true}});")
    js.append(f"db.team_categories.updateOne({json.dumps(filt)}, {{$set: teamCategories}}, {{upsert:true}});")
    js.append('print(\"✅ Ingestion done for company_id=\" + teamConfigs.company_id + \" team_id=\" + teamConfigs.team_id);')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(js))



def execute_mongosh(env: str, out_file: str, mongo_uri: Optional[str] = None) -> None:
    uri = (mongo_uri or "").strip() or os.environ.get(f"MONGO_URI_{env.upper()}") or os.environ.get("MONGO_URI")
    if not uri:
        raise ValueError(
            f"Provide mongo_uri or set MONGO_URI_{env.upper()} (or MONGO_URI). "
            f"Example: export MONGO_URI_DEV=mongodb://localhost:27017"
        )

    # ✅ log what we are ACTUALLY using (super useful for debugging)
    print(f"[execute_mongosh] Using Mongo URI: {uri}")

    if shutil.which("mongosh"):
        subprocess.run(["mongosh", uri, out_file], check=True)
        return

    container = os.environ.get("MONGO_DOCKER_CONTAINER", "mongo-test")
    if shutil.which("docker") is None:
        raise FileNotFoundError("mongosh not found and docker not available. Install mongosh or run via Docker.")

    # Translate localhost -> host.docker.internal but keep the port (27018 stays 27018)
    docker_host = os.environ.get("MONGO_DOCKER_HOST", "host.docker.internal")
    docker_uri = uri.replace("mongodb://localhost", f"mongodb://{docker_host}")
    docker_uri = docker_uri.replace("mongodb://127.0.0.1", f"mongodb://{docker_host}")

    with open(out_file, "rb") as f:
        subprocess.run(
            ["docker", "exec", "-i", container, "mongosh", docker_uri],
            stdin=f,
            check=True,
        )


# --------------------------
# Payload ingestion entry (importable)
# --------------------------

@dataclass
class IngestResult:
    out_file: str
    company_id: str
    team_id: str


def _coalesce_str(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("none", "nan"):
            return s
    return ""


def ingest_from_payload(
    payload: Dict[str, Any],
    *,
    env: str = "dev",
    out_dir: str = "out",
    chunk_size: int = 25,
    model: str = "gpt-5",
    execute: bool = False,
    backup: bool = True,
    use_llm: bool = True,
    mongo_uri: Optional[str] = None,
) -> IngestResult:
    """
    Backend entry:
    - payload can include either:
        A) team_configs (object)  -> skip LLM
        B) grid (list) OR categories (list) -> build team_configs via LLM (unless use_llm=False)
    - Accepts team_id OR channel_id
    """
    company_id = _coalesce_str(payload.get("company_id"), payload.get("companyId"), payload.get("companyID"))
    team_id = _coalesce_str(payload.get("team_id"), payload.get("teamId"), payload.get("channel_id"), payload.get("channelId"))
    manager_id = _coalesce_str(payload.get("manager_id"), payload.get("managerId"), payload.get("managerID"))

    if not company_id or not team_id or not manager_id:
        raise ValueError("payload must include company_id, team_id (or channel_id), and manager_id")

    os.makedirs(out_dir, exist_ok=True)

    team_configs = payload.get("team_configs") or payload.get("teamConfigs")
    if team_configs:
        if not isinstance(team_configs, dict):
            raise ValueError("payload.team_configs must be an object")
        team_configs = dict(team_configs)
        team_configs["company_id"] = company_id
        team_configs["team_id"] = team_id
        if "categories" not in team_configs:
            raise ValueError("payload.team_configs missing 'categories'")
    else:
        rows = payload.get("grid")
        if rows is None:
            rows = payload.get("categories")
        if not isinstance(rows, list):
            raise ValueError("payload must include team_configs OR grid/categories (list)")

        df = grid_payload_to_df(rows)
        cols = {
            "category": "Category",
            "category_slug": "Category Slug",
            "subcategory": "Subcategory",
            "subcategory_slug": "Subcategory Slug",
            "max_score": "Max Score",
            "description": "Description",
            "pitch": "Winning pitch phrases",
            "prompt": "Prompt",
        }

        if not use_llm:
            raise ValueError("use_llm=False but payload.team_configs not provided")

        team_configs = build_team_configs_with_openai(
            df=df,
            cols=cols,
            company_id=company_id,
            team_id=team_id,
            model=model,
            chunk_size=chunk_size,
        )

    team_categories = build_team_categories(team_configs, manager_id)

    # Default: stable filename. If request_id exists, make it unique.
    rid = sanitize_slug_underscore(str(payload.get("request_id", "")).strip())
    suffix = f"_{rid}" if rid else ""
    out_file = os.path.join(out_dir, f"ingest_company{company_id}_team{team_id}{suffix}.mongosh")

    write_mongosh(out_file, company_id, team_id, team_configs, team_categories, backup=backup)

    if execute:
        execute_mongosh(env=env, out_file=out_file, mongo_uri=mongo_uri)

    return IngestResult(out_file=out_file, company_id=company_id, team_id=team_id)


# --------------------------
# CLI entry (kept)
# --------------------------

import sys  # only used for --payload-stdin


def _read_payload(payload_json: Optional[str], payload_stdin: bool) -> Optional[Dict[str, Any]]:
    if not payload_json and not payload_stdin:
        return None
    if payload_json:
        with open(payload_json, "r", encoding="utf-8") as f:
            return json.load(f)
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("--payload-stdin set but stdin is empty")
    return json.loads(raw)


def main(
    excel: str = typer.Argument(..., help='Excel/CSV path. Use "-" when sending payload via --payload-stdin.'),
    company_id: str = typer.Argument(...),
    team_id: str = typer.Argument(...),
    manager_id: str = typer.Argument(...),
    env: str = typer.Option("dev", "--env"),
    sheet: Optional[str] = typer.Option(None, "--sheet"),
    out_dir: str = typer.Option("out", "--out-dir"),
    chunk_size: int = typer.Option(25, "--chunk-size"),
    model: str = typer.Option(lambda: os.environ.get("OPENAI_MODEL", "gpt-5"), "--model"),
    execute: bool = typer.Option(True, "--execute/--no-execute"),
    backup: bool = typer.Option(True, "--backup/--no-backup"),
    payload_json: Optional[str] = typer.Option(None, "--payload-json"),
    payload_stdin: bool = typer.Option(False, "--payload-stdin"),
    no_llm: bool = typer.Option(False, "--no-llm"),
):
    payload = _read_payload(payload_json, payload_stdin)
    if payload is not None:
        payload.setdefault("company_id", company_id)
        payload.setdefault("team_id", team_id)
        payload.setdefault("manager_id", manager_id)
        res = ingest_from_payload(
            payload,
            env=env,
            out_dir=out_dir,
            chunk_size=chunk_size,
            model=model,
            execute=execute,
            backup=backup,
            use_llm=not no_llm,
        )
        print(f"✅ wrote: {res.out_file}")
        return

    df, cols = load_grid(excel, sheet=sheet)
    team_configs = build_team_configs_with_openai(
        df=df,
        cols=cols,
        company_id=company_id,
        team_id=team_id,
        model=model,
        chunk_size=chunk_size,
    )
    team_categories = build_team_categories(team_configs, manager_id)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"ingest_company{company_id}_team{team_id}.mongosh")
    write_mongosh(out_file, company_id, team_id, team_configs, team_categories, backup=backup)
    print(f"✅ wrote: {out_file}")

    if not execute:
        print("ℹ️ --no-execute set; not running mongosh")
        return

    execute_mongosh(env=env, out_file=out_file)
    print("✅ mongosh executed successfully")


if __name__ == "__main__":
    typer.run(main)
