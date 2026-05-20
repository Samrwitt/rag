"""Small JSON cache for on-demand dynamic farm context.

Stores weather, market, and soil lookups under ``data/dynamic`` so the assistant can
reuse recent answers before searching/fetching again.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "data" / "dynamic" / "on_demand_context.json"


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


def _ttl_days(domain: str) -> int:
    env_name = f"RAG_DEMAND_{domain.upper()}_TTL_DAYS"
    default = "1" if domain == "weather" else "7"
    try:
        return max(0, int(os.environ.get(env_name, default)))
    except ValueError:
        return int(default)


def _norm_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())[:400]


def _key(domain: str, query: str) -> str:
    raw = f"{domain}\0{_norm_query(query)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _read_store() -> dict[str, Any]:
    if not STORE_PATH.is_file():
        return {"records": {}}
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"records": {}}
    if not isinstance(data, dict):
        return {"records": {}}
    data.setdefault("records", {})
    return data


def _write_store(data: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now_iso()
    STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_cached(domain: str, query: str) -> dict[str, Any] | None:
    """Return a fresh cached record, or ``None``."""
    domain = (domain or "general").strip().lower()
    k = _key(domain, query)
    rec = (_read_store().get("records") or {}).get(k)
    if not isinstance(rec, dict):
        return None
    fetched_at = rec.get("fetched_at")
    try:
        ts = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
    except Exception:
        return None
    if _ttl_days(domain) == 0:
        return rec
    if _now() - ts > timedelta(days=_ttl_days(domain)):
        return None
    return rec


def put_cached(
    domain: str,
    query: str,
    body: str,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist and return a cache record."""
    domain = (domain or "general").strip().lower()
    data = _read_store()
    records = data.setdefault("records", {})
    rec = {
        "domain": domain,
        "query": query,
        "normalized_query": _norm_query(query),
        "body": body,
        "source": source,
        "metadata": metadata or {},
        "fetched_at": _now_iso(),
    }
    records[_key(domain, query)] = rec
    _write_store(data)
    return rec


def format_cached_block(rec: dict[str, Any]) -> str:
    fetched = rec.get("fetched_at") or ""
    source = rec.get("source") or "stored"
    body = (rec.get("body") or "").strip()
    return f"የተቀመጠ ወቅታዊ መረጃ፦ {source} ({fetched})\n{body}".strip()
