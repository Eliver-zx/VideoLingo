from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.api.schemas import CreateDomainRequest, TermsUpdateRequest
from core.utils.config_utils import load_key_or, update_key
from core.workspace import (
    DEFAULT_GLOSSARY_DOMAIN,
    create_glossary_domain,
    delete_glossary_domain,
    domain_dict_json_path,
    get_glossary_domain_meta,
    list_glossary_domains,
    load_domain_glossary,
    load_global_glossary,
    promote_terms_to_domain,
    sanitize_domain_name,
    save_domain_glossary,
    save_global_glossary,
    search_domain_dict_entries,
)

router = APIRouter()

_SRC_HEADER_NAMES = ("src", "source", "original", "term", "en", "原文", "原词")
_TGT_HEADER_NAMES = (
    "tgt",
    "target",
    "translation",
    "zh",
    "zh-chs",
    "译文",
    "译名",
)
_NOTE_HEADER_NAMES = ("note", "notes", "备注")
_ALIASES_HEADER_NAMES = ("aliases", "alias", "别名")


@router.get("/api/glossary")
def get_global_glossary(domain: str | None = None):
    if domain is not None:
        d = domain.strip()
        if not d:
            return {"domain": "", "terms": [], "has_dict": False, "dict_count": 0}
        try:
            meta = get_glossary_domain_meta(d)
            terms = load_domain_glossary(d)
            return {
                "domain": meta["name"],
                "terms": terms,
                "has_dict": meta["has_dict"],
                "dict_count": meta["dict_count"],
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"terms": load_global_glossary()}


@router.put("/api/glossary")
def put_global_glossary(request: TermsUpdateRequest, domain: str | None = None):
    terms_list = request.terms or []
    target_domain = (
        domain
        if domain is not None
        else (request.glossary_domain if request.glossary_domain is not None else None)
    )
    if target_domain is not None:
        d = target_domain.strip()
        if not d:
            return {"saved": 0, "domain": ""}
        try:
            save_domain_glossary(d, terms_list)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"saved": len(terms_list), "domain": d}
    save_global_glossary(terms_list)
    return {"saved": len(terms_list)}


@router.get("/api/glossary/domains")
def api_list_glossary_domains():
    domains = list_glossary_domains()
    active = str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    return {"domains": domains, "active_domain": active}


@router.post("/api/glossary/domains", status_code=201)
def api_create_glossary_domain(request: CreateDomainRequest):
    raw_name = (request.domain or request.name or "").strip()
    if not raw_name:
        raise HTTPException(422, "Domain name is required")
    try:
        meta = create_glossary_domain(raw_name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    update_key("glossary_domain", meta["name"], add_missing=True)
    return {
        "domain": meta["name"],
        "meta": meta,
        "domains": list_glossary_domains(),
    }


@router.delete("/api/glossary/domains/{domain}")
def api_delete_glossary_domain(domain: str):
    try:
        safe = sanitize_domain_name(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    deleted = delete_glossary_domain(safe)
    if not deleted:
        raise HTTPException(404, f"Glossary domain not found: {safe}")
    active = str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    if active == safe:
        update_key("glossary_domain", "", add_missing=True)
        active = ""
    return {
        "deleted": safe,
        "active_domain": active,
        "domains": list_glossary_domains(),
    }


@router.get("/api/glossary/domains/{domain}")
def api_get_glossary_domain(domain: str):
    try:
        meta = get_glossary_domain_meta(domain)
        terms = load_domain_glossary(meta["name"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**meta, "terms": terms}


@router.put("/api/glossary/domains/{domain}")
def api_put_glossary_domain(domain: str, request: TermsUpdateRequest):
    try:
        safe = sanitize_domain_name(domain)
        save_domain_glossary(safe, request.terms or [])
        meta = get_glossary_domain_meta(safe)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"saved": meta["terms_count"], "domain": meta["name"], "meta": meta}


@router.get("/api/glossary/domains/{domain}/dict")
def api_search_glossary_domain_dict(domain: str, q: str = "", limit: int = 50):
    try:
        dict_file = domain_dict_json_path(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not dict_file.is_file():
        return {"entries": None, "total": 0}
    try:
        total, entries = search_domain_dict_entries(domain, q=q, limit=limit)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            500, f"Failed to read domain dictionary {dict_file.name}: {exc}"
        ) from exc
    return {"entries": entries, "total": total}


@router.delete("/api/glossary/domains/{domain}/dict")
def api_delete_glossary_domain_dict(domain: str):
    try:
        dict_file = domain_dict_json_path(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dict_file.is_file():
        dict_file.unlink()
    meta = get_glossary_domain_meta(domain)
    return {
        "domain": meta["name"],
        "meta": meta,
        "terms": load_domain_glossary(meta["name"]),
    }


def _first_matching_col_idx(
    cols_lower: dict[str, int], names: tuple[str, ...], fallback: int | None = None
) -> int | None:
    for name in names:
        if name in cols_lower:
            return cols_lower[name]
    return fallback


@router.post("/api/glossary/domains/{domain}/import")
def api_import_glossary_domain_file(
    domain: str,
    file: UploadFile = File(...),
    target: str = Form(default="auto"),
):
    import pandas as pd
    from core._4_1_summarize import _extract_generic_kb_pairs, _normalize_aliases

    try:
        safe = sanitize_domain_name(domain)
        create_glossary_domain(safe)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw_name = file.filename or "terms.csv"
    ext = Path(raw_name).suffix.lower()
    content = file.file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty")

    if ext == ".json":
        try:
            parsed = json.loads(content.decode("utf-8-sig"))
        except Exception as exc:
            raise HTTPException(422, f"Invalid JSON file: {exc}") from exc
        extracted: dict[str, dict[str, Any]] = {}
        _extract_generic_kb_pairs(parsed, extracted, source_tag="domain_dict")
        if not extracted:
            raise HTTPException(422, "No valid bilingual term pairs found in JSON")
        dict_file = domain_dict_json_path(safe)
        dict_file.parent.mkdir(parents=True, exist_ok=True)
        dict_file.write_bytes(content)
        meta = get_glossary_domain_meta(safe)
        meta["dict_count"] = len(extracted)
        return {
            "imported_as": "dict.json",
            "imported_kind": "dict",
            "imported_count": len(extracted),
            "dict_count": len(extracted),
            "terms_count": meta["terms_count"],
            "domain": meta["name"],
            "meta": meta,
            "terms": load_domain_glossary(safe),
        }

    def _read_csv_with_encoding_fallback(
        raw_bytes: bytes, **kwargs: Any
    ) -> pd.DataFrame:
        for enc in ("utf-8-sig", "gb18030", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig", **kwargs)

    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        csv_sep = "\t" if ext == ".tsv" else ","
        try:
            if ext in {".csv", ".tsv"}:
                raw_df = _read_csv_with_encoding_fallback(
                    content, sep=csv_sep, header=None
                )
            else:
                raw_df = pd.read_excel(io.BytesIO(content), header=None)
        except Exception as exc:
            raise HTTPException(422, f"Could not parse {ext} file: {exc}") from exc

        ncols = int(raw_df.shape[1])
        if ncols < 2:
            raise HTTPException(
                422, "Table must contain at least source and target columns"
            )

        rows_to_merge: list[dict[str, Any]] = []
        if not raw_df.empty:
            first_row = [
                "" if bool(pd.isna(v)) else str(v).strip()
                for v in raw_df.iloc[0].tolist()
            ]
            cols_lower: dict[str, int] = {}
            for idx, label in enumerate(first_row):
                if label:
                    cols_lower.setdefault(label.lower(), idx)
            has_header = any(k in cols_lower for k in _SRC_HEADER_NAMES)
            if has_header:
                data_df = raw_df.iloc[1:]
                src_idx = _first_matching_col_idx(cols_lower, _SRC_HEADER_NAMES, 0)
                tgt_idx = _first_matching_col_idx(cols_lower, _TGT_HEADER_NAMES, 1)
                note_idx = _first_matching_col_idx(cols_lower, _NOTE_HEADER_NAMES)
                aliases_idx = _first_matching_col_idx(cols_lower, _ALIASES_HEADER_NAMES)
                fix_idx = cols_lower.get("fix_source")
            else:
                data_df = raw_df
                src_idx = 0
                tgt_idx = 1
                note_idx = 2 if ncols >= 3 else None
                aliases_idx = 3 if ncols >= 4 else None
                fix_idx = 4 if ncols >= 5 else None

            assert src_idx is not None and tgt_idx is not None
            for row_vals in data_df.itertuples(index=False, name=None):
                s_val = row_vals[src_idx]
                t_val = row_vals[tgt_idx]
                if (
                    s_val is None
                    or t_val is None
                    or bool(pd.isna(s_val))
                    or bool(pd.isna(t_val))
                ):
                    continue
                s = str(s_val).strip()
                t = str(t_val).strip()
                if (
                    not s
                    or not t
                    or s.lower() in {"src", "source", "original", "term", "nan"}
                ):
                    continue
                n_val = row_vals[note_idx] if note_idx is not None else ""
                n = (
                    ""
                    if (n_val is None or bool(pd.isna(n_val)))
                    else str(n_val).strip()
                )
                al_val = row_vals[aliases_idx] if aliases_idx is not None else ""
                al_raw = (
                    ""
                    if (al_val is None or bool(pd.isna(al_val)))
                    else str(al_val).strip()
                )
                f_val = row_vals[fix_idx] if fix_idx is not None else False
                fix_bool = (
                    False
                    if (f_val is None or bool(pd.isna(f_val)))
                    else str(f_val).strip().lower() in {"1", "true", "yes"}
                )
                rows_to_merge.append(
                    {
                        "src": s,
                        "tgt": t,
                        "aliases": _normalize_aliases(al_raw, s),
                        "note": n,
                        "fix_source": fix_bool,
                    }
                )

        if target.strip().lower() == "dict" or len(rows_to_merge) > 500:
            has_rich_fields = any(
                item.get("note") or item.get("aliases") or item.get("fix_source")
                for item in rows_to_merge
            )
            dict_payload: Any = (
                {"terms": rows_to_merge}
                if has_rich_fields
                else {item["src"]: item["tgt"] for item in rows_to_merge}
            )
            dict_file = domain_dict_json_path(safe)
            dict_file.parent.mkdir(parents=True, exist_ok=True)
            dict_file.write_text(
                json.dumps(dict_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            meta = get_glossary_domain_meta(safe)
            return {
                "imported_as": "dict.json",
                "imported_kind": "dict",
                "imported_count": len(rows_to_merge),
                "dict_count": len(rows_to_merge),
                "terms_count": meta["terms_count"],
                "domain": meta["name"],
                "meta": meta,
                "terms": load_domain_glossary(safe),
            }

        promoted = promote_terms_to_domain(rows_to_merge, domain=safe)
        meta = get_glossary_domain_meta(safe)
        return {
            "imported_as": "terms.csv",
            "imported_kind": "terms",
            "imported_count": promoted,
            "terms_count": meta["terms_count"],
            "dict_count": meta["dict_count"],
            "domain": meta["name"],
            "meta": meta,
            "terms": load_domain_glossary(safe),
        }

    raise HTTPException(
        422,
        f"Unsupported glossary file format: {ext} (expected .csv, .tsv, .xlsx, or .json)",
    )
