from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_kb_disambiguation_prompt
from core.utils import ask_gpt, check_cancel, load_key_or

from .lexical import (
    _EXTRA_COMMON_WORDS,
    _KB_STOP_TOKENS,
    _is_standard_english,
    _tokenize_surface,
)
from .normalize import _normalize_aliases


def _derive_in_video_shorthands(
    terms_map: dict[str, dict[str, Any]], full_text: str = ""
) -> set[str]:
    """Derive 1-word constituent shorthands for unique multi-word terms in the video and return all claimed lowercase surfaces."""
    tok_to_multi: dict[str, list[dict[str, Any]]] = {}
    protected_single: set[str] = set()
    for entry in terms_map.values():
        bw, _, _ = _tokenize_surface(entry["src"])
        if len(bw) == 1:
            if bw[0] not in _EXTRA_COMMON_WORDS:
                protected_single.add(bw[0])
        elif len(bw) >= 2:
            for w in dict.fromkeys(bw):
                if len(w) >= 4 and w not in _KB_STOP_TOKENS:
                    tok_to_multi.setdefault(w, []).append(entry)

    if full_text:
        candidate_ws = {
            w
            for w, ent_list in tok_to_multi.items()
            if len(ent_list) == 1 and w not in protected_single
        }
        if candidate_ws:
            lines = full_text.split("\n")
            lines_low = [ln.lower() for ln in lines]
            for w, ent_list in tok_to_multi.items():
                if w not in candidate_ws:
                    continue
                matching_idxs = [idx for idx, ll in enumerate(lines_low) if w in ll]
                if not matching_idxs:
                    continue
                ent = ent_list[0]
                canon = ent["src"]
                cleaned = "".join(
                    ("\n" + lines[idx] if idx > 0 else lines[idx])
                    for idx in matching_idxs
                )
                for surf in (canon, *(ent.get("aliases") or [])):
                    if surf:
                        cleaned = re.sub(
                            re.escape(surf), " ", cleaned, flags=re.IGNORECASE
                        )
                all_occ = re.findall(
                    rf"\b{re.escape(w)}\b", cleaned, flags=re.IGNORECASE
                )
                if not all_occ:
                    continue
                mid_cap = re.findall(
                    rf"(?<![.!?,:;]\s)(?<![\n])\b{re.escape(w.capitalize())}\b",
                    cleaned,
                )
                cap_ratio = sum(1 for x in all_occ if x[0].isupper()) / len(all_occ)
                if not _is_standard_english(w) or (mid_cap and cap_ratio >= 0.5):
                    shorthand = mid_cap[0] if mid_cap else all_occ[0]
                    cur_als = list(ent.get("aliases") or [])
                    if shorthand.lower() not in {a.lower() for a in cur_als}:
                        cur_als.append(shorthand)
                        ent["aliases"] = _normalize_aliases(cur_als, canon)
                    if w in terms_map and w in _EXTRA_COMMON_WORDS:
                        terms_map.pop(w, None)

    claimed: set[str] = set()
    for entry in terms_map.values():
        claimed.add(entry["src"].lower())
        for al in entry.get("aliases") or []:
            claimed.add(str(al).strip().lower())
    return claimed


def _propagate_possessive_aliases(merged_by_src: dict[str, dict[str, Any]]) -> None:
    multi_by_first_word: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    single_oov_entries: list[tuple[str, str, list[str]]] = []
    for k, entry in merged_by_src.items():
        bw, _, _ = _tokenize_surface(entry["src"])
        if len(bw) == 1 and len(bw[0]) >= 5 and not _is_standard_english(bw[0]):
            als = entry.get("aliases") or []
            if als:
                single_oov_entries.append((k, bw[0], als))
        elif len(bw) >= 2:
            multi_by_first_word.setdefault(bw[0], []).append(
                (k, entry, " ".join(bw[1:]))
            )

    for k, w0, als in single_oov_entries:
        for other_k, other_ent, rest in multi_by_first_word.get(w0, ()):
            if other_k == k:
                continue
            new_als = list(other_ent.get("aliases") or [])
            seen_low = {x.lower() for x in new_als}
            for a in als:
                cand_al = f"{a} {rest}"
                if cand_al.lower() not in seen_low:
                    new_als.append(cand_al)
                    seen_low.add(cand_al.lower())
            other_ent["aliases"] = _normalize_aliases(new_als, other_ent["src"])


def resolve_ambiguous_kb_candidates_with_llm(
    unresolved_items: list[dict[str, Any]],
    *,
    ask_gpt_fn: Callable[..., Any] = ask_gpt,
    load_key_or_fn: Callable[..., Any] = load_key_or,
) -> dict[str, str]:
    if not unresolved_items:
        return {}
    decisions: dict[str, str] = {}
    batch_size = 30
    batches = [
        unresolved_items[i : i + batch_size]
        for i in range(0, len(unresolved_items), batch_size)
    ]

    def _run_batch(batch: list[dict[str, Any]]) -> dict[str, str]:
        check_cancel()
        prompt = get_kb_disambiguation_prompt({"items": batch})
        resp = ask_gpt_fn(prompt, resp_type="json", log_title="kb_disambiguate")
        out: dict[str, str] = {}
        if isinstance(resp, dict):
            if isinstance(resp.get("decisions"), list):
                for d in resp["decisions"]:
                    if isinstance(d, dict):
                        sp = str(d.get("span") or "").strip().lower()
                        ch = str(d.get("canonical_src") or "NONE").strip()
                        if sp and ch and ch.upper() != "NONE":
                            out[sp] = ch
            else:
                for idx, item in enumerate(batch, 1):
                    ch = str(resp.get(str(idx)) or "NONE").strip()
                    sp = str(item.get("span") or "").strip().lower()
                    if sp and ch and ch.upper() != "NONE":
                        out[sp] = ch
        return out

    max_workers = max(1, min(len(batches), int(load_key_or_fn("max_workers", 4) or 4)))
    with ContextThreadPoolExecutor(max_workers=max_workers) as ex:
        for batch_out in ex.map(_run_batch, batches):
            decisions.update(batch_out)
    return decisions
