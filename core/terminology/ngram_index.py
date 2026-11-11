from __future__ import annotations

import math
from typing import Any

from .lexical import (
    _KB_STOP_TOKENS,
    _is_standard_english,
    _phonetic_skeleton,
    _tokenize_surface,
)


def _char_phonetic_ngrams(text: str) -> list[str]:
    bw, _, _ = _tokenize_surface(text)
    if not bw:
        return []
    joined = "".join(bw)
    skel = "".join(_phonetic_skeleton(w) for w in bw)
    grams: list[str] = []
    padded = f"^{joined}$"
    for n in (2, 3):
        for i in range(len(padded) - n + 1):
            grams.append(f"c:{padded[i : i + n]}")
    if skel:
        ps = f"^{skel}$"
        for n in range(2, min(4, len(ps) + 1)):
            for i in range(len(ps) - n + 1):
                grams.append(f"p:{ps[i : i + n]}")
        for w in bw:
            pw = _phonetic_skeleton(w)
            if len(pw) >= 2:
                grams.append(f"pw:{pw}")
                grams.append(f"p2:{pw[:2]}")
    return grams


class UnifiedNgramKBIndex:
    """Indexes canonical surfaces, curated aliases, and unique proper-noun head tokens (df <= 3)."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.surfaces: list[tuple[str, dict[str, Any], float]] = []
        self._surf_meta: list[tuple[float, float, str, dict[str, Any]]] = []
        self._query_cache: dict[
            tuple[str, int, float], list[tuple[float, dict[str, Any]]]
        ] = {}
        tok_df: dict[str, int] = {}
        tokenized_entries: list[tuple[dict[str, Any], list[str]]] = []

        for entry in entries:
            src = str(entry.get("src") or "").strip()
            if not src:
                continue
            bw, _, _ = _tokenize_surface(src)
            tokenized_entries.append((entry, bw))
            for w in set(bw):
                if len(w) >= 4 and w not in _KB_STOP_TOKENS:
                    tok_df[w] = tok_df.get(w, 0) + 1

        for entry, bw in tokenized_entries:
            src = entry["src"]
            penalty = 0.92 if ("(" in src or ":" in src) else 1.0
            self.surfaces.append((src, entry, penalty))
            for al in entry.get("aliases") or []:
                al_s = str(al).strip()
                if al_s:
                    self.surfaces.append((al_s, entry, penalty * 0.98))
            if len(bw) >= 2 and "(" not in src and ":" not in src:
                for w in bw:
                    if (
                        len(w) >= 5
                        and w not in _KB_STOP_TOKENS
                        and tok_df.get(w, 99) <= 3
                        and not _is_standard_english(w)
                    ):
                        self.surfaces.append((w, entry, 0.94))

        df: dict[str, int] = {}
        raw_grams_per_surf: list[list[str]] = []
        for surf_text, _, _ in self.surfaces:
            g = _char_phonetic_ngrams(surf_text)
            raw_grams_per_surf.append(g)
            for tok in set(g):
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(1, len(self.surfaces))
        self.idf = {
            tok: math.log(1.0 + n_docs / (1.0 + freq)) for tok, freq in df.items()
        }
        self.surf_norms: list[float] = []
        self._surf_vecs: list[dict[str, float]] = []
        self.inv: dict[str, list[tuple[int, float]]] = {}
        self._tok_max_scaled: dict[str, float] = {}

        for surf_id, g_list in enumerate(raw_grams_per_surf):
            _, entry, penalty = self.surfaces[surf_id]
            sl = entry["src"].lower()
            if not g_list:
                self.surf_norms.append(0.0)
                self._surf_vecs.append({})
                self._surf_meta.append((0.0, penalty, sl, entry))
                continue
            tf: dict[str, int] = {}
            for tok in g_list:
                tf[tok] = tf.get(tok, 0) + 1
            vec = {
                tok: (1.0 + math.log(cnt)) * self.idf[tok] for tok, cnt in tf.items()
            }
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.surf_norms.append(norm)
            self._surf_vecs.append(vec)
            self._surf_meta.append((norm, penalty, sl, entry))
            inv_norm_pen = (penalty / norm) if norm > 0.0 else 0.0
            for tok, w in vec.items():
                self.inv.setdefault(tok, []).append((surf_id, w))
                scaled = w * inv_norm_pen
                if scaled > self._tok_max_scaled.get(tok, 0.0):
                    self._tok_max_scaled[tok] = scaled

    def query(
        self, span: str, top_k: int = 3, min_score: float = 0.45
    ) -> list[tuple[float, dict[str, Any]]]:
        ck = (span, top_k, min_score)
        cached = self._query_cache.get(ck)
        if cached is not None:
            return cached

        g_list = _char_phonetic_ngrams(span)
        if not g_list:
            return []
        tf: dict[str, int] = {}
        for tok in g_list:
            tf[tok] = tf.get(tok, 0) + 1
        q_vec = {
            tok: (1.0 + math.log(cnt)) * self.idf.get(tok, 1.0)
            for tok, cnt in tf.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []

        inv = self.inv
        tok_max = self._tok_max_scaled
        inv_q_norm = 1.0 / q_norm
        q_terms = [
            (qw * tok_max[tok] * inv_q_norm, tok, qw)
            for tok, qw in q_vec.items()
            if tok in inv
        ]
        if not q_terms:
            return []
        q_terms.sort(key=lambda x: x[0], reverse=True)

        m = len(q_terms)
        rem_ub = [0.0] * (m + 1)
        for idx in range(m - 1, -1, -1):
            rem_ub[idx] = rem_ub[idx + 1] + q_terms[idx][0]

        if rem_ub[0] + 1e-9 < min_score:
            if len(self._query_cache) < 8192:
                self._query_cache[ck] = []
            return []

        surf_meta = self._surf_meta
        surf_vecs = self._surf_vecs
        dots: dict[int, float] = {}
        split_idx = m
        for idx, (_, tok, qw) in enumerate(q_terms):
            if idx > 0 and rem_ub[idx] + 1e-9 < min_score:
                split_idx = idx
                break
            for surf_id, dw in inv[tok]:
                dots[surf_id] = dots.get(surf_id, 0.0) + qw * dw

        best_by_src: dict[str, tuple[float, dict[str, Any]]] = {}
        tail_terms = q_terms[split_idx:]
        rem_tail = rem_ub[split_idx] + 1e-9
        for surf_id, dot in dots.items():
            d_norm, penalty, sl, entry = surf_meta[surf_id]
            if d_norm <= 0.0:
                continue
            scale = penalty / (q_norm * d_norm)
            if tail_terms:
                if dot * scale + rem_tail < min_score:
                    continue
                s_vec = surf_vecs[surf_id]
                for _, tok, qw in tail_terms:
                    dw = s_vec.get(tok)
                    if dw is not None:
                        dot += qw * dw
            sc = dot * scale
            if sc < min_score:
                continue
            prev = best_by_src.get(sl)
            if prev is None or sc > prev[0]:
                best_by_src[sl] = (sc, entry)

        scored = sorted(best_by_src.values(), key=lambda x: x[0], reverse=True)[:top_k]
        if len(self._query_cache) < 8192:
            self._query_cache[ck] = scored
        return scored
