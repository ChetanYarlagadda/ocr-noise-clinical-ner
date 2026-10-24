"""Entity-level scoring (strict and overlap matching), micro P/R/F1, document bootstrap and Holm correction."""

from __future__ import annotations

from collections import Counter
from math import erf, sqrt

import numpy as np
import pandas as pd

from .data import TYPES, Entity

Gold = list[tuple[tuple[int, int] | None, str]]  # projected span (None = lost in OCR) and type


def doc_counts(gold: Gold, pred: list[Entity]) -> dict[str, int]:
    """True positives, false positives and false negatives per type under strict and overlap matching.

    Strict: same start, end and type. Overlap: same type and at least one shared character, matched one-to-one
    greedily in text order. A lost gold mention (None) is always a false negative.
    """
    out: dict[str, int] = {}
    for t in TYPES:
        g = [s for s, typ in gold if typ == t]
        p = sorted((e.start, e.end) for e in pred if e.type == t)
        kept = [s for s in g if s is not None]
        tp = sum((Counter(kept) & Counter(p)).values())
        out[f"strict_tp_{t}"], out[f"strict_fp_{t}"], out[f"strict_fn_{t}"] = tp, len(p) - tp, len(g) - tp
        used = [False] * len(kept)
        order = sorted(range(len(kept)), key=lambda k: kept[k])
        tp_o = 0
        for a, b in p:
            for k in order:
                if not used[k] and kept[k][0] < b and a < kept[k][1]:
                    used[k] = True
                    tp_o += 1
                    break
        out[f"overlap_tp_{t}"], out[f"overlap_fp_{t}"], out[f"overlap_fn_{t}"] = tp_o, len(p) - tp_o, len(g) - tp_o
        out[f"lost_{t}"] = len(g) - len(kept)
    return out


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts (0 when undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def count_columns(mode: str = "strict", etype: str | None = None) -> tuple[list[str], list[str], list[str]]:
    """Column names holding tp / fp / fn for a matching mode and an entity type (None = both types)."""
    types = TYPES if etype is None else (etype,)
    return tuple([f"{mode}_{k}_{t}" for t in types] for k in ("tp", "fp", "fn"))  # type: ignore[return-value]


def micro(df: pd.DataFrame, mode: str = "strict", etype: str | None = None) -> tuple[float, float, float]:
    """Micro-averaged precision, recall and F1 over the documents (rows) of ``df``."""
    tp, fp, fn = (df[c].to_numpy().sum() for c in count_columns(mode, etype))
    return prf(tp, fp, fn)


def _f1_from_sums(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def doc_arrays(df: pd.DataFrame, mode: str = "strict", etype: str | None = None) -> np.ndarray:
    """(documents x 3) array of tp, fp, fn."""
    return np.stack([df[c].to_numpy().sum(axis=1) for c in count_columns(mode, etype)], axis=1).astype(float)


def bootstrap_indices(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Resampling indices (n_boot x n), shared by every comparison so that paired tests line up."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_f1(arr: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of micro F1 when documents are resampled."""
    s = arr[idx].sum(axis=1)
    f = _f1_from_sums(s[:, 0], s[:, 1], s[:, 2])
    return float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def paired_diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    """Micro-F1 difference a - b on the same documents with a paired bootstrap CI and a two-sided p-value.

    The p-value uses a normal approximation with the bootstrap standard error of the difference.
    """
    sa, sb = a.sum(axis=0), b.sum(axis=0)
    diff = _f1_from_sums(*sa) - _f1_from_sums(*sb)
    ba, bb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    d = _f1_from_sums(ba[:, 0], ba[:, 1], ba[:, 2]) - _f1_from_sums(bb[:, 0], bb[:, 1], bb[:, 2])
    se = float(d.std(ddof=1))
    p = 1.0 if se == 0 else 2 * (1 - 0.5 * (1 + erf(abs(float(diff)) / se / sqrt(2))))
    return {"diff": float(diff), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "se": se, "p": float(p)}


def holm(p: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = running
    return adj
