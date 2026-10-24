"""Score NER output on OCR text against gold mentions projected from the clean text."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .align import char_map, project, span_errors
from .data import TYPES, Doc, Entity
from .metrics import Gold, bootstrap_f1, doc_arrays, doc_counts, micro


FATES = ("intact", "merged", "corrupted", "lost")


def fate(doc: Doc, hyp: str, p: tuple[int, int] | None, e: Entity) -> str:
    """What OCR did to one gold mention.

    intact: same characters and still a separate word; merged: same characters but glued to a neighbouring word
    (a lost space or hyphen); corrupted: at least one character changed; lost: every character deleted.
    """
    if p is None:
        return "lost"
    if hyp[p[0]:p[1]] != doc.text[e.start:e.end]:
        return "corrupted"
    glued_left = p[0] > 0 and hyp[p[0] - 1].isalnum() and not (e.start > 0 and doc.text[e.start - 1].isalnum())
    glued_right = (p[1] < len(hyp) and hyp[p[1]].isalnum()
                   and not (e.end < len(doc.text) and doc.text[e.end].isalnum()))
    return "merged" if glued_left or glued_right else "intact"


def project_gold(doc: Doc, hyp: str) -> tuple[Gold, dict[str, float]]:
    """Gold mentions projected onto ``hyp`` and the document's OCR quality (CER, WER, mention CER, mention fates)."""
    spans = [(e.start, e.end) for e in doc.entities]
    proj = project(spans, char_map(doc.text, hyp), hyp)
    gold = [(p, e.type) for p, e in zip(proj, doc.entities)]
    fates = [fate(doc, hyp, p, e) for p, e in zip(proj, doc.entities)]
    ref_words = doc.text.split()
    m_err, m_chars = span_errors(doc.text, hyp, spans, proj)
    quality = {"chars": len(doc.text), "char_errors": Levenshtein.distance(doc.text, hyp),
               "words": len(ref_words), "word_errors": Levenshtein.distance(ref_words, hyp.split()),
               "mention_chars": m_chars, "mention_char_errors": m_err,
               "mentions": len(spans), **{f"mentions_{f}": fates.count(f) for f in FATES}}
    quality["cer"] = quality["char_errors"] / max(1, quality["chars"])
    quality["wer"] = quality["word_errors"] / max(1, quality["words"])
    return gold, quality


def mention_status(doc: Doc, hyp: str, gold: Gold, pred: list[Entity]) -> list[dict]:
    """Per gold mention: its fate in the OCR text and whether the tagger found it (exactly, or overlapping)."""
    exact = {(e.start, e.end, e.type) for e in pred}
    rows = []
    for (p, t), e in zip(gold, doc.entities):
        strict = p is not None and (p[0], p[1], t) in exact
        overlap = p is not None and any(q.type == t and q.start < p[1] and p[0] < q.end for q in pred)
        rows.append({"type": t, "status": fate(doc, hyp, p, e), "found_strict": strict, "found_overlap": overlap})
    return rows


def score_condition(docs: list[Doc], hyps: list[str], preds: list[list[Entity]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-document counts and OCR quality, and per-mention status rows, for one (condition, model)."""
    rows, mentions = [], []
    for doc, hyp, pred in zip(docs, hyps, preds):
        gold, quality = project_gold(doc, hyp)
        rows.append({"pmid": doc.pmid, **quality, **doc_counts(gold, pred)})
        mentions += [{"pmid": doc.pmid, **m} for m in mention_status(doc, hyp, gold, pred)]
    return pd.DataFrame(rows), pd.DataFrame(mentions)


def summarize(df: pd.DataFrame, idx: np.ndarray) -> dict[str, float]:
    """OCR quality and micro P/R/F1 (strict and overlap, per type) with bootstrap CIs for one condition."""
    out = {"documents": len(df), "cer": df.char_errors.sum() / df.chars.sum(), "cer_doc_mean": df.cer.mean(),
           "cer_doc_median": df.cer.median(), "wer": df.word_errors.sum() / df.words.sum(),
           "mention_cer": df.mention_char_errors.sum() / max(1, df.mention_chars.sum()),
           "mentions": int(df.mentions.sum()),
           **{f"{f}_share": df[f"mentions_{f}"].sum() / max(1, df.mentions.sum()) for f in FATES}}
    for mode in ("strict", "overlap"):
        p, r, f = micro(df, mode)
        lo, hi = bootstrap_f1(doc_arrays(df, mode), idx)
        out.update({f"{mode}_precision": p, f"{mode}_recall": r, f"{mode}_f1": f, f"{mode}_f1_low": lo,
                    f"{mode}_f1_high": hi})
        for t in TYPES:
            out[f"{mode}_f1_{t.lower()}"] = micro(df, mode, t)[2]
    return out
