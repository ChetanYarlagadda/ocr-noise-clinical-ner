"""Character error rate, word error rate and projection of gold spans from clean text onto OCR text.

Alignment is the minimum-edit-distance (Levenshtein) alignment between the clean reference and the hypothesis,
computed with RapidFuzz and checked against a plain dynamic-programming implementation in the tests.

Span projection rule. Every reference character is either kept (aligned to one hypothesis character, identical or
substituted) or deleted. A gold span [s, e) maps to the hypothesis span from the first to the last kept character
inside it; characters the OCR inserted *between* kept characters of the span are part of the projected span, those
inserted at its edges are not. Leading and trailing whitespace is trimmed. A span whose characters were all deleted
is *lost*: it still counts as a gold mention (a false negative for every system), because the information is gone.
"""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace runs (including line breaks) to one space and strip."""
    return _WS.sub(" ", text).strip()


def levenshtein(a: str | list, b: str | list) -> int:
    """Reference edit distance (unit costs) by dynamic programming; O(len(a) * len(b))."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate: edit distance divided by the reference length."""
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    """Word error rate: edit distance over whitespace tokens divided by the number of reference words."""
    r = ref.split()
    return Levenshtein.distance(r, hyp.split()) / max(1, len(r))


def char_map(ref: str, hyp: str) -> list[int]:
    """For every reference character, the index of the hypothesis character it is aligned to, or -1 if deleted."""
    out = [-1] * len(ref)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag in ("equal", "replace"):
            n, m = i2 - i1, j2 - j1
            for k in range(n):
                out[i1 + k] = j1 + (k if n == m else min(m - 1, k * m // n))
    return out


def project(spans: list[tuple[int, int]], cmap: list[int], hyp: str) -> list[tuple[int, int] | None]:
    """Project half-open reference spans onto the hypothesis with a ``char_map``; None marks a lost span."""
    out: list[tuple[int, int] | None] = []
    for s, e in spans:
        kept = [cmap[i] for i in range(s, e) if cmap[i] >= 0]
        if not kept:
            out.append(None)
            continue
        a, b = min(kept), max(kept) + 1
        while a < b and hyp[a].isspace():
            a += 1
        while b > a and hyp[b - 1].isspace():
            b -= 1
        out.append((a, b) if a < b else None)
    return out


def project_spans(ref: str, hyp: str, spans: list[tuple[int, int]]) -> list[tuple[int, int] | None]:
    """Align ``ref`` to ``hyp`` and project the spans (see the module docstring for the rule)."""
    return project(spans, char_map(ref, hyp), hyp)


def span_errors(ref: str, hyp: str, spans: list[tuple[int, int]],
                projected: list[tuple[int, int] | None]) -> tuple[int, int]:
    """Edit operations needed to turn every gold mention string into its projected OCR string, and their length.

    Their ratio is the character error rate restricted to the mentions (a lost mention costs its full length).
    """
    errors = total = 0
    for (s, e), p in zip(spans, projected):
        errors += Levenshtein.distance(ref[s:e], hyp[p[0]:p[1]] if p else "")
        total += e - s
    return errors, total
