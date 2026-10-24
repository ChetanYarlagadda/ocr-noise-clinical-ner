"""A character-level OCR noise channel learned from real OCR output, used to make noisy training text.

The channel is estimated from minimum-edit alignments of clean development abstracts and their OCR text: for every
reference character c it stores how often c was deleted or substituted (and by what), and after every reference
character how often a string was inserted (and which). Sampling from it at a scale factor produces noisy copies of
training documents whose gold spans are carried along exactly (same projection rule as ``align.project``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import Levenshtein

from .data import Entity


@dataclass
class Channel:
    """Per-character deletion / substitution counts and insertion counts, with sampling tables."""

    char_count: Counter = field(default_factory=Counter)
    deletions: Counter = field(default_factory=Counter)
    substitutions: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    insertions: Counter = field(default_factory=Counter)  # inserted string -> count
    n_ref: int = 0

    def p_del(self, c: str) -> float:
        n = self.char_count[c]
        return self.deletions[c] / n if n else self.total_rate("del")

    def p_sub(self, c: str) -> float:
        n = self.char_count[c]
        return sum(self.substitutions[c].values()) / n if n else self.total_rate("sub")

    def p_ins(self) -> float:
        return sum(self.insertions.values()) / max(1, self.n_ref)

    def total_rate(self, kind: str) -> float:
        if kind == "del":
            return sum(self.deletions.values()) / max(1, self.n_ref)
        return sum(sum(v.values()) for v in self.substitutions.values()) / max(1, self.n_ref)

    def expected_cer(self) -> float:
        """Edit operations per reference character at scale 1 (deletions + substitutions + inserted characters)."""
        ins_chars = sum(len(s) * n for s, n in self.insertions.items())
        return self.total_rate("del") + self.total_rate("sub") + ins_chars / max(1, self.n_ref)


def fit_channel(pairs: list[tuple[str, str]]) -> Channel:
    """Estimate the channel from (reference, OCR) text pairs via Levenshtein opcodes."""
    ch = Channel()
    for ref, hyp in pairs:
        ch.char_count.update(ref)
        ch.n_ref += len(ref)
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
            if tag == "delete":
                ch.deletions.update(ref[i1:i2])
            elif tag == "replace":
                for k in range(i2 - i1):
                    ch.substitutions[ref[i1 + k]][hyp[j1 + k] if j2 - j1 == i2 - i1 else hyp[j1:j2]] += 1
            elif tag == "insert":
                ch.insertions[hyp[j1:j2]] += 1
    return ch


def _sampler(counter: Counter) -> tuple[list[str], np.ndarray]:
    keys = list(counter)
    p = np.array([counter[k] for k in keys], dtype=float)
    return keys, p / p.sum()


def corrupt(text: str, entities: list[Entity], ch: Channel, scale: float,
            rng: np.random.Generator) -> tuple[str, list[Entity], list[int]]:
    """Sample a noisy copy of ``text`` at ``scale`` times the fitted rates.

    Returns the noisy text, the entities projected onto it (spans whose characters were all deleted are dropped)
    and the character map (reference index -> noisy index or -1), which tests compare with ``align.char_map``.
    """
    ins_keys, ins_p = _sampler(ch.insertions) if ch.insertions else ([], np.array([]))
    p_ins = min(0.5, scale * ch.p_ins()) if ins_keys else 0.0
    subs = {c: _sampler(v) for c, v in ch.substitutions.items() if v}
    out: list[str] = []
    cmap = []
    u = rng.random((len(text), 2))
    for i, c in enumerate(text):
        pd_, ps = min(0.9, scale * ch.p_del(c)), min(0.9, scale * ch.p_sub(c))
        if pd_ + ps > 0.95:
            pd_, ps = pd_ * 0.95 / (pd_ + ps), ps * 0.95 / (pd_ + ps)
        if u[i, 0] < pd_:
            cmap.append(-1)
        elif u[i, 0] < pd_ + ps and c in subs:
            keys, p = subs[c]
            rep = keys[rng.choice(len(keys), p=p)]
            cmap.append(len(out) if rep else -1)
            out.extend(rep)
        else:
            cmap.append(len(out))
            out.append(c)
        if u[i, 1] < p_ins:
            out.extend(ins_keys[rng.choice(len(ins_keys), p=ins_p)])
    noisy = "".join(out)
    ents = []
    for e in entities:
        kept = [cmap[i] for i in range(e.start, e.end) if cmap[i] >= 0]
        if not kept:
            continue
        a, b = min(kept), max(kept) + 1
        while a < b and noisy[a].isspace():
            a += 1
        while b > a and noisy[b - 1].isspace():
            b -= 1
        if a < b:
            ents.append(Entity(a, b, e.type))
    return noisy, ents, cmap


def channel_table(ch: Channel, top: int = 25) -> list[dict]:
    """The most frequent learned confusions (for reports)."""
    rows = [{"operation": "delete", "reference": c, "ocr": "", "count": n, "rate": n / ch.char_count[c]}
            for c, n in ch.deletions.items()]
    rows += [{"operation": "substitute", "reference": c, "ocr": d, "count": n, "rate": n / ch.char_count[c]}
             for c, v in ch.substitutions.items() for d, n in v.items()]
    rows += [{"operation": "insert", "reference": "", "ocr": s, "count": n, "rate": n / max(1, ch.n_ref)}
             for s, n in ch.insertions.items()]
    return sorted(rows, key=lambda r: -r["count"])[:top]
