"""Edit distance, CER/WER and span projection on synthetic corruptions with known answers."""

from collections import Counter

import numpy as np
import pytest
from rapidfuzz.distance import Levenshtein

from ocrner.align import cer, char_map, levenshtein, normalize, project, project_spans, span_errors, wer
from ocrner.data import Entity
from ocrner.noise import Channel, corrupt, fit_channel


def test_levenshtein_matches_rapidfuzz_on_random_strings():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = "".join(rng.choice(list("ab c."), size=rng.integers(0, 15)))
        b = "".join(rng.choice(list("ab c."), size=rng.integers(0, 15)))
        assert levenshtein(a, b) == Levenshtein.distance(a, b)


def test_cer_and_wer_known_values():
    assert cer("aspirin", "aspirin") == 0
    assert cer("aspirin", "asprin") == pytest.approx(1 / 7)  # one deletion
    assert cer("abc", "xyzw") == pytest.approx(4 / 3)  # CER can exceed 1 with insertions
    assert wer("low dose aspirin", "low dosc aspirin") == pytest.approx(1 / 3)
    assert wer("low dose aspirin", "lowdose aspirin") == pytest.approx(2 / 3)  # merge = 1 sub + 1 del
    assert normalize("  a \n b\t\tc ") == "a b c"


def test_projection_on_hand_made_corruptions():
    ref = "Famotidine causes delirium in rats."
    fam, deli = (0, 10), (18, 26)
    # substitution inside a mention keeps its span
    assert project_spans(ref, ref.replace("delirium", "de1irium"), [deli]) == [(18, 26)]
    # a deleted word shifts the following span
    hyp = ref.replace("causes ", "")
    assert project_spans(ref, hyp, [fam, deli]) == [(0, 10), (11, 19)]
    # characters inserted inside the span belong to it, those at the edge do not
    assert project_spans(ref, "Famoti dine causes delirium in rats.", [fam]) == [(0, 11)]
    assert project_spans(ref, "Famotidine; causes delirium in rats.", [fam]) == [(0, 10)]
    # a mention whose characters are all deleted is lost
    assert project_spans(ref, "causes delirium in rats.", [fam]) == [None]
    # merged words: the span covers the surviving characters only
    hyp = "Famotidine causesdelirium in rats."
    s = project_spans(ref, hyp, [deli])[0]
    assert hyp[s[0]:s[1]] == "delirium"


def test_span_errors_count_only_mention_characters():
    ref, hyp = "aspirin and heparin", "aspirln and heparin"
    spans = [(0, 7), (12, 19)]
    proj = project_spans(ref, hyp, spans)
    assert span_errors(ref, hyp, spans, proj) == (1, 14)
    assert span_errors(ref, "aspirin and", spans, project_spans(ref, "aspirin and", spans)) == (7, 14)


def _toy_channel(substitute_only: bool) -> Channel:
    ch = Channel()
    text = "abcdefghij klmnopqrstuvwxyz"
    ch.char_count.update(text * 10)
    ch.n_ref = len(text) * 10
    for c in text:
        ch.substitutions[c] = Counter({"#": 2})
        if not substitute_only:
            ch.deletions[c] = 1
    if not substitute_only:
        ch.insertions.update({"@": 3, "%%": 1})
    return ch


def test_alignment_recovers_generator_spans_exactly_for_substitutions():
    rng = np.random.default_rng(1)
    text = "the quick brown fox jumps over the lazy dog " * 4
    ents = [Entity(4, 9, "Chemical"), Entity(20, 25, "Disease"), Entity(100, 115, "Chemical")]
    ch = _toy_channel(substitute_only=True)
    for _ in range(20):
        noisy, tracked, cmap = corrupt(text, ents, ch, scale=1.0, rng=rng)
        assert len(noisy) == len(text)
        aligned = project_spans(text, noisy, [(e.start, e.end) for e in ents])
        assert [(e.start, e.end) for e in tracked] == [s for s in aligned if s is not None]
        assert cmap == char_map(text, noisy) == list(range(len(text)))


def test_alignment_mostly_agrees_with_generator_under_mixed_noise():
    rng = np.random.default_rng(2)
    text = "patients treated with lamotrigine developed myoclonic jerks and status epilepticus " * 3
    ents = [Entity(m, m + 11, "Chemical") for m in range(22, len(text), 84)]
    ch = _toy_channel(substitute_only=False)
    agree = total = 0
    for _ in range(30):
        noisy, _, cmap = corrupt(text, ents, ch, scale=1.0, rng=rng)
        spans = [(e.start, e.end) for e in ents]
        a, b = project(spans, cmap, noisy), project_spans(text, noisy, spans)
        agree += sum(x == y for x, y in zip(a, b))
        total += len(spans)
    assert agree / total > 0.8  # alignment is ambiguous only where edits sit next to mention edges


def test_fit_channel_counts_operations():
    ch = fit_channel([("abcd", "abxd"), ("abcd", "abd"), ("ab", "ab!")])
    assert ch.substitutions["c"]["x"] == 1
    assert ch.deletions["c"] == 1
    assert ch.insertions["!"] == 1
    assert ch.expected_cer() == pytest.approx(3 / 10)
