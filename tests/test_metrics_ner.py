"""Entity scoring, bootstrap/Holm, and the BIO tagging helpers."""

import numpy as np
import pandas as pd
import pytest
from transformers import BertTokenizer

from ocrner.data import Entity
from ocrner.metrics import (
    bootstrap_f1, bootstrap_indices, doc_arrays, doc_counts, holm, micro, paired_diff, prf,
)
from ocrner.ner import bio, boundary_alignment, decode, encode, windows, words


def test_doc_counts_strict_overlap_and_lost():
    gold = [((0, 10), "Chemical"), ((20, 28), "Disease"), (None, "Chemical")]
    pred = [Entity(0, 10, "Chemical"), Entity(20, 25, "Disease"), Entity(40, 45, "Disease")]
    c = doc_counts(gold, pred)
    assert (c["strict_tp_Chemical"], c["strict_fp_Chemical"], c["strict_fn_Chemical"]) == (1, 0, 1)
    assert (c["strict_tp_Disease"], c["strict_fp_Disease"], c["strict_fn_Disease"]) == (0, 2, 1)
    assert (c["overlap_tp_Disease"], c["overlap_fp_Disease"], c["overlap_fn_Disease"]) == (1, 1, 0)
    assert c["lost_Chemical"] == 1
    # a type mismatch is never a match
    assert doc_counts([((0, 4), "Chemical")], [Entity(0, 4, "Disease")])["overlap_tp_Chemical"] == 0


def test_overlap_matching_is_one_to_one():
    gold = [((0, 10), "Disease")]
    pred = [Entity(0, 4, "Disease"), Entity(5, 10, "Disease")]
    c = doc_counts(gold, pred)
    assert (c["overlap_tp_Disease"], c["overlap_fp_Disease"]) == (1, 1)


def test_prf_micro_and_bootstrap():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(8, 2, 8)[2] == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    rows = [doc_counts([((0, 3), "Chemical")], [Entity(0, 3, "Chemical")]),
            doc_counts([((0, 3), "Chemical")], []), doc_counts([], [Entity(1, 2, "Disease")])]
    df = pd.DataFrame(rows)
    p, r, f = micro(df)
    assert (p, r) == (pytest.approx(0.5), pytest.approx(0.5))
    idx = bootstrap_indices(len(df), 200)
    lo, hi = bootstrap_f1(doc_arrays(df), idx)
    assert lo <= f <= hi


def test_paired_diff_and_holm():
    a = np.array([[5, 1, 1], [4, 0, 2], [6, 2, 0]], dtype=float)
    idx = bootstrap_indices(3, 100)
    r = paired_diff(a, a, idx)
    assert r["diff"] == 0 and r["p"] == 1.0
    better = a.copy()
    better[:, 0] += 1
    better[:, 2] -= 1
    assert paired_diff(better, a, idx)["diff"] > 0
    adj = holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_bio_decode_round_trip_and_boundaries():
    text = "Naloxone-induced hypotension in rats with low-dose aspirin."
    ents = [Entity(0, 8, "Chemical"), Entity(17, 28, "Disease"), Entity(51, 58, "Chemical")]
    spans = words(text)
    labels = bio(spans, ents)
    assert decode(spans, labels) == ents
    assert boundary_alignment(text, ents) == 1.0
    # a mention ending inside a word is not on a word boundary
    assert boundary_alignment(text, [Entity(0, 5, "Chemical")]) == 0.0


def test_windows_cover_every_word_and_respect_max():
    text = " ".join(f"word{i}" + (" ." if i % 7 == 6 else "") for i in range(100))
    spans = words(text)
    w = windows(spans, text, max_words=20)
    assert w[0][0] == 0 and w[-1][1] == len(spans)
    assert all(b - a <= 20 for a, b in w)
    assert all(w[k][1] == w[k + 1][0] for k in range(len(w) - 1))
    # cuts fall right after a full stop when one is available
    assert all(text[spans[b - 1][0]:spans[b - 1][1]] == "." for a, b in w[:-1])


def test_encode_labels_first_subword_only():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "low", "dose", "as", "##pi", "##rin", "-", "."]
    tok = BertTokenizer(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=True)
    text = "low-dose aspirin."
    ex = encode(tok, text, [Entity(9, 16, "Chemical")], max_words=50, max_length=32)[0]
    pieces = tok.convert_ids_to_tokens(ex["input_ids"])
    assert pieces == ["[CLS]", "low", "-", "dose", "as", "##pi", "##rin", ".", "[SEP]"]
    assert ex["labels"] == [-100, 0, 0, 0, 1, -100, -100, 0, -100]
    assert ex["first"][4] == 3  # word index of "aspirin"
