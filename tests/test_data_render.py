"""PubTator parsing, whitespace normalisation, concept scoring, page rendering and deskew."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ocrner.data import Entity, concept_scores, normalize_whitespace, parse_mentions, parse_pubtator
from ocrner.preprocess import apply, estimate_skew
from ocrner.render import LEVELS, MASTER_DPI, PAPER, degrade, render_page, scanned_page

PUBTATOR = (
    "1|t|Lithium  toxicity.\n"
    "1|a|A newborn with lithium-induced flutter.\n"
    "1\t0\t7\tLithium\tChemical\tD008094\n"
    "1\t34\t41\tlithium\tChemical\tD008094\n"
    "1\t9\t17\ttoxicity\tDisease\t-1\n"
    "1\tCID\tD008094\tD005414\n\n"
    "2|t|Title two.\n"
    "2|a|Body.\n"
)


def test_parse_pubtator_normalises_whitespace_and_keeps_spans():
    docs = parse_pubtator(PUBTATOR)
    assert [d.pmid for d in docs] == ["1", "2"]
    d = docs[0]
    assert d.text == "Lithium toxicity. A newborn with lithium-induced flutter."
    assert d.title_end == len("Lithium toxicity.")
    assert [d.text[e.start:e.end] for e in d.entities] == ["Lithium", "toxicity", "lithium"]
    assert docs[1].entities == []


def test_parse_pubtator_rejects_wrong_spans():
    with pytest.raises(ValueError):
        parse_pubtator("1|t|Abc.\n1|a|Def.\n1\t0\t3\tXyz\tChemical\tD1\n")


def test_normalize_whitespace_remaps_offsets():
    text, ents = normalize_whitespace("  a  bc\n\nd ", [Entity(5, 7, "Chemical"), Entity(9, 10, "Disease")])
    assert text == "a bc d"
    assert [text[e.start:e.end] for e in ents] == ["bc", "d"]


def test_concept_scores_on_document_sets():
    gold = parse_mentions("1\t0\t3\tabc\tDisease\tD1|D2\n2\t0\t3\tabc\tDisease\tD3\n2\t5\t8\txyz\tDisease\t-1\n")
    pred = parse_mentions("1\t0\t3\tabc\tDisease\tMESH:D1\n1\t9\t12\tabc\tDisease\tMESH:D1\n2\t0\t3\tabc\tDisease\tD9\n")
    assert isinstance(gold, pd.DataFrame) and gold.ids.iloc[0] == ["D1", "D2"]
    p, r, f = concept_scores(gold, pred, "Disease")
    assert (p, r) == (pytest.approx(1 / 2), pytest.approx(1 / 3))


def test_render_and_degrade_are_deterministic():
    text = "A short title. The abstract body follows with several words to wrap across lines " * 3
    page = render_page(text, 14)
    assert page.mode == "L" and page.width == int(8.5 * MASTER_DPI)
    a = np.asarray(scanned_page(text, 14, 3, seed=11))
    b = np.asarray(scanned_page(text, 14, 3, seed=11))
    c = np.asarray(scanned_page(text, 14, 3, seed=12))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape[1] == pytest.approx(page.width * LEVELS[3].dpi / MASTER_DPI, abs=40)
    assert sorted(LEVELS) == list(range(5)) and [LEVELS[k].dpi for k in range(5)] == sorted(
        (LEVELS[k].dpi for k in range(5)), reverse=True)
    clean = degrade(page, LEVELS[0], np.random.default_rng(0))
    assert np.array_equal(np.asarray(clean), np.asarray(page))


@pytest.mark.parametrize("angle", [-1.4, 0.0, 0.9])
def test_deskew_recovers_a_known_rotation(angle):
    text = "Lamotrigine associated with exacerbation of myoclonus in generalized epilepsies. " * 8
    page = render_page(text, 40)
    page = page.resize((850, page.height * 850 // page.width), Image.Resampling.BOX)
    rotated = page.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=PAPER)
    est = estimate_skew(np.asarray(rotated))
    assert est == pytest.approx(angle, abs=0.25)
    assert apply(np.asarray(rotated), ("deskew",)).shape == np.asarray(rotated).shape
