"""BC5CDR corpus: pinned download, PubTator parsing and checks against the published corpus statistics.

The BioCreative V CDR corpus (Li et al., 2016) has 1,500 PubMed titles and abstracts (500 train / 500 development /
500 test) with every chemical and disease mention annotated as a character span. The archive is the one mirrored
by the BigBio project on the Hugging Face Hub, pinned by commit and SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

HF_REVISION = "6ba1463320d003e8232dfa0c3ee9aaa2559998ec"
URL = f"https://huggingface.co/datasets/bigbio/bc5cdr/resolve/{HF_REVISION}/CDR_Data.zip"
SHA256 = "0a359a7f038d283a7b05b084fa73de014e7410e1f5d7034bf3fd01f016fc2444"
ARCHIVE = "CDR_Data.zip"
SPLIT_FILES = {
    "train": "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt",
    "dev": "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt",
}
TOOL_FILES = {  # outputs of NCBI's own tools on the test set, shipped in the archive
    "tmChem": "CDR_Data/tmChem.TestSet/TestSet.tmChem.PubTator.txt",
    "DNorm": "CDR_Data/DNorm.TestSet/TestSet.DNorm.PubTator.txt",
}
TOOL_TYPE = {"tmChem": "Chemical", "DNorm": "Disease"}
# Concept-level (document-level MeSH ID) scores of the two tools as stated in the archive README.
TOOL_README = {"tmChem": (0.92, 0.90, 0.91), "DNorm": (0.81, 0.80, 0.81)}
TYPES = ("Chemical", "Disease")
LICENSE = "Public domain (United States Government Work, NCBI/NLM notice in the archive README)"

# Li et al. (2016), Database, baw068, Table 1: articles and mention counts per split.
PUBLISHED = {
    ("train", "articles"): 500, ("train", "Chemical"): 5203, ("train", "Disease"): 4182,
    ("dev", "articles"): 500, ("dev", "Chemical"): 5347, ("dev", "Disease"): 4244,
    ("test", "articles"): 500, ("test", "Chemical"): 5385, ("test", "Disease"): 4424,
}


@dataclass(frozen=True)
class Entity:
    """One annotated mention: half-open character span [start, end) and its type."""

    start: int
    end: int
    type: str


@dataclass
class Doc:
    """A title + abstract with its gold mentions (offsets into ``text``)."""

    pmid: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    title_end: int = 0  # the title is text[:title_end]


def download(raw_dir: Path) -> Path:
    """Download the corpus archive once into ``raw_dir`` and verify its SHA-256."""
    path = raw_dir / ARCHIVE
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {SHA256}")
    return path


def normalize_whitespace(text: str, entities: list[Entity]) -> tuple[str, list[Entity]]:
    """Collapse whitespace runs to one space and strip, remapping entity offsets accordingly."""
    new_pos = []  # new index of every old character position (and of len(text))
    out: list[str] = []
    prev_space = True  # drops leading whitespace
    for ch in text:
        new_pos.append(len(out))
        if ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    new_pos.append(len(out))
    if out and out[-1] == " ":
        out.pop()
    norm = "".join(out)
    ents = [Entity(new_pos[e.start], min(new_pos[e.end], len(norm)), e.type) for e in entities]
    return norm, ents


def parse_pubtator(raw: str) -> list[Doc]:
    """Parse a PubTator file: ``pmid|t|title``, ``pmid|a|abstract``, then tab-separated mention lines.

    The document text is ``title + " " + abstract`` (the PubTator offset convention). Relation lines (CID) are
    skipped. Every mention's text is checked against its span before whitespace is normalised.
    """
    docs = []
    for block in raw.strip().split("\n\n"):
        lines = block.strip("\n").split("\n")
        pmid, _, title = lines[0].partition("|t|")
        _, _, abstract = lines[1].partition("|a|")
        text = f"{title} {abstract}"
        ents = []
        for line in lines[2:]:
            f = line.split("\t")
            if len(f) < 5 or not f[1].isdigit():
                continue
            start, end, mention, etype = int(f[1]), int(f[2]), f[3], f[4]
            if text[start:end] != mention:
                raise ValueError(f"{pmid}: span {start}-{end} is {text[start:end]!r}, expected {mention!r}")
            ents.append(Entity(start, end, etype))
        text, ents = normalize_whitespace(text, sorted(ents, key=lambda e: (e.start, e.end)))
        docs.append(Doc(pmid, text, ents, len(" ".join(title.split()))))
    return docs


def load(raw_dir: Path) -> dict[str, list[Doc]]:
    """All three splits of the corpus, downloading the archive if needed."""
    path = download(raw_dir)
    with zipfile.ZipFile(path) as z:
        return {split: parse_pubtator(z.read(name).decode("utf-8")) for split, name in SPLIT_FILES.items()}


def crosscheck(splits: dict[str, list[Doc]]) -> pd.DataFrame:
    """Article and mention counts per split against Table 1 of Li et al. (2016)."""
    rows = []
    for split, docs in splits.items():
        counts = {"articles": len(docs)}
        for t in TYPES:
            counts[t] = sum(e.type == t for d in docs for e in d.entities)
        for what, n in counts.items():
            pub = PUBLISHED[(split, what)]
            rows.append({"split": split, "statistic": "articles" if what == "articles" else f"{what} mentions",
                         "published": pub, "loaded": n, "match": pub == n})
    return pd.DataFrame(rows)


def split_summary(splits: dict[str, list[Doc]]) -> pd.DataFrame:
    """Documents, words, characters and mentions per split, plus overlap and whitespace checks."""
    rows = []
    for split, docs in splits.items():
        words = [len(d.text.split()) for d in docs]
        overlaps = sum(b.start < a.end for d in docs for a, b in zip(d.entities, d.entities[1:]))
        rows.append({"split": split, "documents": len(docs), "words_mean": sum(words) / len(words),
                     "words_max": max(words), "characters": sum(len(d.text) for d in docs),
                     "chemical_mentions": sum(e.type == "Chemical" for d in docs for e in d.entities),
                     "disease_mentions": sum(e.type == "Disease" for d in docs for e in d.entities),
                     "overlapping_mentions": overlaps,
                     "non_ascii_characters": sum(bool(re.search(r"[^\x00-\x7f]", d.text)) for d in docs)})
    return pd.DataFrame(rows)


def read_member(raw_dir: Path, name: str) -> str:
    """One text file from the (verified) archive."""
    with zipfile.ZipFile(download(raw_dir)) as z:
        return z.read(name).decode("utf-8")


def parse_mentions(raw: str) -> pd.DataFrame:
    """Every mention line of a PubTator file with its concept IDs (prefix ``MESH:`` removed, ``-1`` dropped)."""
    rows = []
    for line in raw.splitlines():
        f = line.split("	")
        if len(f) >= 6 and f[1].isdigit():
            ids = [i.replace("MESH:", "") for i in re.split(r"[|;]", f[5]) if i and i != "-1"]
            rows.append({"pmid": f[0], "start": int(f[1]), "end": int(f[2]), "type": f[4], "ids": ids})
    return pd.DataFrame(rows)


def concept_scores(gold: pd.DataFrame, pred: pd.DataFrame, etype: str) -> tuple[float, float, float]:
    """Micro precision, recall and F1 of the per-document sets of concept IDs (the BC5CDR DNER measure)."""
    def sets(df: pd.DataFrame) -> set[tuple[str, str]]:
        return {(r.pmid, i) for r in df[df.type == etype].itertuples() for i in r.ids}

    g, p = sets(gold), sets(pred)
    tp = len(g & p)
    prec, rec = tp / max(1, len(p)), tp / max(1, len(g))
    return prec, rec, 2 * prec * rec / max(1e-12, prec + rec)
