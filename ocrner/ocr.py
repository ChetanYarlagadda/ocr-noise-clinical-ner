"""docTR OCR (DBNet ResNet-50 text detection + CRNN VGG-16 recognition) with a keyed on-disk cache of page texts.

docTR returns words grouped into lines and blocks; the page text is the reading-order rendering with every
whitespace run collapsed to one space, matching the whitespace-normalised reference text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .align import normalize

DET_ARCH, RECO_ARCH = "db_resnet50", "crnn_vgg16_bn"


class DocTR:
    """Batched docTR predictor in fp32 (half precision broke recognition in a probe: CER 41% on clean pages)."""

    def __init__(self, cache_dir: Path, device: str = "cuda", det_arch: str = DET_ARCH, reco_arch: str = RECO_ARCH,
                 batch: int = 8) -> None:
        os.environ.setdefault("DOCTR_CACHE_DIR", str(cache_dir))
        from doctr.models import ocr_predictor  # imported late so DOCTR_CACHE_DIR takes effect

        self.predictor = ocr_predictor(det_arch, reco_arch, pretrained=True, assume_straight_pages=True)
        if device == "cuda":
            self.predictor = self.predictor.cuda()
        self.batch = batch
        self.name = f"doctr:{det_arch}+{reco_arch}"

    def __call__(self, pages: list[np.ndarray]) -> list[str]:
        """OCR greyscale uint8 pages; returns one whitespace-normalised string per page."""
        out = []
        for i in range(0, len(pages), self.batch):
            rgb = [np.repeat(p[..., None], 3, axis=2) for p in pages[i:i + self.batch]]
            out += [normalize(page.render()) for page in self.predictor(rgb).pages]
        return out


class TextCache:
    """A JSON file of {document id: text} for one condition; written after every batch so runs can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def missing(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self.data]

    def update(self, new: dict[str, str]) -> None:
        self.data.update(new)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(self.path)


def ocr_condition(ids: list[str], make_page: Callable[[str], np.ndarray], cache: TextCache,
                  get_engine: Callable[[], DocTR], chunk: int = 16) -> dict[str, str]:
    """OCR texts for ``ids`` under one condition, computing only the pages missing from ``cache``."""
    todo = cache.missing(ids)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        texts = get_engine()([make_page(j) for j in part])
        cache.update(dict(zip(part, texts)))
    return {i: cache.data[i] for i in ids}
