"""Classical OpenCV clean-up of scanned pages before OCR: deskew, upscaling, denoising and binarisation.

Each step is a function from a greyscale uint8 image to a greyscale uint8 image; a pipeline is a tuple of step
names applied in order. The pipeline used on the test pages is chosen on development pages (see run_pipeline.py).
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

PAPER = 246


def estimate_skew(gray: np.ndarray, max_angle: float = 4.0, step: float = 0.1, width: int = 800) -> float:
    """Skew angle in degrees (counter-clockwise positive, as in PIL.Image.rotate) by projection-profile search.

    The page is binarised (Otsu, ink = 1) and down-scaled; for each candidate angle the ink is rotated back and the
    score is the sum of squared differences between consecutive row sums, which peaks when text lines are level.
    """
    scale = min(1.0, width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink.astype(np.float32)
    h, w = ink.shape
    centre = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        m = cv2.getRotationMatrix2D(centre, -angle, 1.0)
        rows = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1)
        score = float(np.sum(np.diff(rows) ** 2))
        if score > best_score:
            best, best_score = float(angle), score
    return round(best, 2)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the page so that its text lines are horizontal (paper-coloured fill)."""
    angle = estimate_skew(gray)
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=PAPER)


def upscale(gray: np.ndarray, min_width: int = 1700) -> np.ndarray:
    """Enlarge low-resolution pages (bicubic) to at least ``min_width`` pixels, about 200 dpi for letter width."""
    if gray.shape[1] >= min_width:
        return gray
    f = min_width / gray.shape[1]
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def median(gray: np.ndarray) -> np.ndarray:
    """3 x 3 median filter (removes speckle and some sensor noise)."""
    return cv2.medianBlur(gray, 3)


def nlmeans(gray: np.ndarray) -> np.ndarray:
    """Non-local-means denoising with strength scaled to the estimated noise level."""
    return cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)


def otsu(gray: np.ndarray) -> np.ndarray:
    """Global Otsu binarisation to black ink on white paper."""
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


STEPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "deskew": deskew, "upscale": upscale, "median": median, "nlmeans": nlmeans, "otsu": otsu,
}


def apply(gray: np.ndarray, pipeline: tuple[str, ...]) -> np.ndarray:
    """Run the named steps in order on a greyscale page."""
    for name in pipeline:
        gray = STEPS[name](gray)
    return gray
