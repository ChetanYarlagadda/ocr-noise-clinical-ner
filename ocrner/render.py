"""Render an abstract as a synthetic scanned page and degrade it in a controlled way.

A page is typeset at a 300 dpi master resolution (DejaVu Serif, the font bundled with matplotlib, 11 pt body and a
12 pt bold title, 6.5 inch text width), then passed through a scanner model: rotation (skew), optical blur,
resampling to the scan resolution, sensor noise, speckle and JPEG compression. Each degradation level fixes these
parameters; the skew sign and size and the noise are drawn per document from a seeded generator.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MASTER_DPI = 300
PAPER, INK = 246, 28
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


@dataclass(frozen=True)
class Degradation:
    """Scanner settings of one degradation level (blur in master pixels, noise in grey levels)."""

    dpi: int
    blur: float = 0.0
    noise: float = 0.0
    jpeg: int | None = None
    skew: float = 0.0  # maximum absolute rotation in degrees
    speckle: float = 0.0  # share of pixels flipped to black or white


LEVELS: dict[int, Degradation] = {
    0: Degradation(dpi=300),
    1: Degradation(dpi=85, blur=1.6, noise=16, jpeg=50, skew=0.8, speckle=0.001),
    2: Degradation(dpi=78, blur=1.8, noise=18, jpeg=40, skew=1.0, speckle=0.002),
    3: Degradation(dpi=72, blur=2.0, noise=20, jpeg=35, skew=1.2, speckle=0.0025),
    4: Degradation(dpi=66, blur=2.2, noise=24, jpeg=30, skew=1.5, speckle=0.003),
}


def level_table() -> list[dict]:
    """The degradation ladder as rows (for reports)."""
    return [{"level": k, **asdict(v)} for k, v in LEVELS.items()]


def _font(bold: bool, pt: float, dpi: int) -> ImageFont.FreeTypeFont:
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size=round(pt * dpi / 72))


def wrap(text: str, font: ImageFont.FreeTypeFont, width_px: int) -> list[str]:
    """Greedy word wrap at spaces (no hyphenation), so every word stays on one line."""
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if line and font.getlength(trial) > width_px:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def render_page(text: str, title_end: int, body_pt: float = 11.0, title_pt: float = 12.0) -> Image.Image:
    """Typeset ``text`` (title = ``text[:title_end]``) on a white page at the master resolution (greyscale)."""
    dpi = MASTER_DPI
    margin, width = int(1.0 * dpi), int(6.5 * dpi)
    title_font, body_font = _font(True, title_pt, dpi), _font(False, body_pt, dpi)
    blocks = [(wrap(text[:title_end].strip(), title_font, width), title_font, title_pt),
              (wrap(text[title_end:].strip(), body_font, width), body_font, body_pt)]
    line_h = {id(f): round(pt * dpi / 72 * 1.3) for _, f, pt in blocks}
    gap = round(body_pt * dpi / 72)
    height = int(0.75 * dpi) * 2 + sum(len(ls) * line_h[id(f)] for ls, f, _ in blocks) + gap
    img = Image.new("L", (int(8.5 * dpi), height), PAPER)
    draw = ImageDraw.Draw(img)
    y = int(0.75 * dpi)
    for i, (lines, font, _) in enumerate(blocks):
        for line in lines:
            draw.text((margin, y), line, fill=INK, font=font)
            y += line_h[id(font)]
        if i == 0:
            y += gap
    return img


def degrade(img: Image.Image, d: Degradation, rng: np.random.Generator) -> Image.Image:
    """Apply skew, blur, down-sampling to ``d.dpi``, noise, speckle and JPEG to a master-resolution page."""
    out = img
    if d.skew > 0:
        angle = float(rng.uniform(0.5, 1.0) * d.skew * rng.choice([-1, 1]))
        out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=PAPER)
    if d.blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(d.blur))
    if d.dpi != MASTER_DPI:
        scale = d.dpi / MASTER_DPI
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.Resampling.BOX)
    arr = np.asarray(out, dtype=np.float32)
    if d.noise > 0:
        arr = arr + rng.normal(0.0, d.noise, arr.shape)
    if d.speckle > 0:
        u = rng.random(arr.shape)
        arr[u < d.speckle / 2] = INK
        arr[u > 1 - d.speckle / 2] = 255
    out = Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))
    if d.jpeg is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=d.jpeg)
        out = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return out


def scanned_page(text: str, title_end: int, level: int, seed: int) -> Image.Image:
    """Render and degrade one document at one level; the same (seed, level) always gives the same image."""
    rng = np.random.default_rng([seed, level])
    return degrade(render_page(text, title_end), LEVELS[level], rng)
