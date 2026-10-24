"""Redraw every figure from reports/results/ without re-running the experiments.

    python scripts/make_figures.py            # report figures
    python scripts/make_figures.py --smoke    # figures of the smoke run in data/cache/smoke/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ocrner.figures import make_all  # noqa: E402

if __name__ == "__main__":
    base = ROOT / "data" / "cache" / "smoke" if "--smoke" in sys.argv else ROOT
    for p in make_all(base / "reports" / "results", base / "reports" / "figures"):
        print("wrote", p.relative_to(ROOT))
