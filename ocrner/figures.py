"""Every report figure, drawn from the tables in reports/results/ (no recomputation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plotting import INK_2, MUTED, SERIES, SYSTEM_COLOR, TYPE_COLOR, apply_style, plt, save

FATE_COLOR = {"intact": SERIES[2], "merged": SERIES[3], "corrupted": SERIES[7], "lost": INK_2}
LEVEL_COLOR = {0: SERIES[2], 3: SERIES[3], 4: SERIES[6]}


def _pct(x: float) -> str:
    return f"{100 * x:.2g}%" if x < 0.01 else f"{100 * x:.1f}%"


def _pct_axis(ax, which: str = "x") -> None:
    fmt = plt.FuncFormatter(lambda v, _: f"{100 * v:g}%")
    (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(fmt)


def fig_pages(res: Path, out: Path) -> Path:
    """Crops of one abstract at every degradation level and after preprocessing."""
    z = np.load(res / "page_examples.npz")
    names = list(z["names"])
    lv = pd.read_csv(res / "degradation_levels.csv").set_index("level")
    docs = pd.read_csv(res / "doc_results.csv", dtype={"pmid": str})
    docs = docs[(docs.model == "clean-trained") & (docs.pmid == str(z["pmid"]))].set_index("condition")
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 1.75 * rows + 0.5))
    for ax, (i, name) in zip(axes.flat, enumerate(names)):
        ax.imshow(z[f"page{i}"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        level = int(name.split()[0][1:])
        label = f"Level {level}: {lv.loc[level, 'dpi']} dpi" + (" + OpenCV clean-up" if "preprocessing" in name else "")
        cer = docs.loc[name, "cer"] if name in docs.index else np.nan
        ax.set_title(f"{label}  |  CER of this page {_pct(cer)}", loc="left", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("One test abstract at the five scan levels (pixels shown 1:1): a reader copes with all of them, OCR "
                 "does not", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out / "00_pages.png"
    save(fig, path)
    return path


def fig_f1_vs_cer(res: Path, out: Path) -> Path:
    """Hero: entity F1 against character error rate, binned over every raw-OCR page, and by level."""
    bins = pd.read_csv(res / "f1_vs_cer_bins.csv")
    docs = pd.read_csv(res / "doc_results.csv")
    summ = pd.read_csv(res / "condition_summary.csv")
    llm = pd.read_csv(res / "llm_summary.csv")
    slope = pd.read_csv(res / "f1_cer_slope.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    raw = docs[(docs.model == "clean-trained") & docs.condition.str.fullmatch(r"L\d")]
    tp = raw.filter(like="strict_tp_").sum(1)
    f1_doc = 2 * tp / (2 * tp + raw.filter(like="strict_fp_").sum(1) + raw.filter(like="strict_fn_").sum(1)).clip(lower=1)
    ax.scatter(raw.cer.clip(lower=1e-4), f1_doc, s=9, color=MUTED, alpha=0.35, lw=0,
               label="one page, clean-trained tagger")
    for model in ("clean-trained", "noise-augmented"):
        b = bins[(bins.model == model) & (bins.metric == "strict_f1")]
        ax.errorbar(b.cer.clip(lower=1e-4), b.f1, yerr=[b.f1 - b.f1_low, b.f1_high - b.f1], color=SYSTEM_COLOR[model],
                    marker="o", capsize=3, label=f"{model} tagger, pages binned by CER (95% CI)")
    clean = summ[(summ.condition == "clean text") & (summ.model == "clean-trained")].iloc[0]
    ax.axhline(clean.strict_f1, color=INK_2, ls="--", lw=1.2, label=f"clean text, no OCR (F1 {clean.strict_f1:.3f})")
    ax.set_xscale("log")
    ax.set_xlim(8e-5, 0.6)
    ax.set_ylim(0, 1.0)
    _pct_axis(ax)
    ax.set_xlabel("character error rate of the page (log scale; CER 0 drawn at 0.01%)")
    ax.set_ylabel("entity F1 (exact span and type)")
    ax.legend(loc="lower left", fontsize=8.5)
    s = slope.loc["clean-trained"]
    ax.set_title(f"Below 10% CER, each point of CER costs {abs(100 * s.f1_per_cer_point):.1f} points of entity F1",
                 loc="left")

    ax = axes[1]
    lev = summ[summ.variant == "none"].sort_values("level")
    for model in ("clean-trained", "noise-augmented"):
        m = lev[lev.model == model]
        ax.errorbar(m.level, m.strict_f1, yerr=[m.strict_f1 - m.strict_f1_low, m.strict_f1_high - m.strict_f1],
                    color=SYSTEM_COLOR[model], marker="o", capsize=3, label=f"{model} tagger, 60 abstracts")
    pre = summ[(summ.variant == "preprocessing") & (summ.model == "clean-trained")]
    ax.scatter(pre.level + 0.1, pre.strict_f1, marker="s", s=44, color=SYSTEM_COLOR["preprocessing"], zorder=5,
               label="+ OpenCV clean-up (clean-trained)")
    for lvl in sorted(llm.level.unique()):
        a = llm[(llm.level == lvl) & (llm.text == "raw OCR")].iloc[0]
        b = llm[(llm.level == lvl) & (llm.text == "LLM-corrected")].iloc[0]
        x = lvl - 0.12
        ax.annotate("", (x, b["strict_f1_clean-trained"]), (x, a["strict_f1_clean-trained"]),
                    arrowprops={"arrowstyle": "->", "color": SYSTEM_COLOR["LLM correction"], "lw": 1.6})
        ax.scatter([x], [a["strict_f1_clean-trained"]], marker="D", s=22, facecolor="white",
                   edgecolor=SYSTEM_COLOR["LLM correction"], zorder=5)
        ax.scatter([x], [b["strict_f1_clean-trained"]], marker="D", s=34, color=SYSTEM_COLOR["LLM correction"], zorder=5)
    ax.scatter([], [], marker="D", color=SYSTEM_COLOR["LLM correction"],
               label=f"+ LLM correction ({int(llm.documents.max())}-abstract subset: raw to corrected)")
    m = lev[lev.model == "clean-trained"]
    for _, r in m.iterrows():
        ax.annotate(f"CER {_pct(r.cer)}", (r.level, 0.03), ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(m.level)
    ax.set_xlabel("scan level (lower resolution and more noise to the right)")
    ax.set_ylabel("entity F1 (exact span and type)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 0.07))
    ax.set_title("By scan level, with the three mitigations", loc="left")
    fig.tight_layout()
    path = out / "01_f1_vs_cer.png"
    save(fig, path)
    return path


def fig_levels(res: Path, out: Path) -> Path:
    """Error rates, mention fates and F1 per entity type by level."""
    summ = pd.read_csv(res / "condition_summary.csv")
    s = summ[(summ.variant == "none") & (summ.model == "clean-trained")].sort_values("level")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    ax = axes[0]
    ax.plot(s.level, s.cer, marker="o", color=SERIES[0], label="characters, whole page (CER)")
    ax.plot(s.level, s.mention_cer, marker="s", color=SERIES[3], label="characters inside gold mentions")
    ax.plot(s.level, s.wer, marker="^", color=SERIES[6], label="words (WER)")
    ax.set_yscale("log")
    _pct_axis(ax, "y")
    ax.set_xlabel("scan level")
    ax.set_ylabel("error rate (log scale)")
    ax.set_xticks(s.level)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("WER runs 3-8x above CER", loc="left")
    ax = axes[1]
    bottom = np.zeros(len(s))
    for f in ("intact", "merged", "corrupted", "lost"):
        v = s[f"{f}_share"].to_numpy()
        ax.bar(s.level, v, bottom=bottom, color=FATE_COLOR[f], width=0.65, label=f)
        bottom += v
    _pct_axis(ax, "y")
    ax.set_xlabel("scan level")
    ax.set_ylabel(f"share of the {int(s.mentions.iloc[0]):,} gold mentions")
    ax.set_xticks(s.level)
    ax.set_xlim(-0.5, 6.3)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, loc="center right", title="mention text\nafter OCR", title_fontsize=8.5)
    ax.set_title("At level 4 only a quarter of mentions survive intact", loc="left")
    ax = axes[2]
    for t in ("chemical", "disease"):
        for model, ls in (("clean-trained", "-"), ("noise-augmented", "--")):
            m = summ[(summ.variant == "none") & (summ.model == model)].sort_values("level")
            ax.plot(m.level, m[f"strict_f1_{t}"], ls=ls, marker="o", color=TYPE_COLOR[t.capitalize()],
                    label=f"{t}, {model}")
    ax.set_xlabel("scan level")
    ax.set_ylabel("entity F1 (exact span and type)")
    ax.set_xticks(s.level)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Disease mentions lose more F1 than chemicals", loc="left")
    fig.tight_layout()
    path = out / "02_levels.png"
    save(fig, path)
    return path


def fig_mitigations(res: Path, out: Path) -> Path:
    """Forest plot of paired F1 differences for every mitigation (exact and overlap matching)."""
    sig = pd.read_csv(res / "significance.csv")
    sig = sig[sig.family != "OCR vs clean text"].copy()
    fam_order = {"noise-augmented vs clean-trained": 0, "preprocessing vs raw OCR": 1,
                 "preprocessing + noise-augmented vs raw + clean-trained": 2, "LLM correction vs raw OCR": 3}
    sig["order"] = sig.family.map(fam_order)
    sig["lvl"] = sig.condition_b.map(lambda c: -1 if c == "clean text" else int(c.split()[0][1:]))
    sig = sig.sort_values(["order", "model_a", "lvl"])
    fam_color = {"noise-augmented vs clean-trained": SYSTEM_COLOR["noise-augmented"],
                 "preprocessing vs raw OCR": SYSTEM_COLOR["preprocessing"],
                 "LLM correction vs raw OCR": SYSTEM_COLOR["LLM correction"],
                 "preprocessing + noise-augmented vs raw + clean-trained": SERIES[5]}

    def label(r) -> str:
        where = r.condition_b
        if r.family == "noise-augmented vs clean-trained":
            return f"noise-augmented vs clean-trained tagger, {where}"
        if r.family.startswith("preprocessing +"):
            return f"clean-up + noise-augmented vs raw + clean-trained, {where}"
        what = "OpenCV clean-up" if r.family.startswith("preprocessing") else "LLM correction"
        return f"{what}, {r.model_a} tagger, {where} (n={r.documents})"

    strict = sig[sig.metric == "strict_f1"].reset_index(drop=True)
    overlap = sig[sig.metric == "overlap_f1"].set_index(["family", "model_a", "condition_a"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.36 * len(strict) + 1.6), sharey=True)
    y = np.arange(len(strict))[::-1]
    for ax, metric in zip(axes, ("strict", "overlap")):
        for yi, r in zip(y, strict.itertuples()):
            q = r._asdict() if metric == "strict" else overlap.loc[(r.family, r.model_a, r.condition_a)].to_dict()
            c = fam_color[r.family]
            ax.plot([q["ci_low"], q["ci_high"]], [yi, yi], color=c, lw=2)
            ax.scatter(q["diff"], yi, color=c if q["p_holm"] < 0.05 else "white", edgecolor=c, s=46, zorder=3, lw=1.6)
        ax.axvline(0, color=INK_2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("change in entity F1 vs the baseline on the same pages (95% paired bootstrap CI)", fontsize=9)
        ax.set_title("Exact span and type" if metric == "strict" else "Any overlap with the right type", loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label(r) for r in strict.itertuples()], fontsize=8.5)
    handles = [Line2D([], [], marker="o", color=MUTED, markerfacecolor=MUTED, ls="", label="Holm-adjusted p < 0.05"),
               Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", ls="", label="not significant")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.suptitle("OpenCV clean-up and LLM correction recover exact-match F1 on bad scans; noise-augmented training only "
                 "finds mentions approximately", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out / "03_mitigations.png"
    save(fig, path)
    return path


def fig_mentions(res: Path, out: Path) -> Path:
    """Recall by mention fate and level, for both taggers."""
    ms = pd.read_csv(res / "mention_status.csv")
    ms = ms[(ms.type == "all") & ms.condition.str.fullmatch(r"L\d")].copy()
    ms["level"] = ms.condition.str[1:].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), sharey=True)
    for ax, col, title in zip(axes, ("found_strict", "found_overlap"),
                              ("Found with the exact span", "Found by an overlapping prediction")):
        for f in ("intact", "merged", "corrupted"):
            for model, ls in (("clean-trained", "-"), ("noise-augmented", "--")):
                m = ms[(ms.model == model) & (ms.status == f) & (ms.mentions >= 5)].sort_values("level")
                ax.plot(m.level, m[col], marker="o", ls=ls, color=FATE_COLOR[f], label=f"{f}, {model}")
        ax.set_xlabel("scan level")
        ax.set_xticks(sorted(ms.level.unique()))
        ax.set_ylim(-0.02, 1.02)
        _pct_axis(ax, "y")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("share of gold mentions found")
    axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.06), ncol=1)
    fig.suptitle("Intact mentions are still found in noisy text; changed or glued ones are not "
                 "(points with at least 5 mentions)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out / "04_mention_fate.png"
    save(fig, path)
    return path


def fig_llm(res: Path, out: Path) -> Path:
    """Per-abstract CER, WER and F1 before and after LLM correction."""
    llm = pd.read_csv(res / "llm_correction.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (before, after, name) in zip(axes[:2], (("cer_before", "cer_after", "CER"),
                                                    ("wer_before", "wer_after", "WER"))):
        for lv in sorted(llm.level.unique()):
            m = llm[llm.level == lv]
            ax.scatter(m[before].clip(lower=1e-4), m[after].clip(lower=1e-4), color=LEVEL_COLOR.get(lv, MUTED), s=32,
                       edgecolor="white", lw=0.5, label=f"level {lv} (n={len(m)})")
        lim = (8e-5, 1.2)
        ax.plot(lim, lim, color=INK_2, lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        _pct_axis(ax, "x")
        _pct_axis(ax, "y")
        ax.set_xlabel(f"{name} of the OCR text (log; 0 drawn at 0.01%)")
        ax.set_ylabel(f"{name} after LLM correction")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_title("Characters: fewer errors at level 3, not at level 4", loc="left")
    axes[1].set_title("Words: fewer errors at both", loc="left")
    ax = axes[2]
    for lv in sorted(llm.level.unique()):
        m = llm[llm.level == lv]
        ax.scatter(m["f1_before_clean-trained"], m["f1_after_clean-trained"], color=LEVEL_COLOR.get(lv, MUTED), s=32,
                   edgecolor="white", lw=0.5, label=f"level {lv}")
    ax.plot((0, 1), (0, 1), color=INK_2, lw=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("entity F1 per abstract, raw OCR (clean-trained)")
    ax.set_ylabel("entity F1 after LLM correction")
    ax.legend(fontsize=8.5, loc="lower right")
    better = int((llm["f1_after_clean-trained"] > llm["f1_before_clean-trained"]).sum())
    worse = int((llm["f1_after_clean-trained"] < llm["f1_before_clean-trained"]).sum())
    ax.set_title(f"Entity F1: {better} abstracts up, {worse} down", loc="left")
    fig.tight_layout()
    path = out / "05_llm_correction.png"
    save(fig, path)
    return path


def make_all(results: Path, figures: Path) -> list[Path]:
    """Draw every figure; returns the written paths."""
    apply_style()
    figures.mkdir(parents=True, exist_ok=True)
    return [f(results, figures) for f in (fig_pages, fig_f1_vs_cer, fig_levels, fig_mitigations, fig_mentions, fig_llm)]
