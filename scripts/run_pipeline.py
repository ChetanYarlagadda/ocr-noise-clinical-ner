"""Run every experiment end to end.

Renders test abstracts as scanned pages at five degradation levels, OCRs them with docTR (plus OpenCV-cleaned
pages at the two worst levels and LLM-corrected text on a subset), fine-tunes a clean and a noise-augmented
PubMedBERT tagger, projects the gold mentions onto every OCR text and scores both taggers. Every expensive step
(OCR text, trained weights, predictions, LLM replies) is cached in data/cache under keys that include its settings,
so a re-run only recomputes what changed.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --quick    # smoke test on a few documents; outputs go to data/cache/smoke/
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ocrner import data as D  # noqa: E402
from ocrner import ner, noise, preprocess, render  # noqa: E402
from ocrner.align import cer as cer_of  # noqa: E402
from ocrner.align import wer as wer_of  # noqa: E402
from ocrner.align import char_map, project  # noqa: E402
from ocrner.evaluate import score_condition, summarize  # noqa: E402
from ocrner.figures import make_all  # noqa: E402
from ocrner.llm import OllamaCorrector  # noqa: E402
from ocrner.metrics import (  # noqa: E402
    bootstrap_f1, bootstrap_indices, doc_arrays, doc_counts, holm, micro, paired_diff,
)
from ocrner.ocr import DocTR, TextCache, ocr_condition  # noqa: E402
from ocrner.tracking import init_mlflow, log_result  # noqa: E402

torch.set_num_threads(2)

CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
VERSION = "v1"  # bump to invalidate every cached output
MODELS = ("clean-trained", "noise-augmented")

# Gu et al. (2021), Table 6: entity-level test F1 on BLURB BC5-chem / BC5-disease (separate models per type).
PUBLISHED_F1 = {("PubMedBERT (uncased, abstracts), Gu et al. 2021", "Chemical"): 0.9333,
                ("PubMedBERT (uncased, abstracts), Gu et al. 2021", "Disease"): 0.8562,
                ("BioBERT (cased), Gu et al. 2021", "Chemical"): 0.9285,
                ("BioBERT (cased), Gu et al. 2021", "Disease"): 0.8470}
# Li et al. (2016), Table 2: inter-annotator agreement on the test set.
IAA = {"Chemical": 0.9630, "Disease": 0.8875}


@dataclass(frozen=True)
class Settings:
    """Sample sizes, conditions and training settings of one run."""

    test_docs: int = 60  # test abstracts rendered and OCR'd at every level
    levels: tuple[int, ...] = (0, 1, 2, 3, 4)
    channel_docs: int = 6  # development abstracts OCR'd at levels 1-4 to learn the noise channel
    pre_docs: int = 4  # development abstracts (a subset of the above) used to choose the preprocessing pipeline
    pre_candidates: tuple[tuple[str, ...], ...] = (("deskew",), ("deskew", "nlmeans"),
                                                   ("deskew", "upscale", "median", "otsu"))
    pre_levels: tuple[int, ...] = (4,)  # levels where the OpenCV clean-up is chosen and applied
    llm_docs: int = 8  # test abstracts (the first of the sample) corrected by the LLM at llm_levels
    llm_levels: tuple[int, ...] = (3, 4)
    llm_clean_docs: int = 4  # ... and at level 0, to measure damage to near-clean text
    dev_docs: int = 150  # development abstracts used to pick the best epoch
    clean: ner.TrainConfig = field(default_factory=ner.TrainConfig)
    noisy: ner.TrainConfig = field(default_factory=lambda: ner.TrainConfig(noise_prob=0.5))
    n_boot: int = 1000
    cer_bins: tuple[float, ...] = (0.0, 0.0025, 0.01, 0.025, 0.05, 0.1, 0.2, 1.0)


QUICK = Settings(test_docs=4, channel_docs=3, pre_docs=2, llm_docs=2, llm_clean_docs=1, dev_docs=40, n_boot=50,
                 clean=ner.TrainConfig(epochs=1, subsample=40),
                 noisy=ner.TrainConfig(epochs=1, subsample=40, noise_prob=0.5))


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Timer:
    """Wall-clock seconds per stage; GPU stages are listed separately."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.gpu: set[str] = set()

    def run(self, name: str, fn: Callable, gpu: bool = False):
        t0 = time.perf_counter()
        out = fn()
        if gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        if gpu:
            self.gpu.add(name)
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0
        return out


def key_of(spec: dict) -> str:
    """Short hash of everything that changes an output."""
    return hashlib.sha1(json.dumps({**spec, "version": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def condition_name(level: int, variant: str = "") -> str:
    return f"L{level}" + (f" + {variant}" if variant else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on a few documents with tiny budgets")
    args = ap.parse_args()
    S = QUICK if args.quick else Settings()
    out_root = CACHE / "smoke" if args.quick else ROOT
    results, figures = out_root / "reports" / "results", out_root / "reports" / "figures"
    results.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = datetime.now(timezone.utc)
    os.environ.setdefault("DOCTR_CACHE_DIR", str(CACHE / "doctr"))
    pages_ocr: dict[str, int] = {}

    # ------------------------------------------------------------------ data and checks
    splits = timer.run("data", lambda: D.load(RAW))
    checks = D.crosscheck(splits)
    log(f"published corpus statistics matched: {int(checks.match.sum())}/{len(checks)}")
    checks.to_csv(results / "data_crosscheck.csv", index=False)
    summary = D.split_summary(splits)
    summary["mentions_on_word_boundaries"] = [
        float(np.mean([ner.boundary_alignment(d.text, [e]) for d in splits[s] for e in d.entities])) for s in summary.split]
    summary.to_csv(results / "split_summary.csv", index=False)
    pd.DataFrame(render.level_table()).to_csv(results / "degradation_levels.csv", index=False)

    gold_df = D.parse_mentions(D.read_member(RAW, D.SPLIT_FILES["test"]))
    tool_rows, tool_preds = [], {}
    for tool, member in D.TOOL_FILES.items():
        pred = D.parse_mentions(D.read_member(RAW, member))
        p, r, f = D.concept_scores(gold_df, pred, D.TOOL_TYPE[tool])
        rp, rr, rf = D.TOOL_README[tool]
        tool_rows.append({"tool": tool, "type": D.TOOL_TYPE[tool], "concept_precision": p, "concept_recall": r,
                          "concept_f1": f, "readme_precision": rp, "readme_recall": rr, "readme_f1": rf,
                          "match_to_2dp": all(abs(a - b) < 0.005 for a, b in [(p, rp), (r, rr), (f, rf)])})
        for row in pred.itertuples():
            tool_preds.setdefault(row.pmid, []).append(D.Entity(row.start, row.end, row.type))
    pd.DataFrame(tool_rows).to_csv(results / "scorer_check.csv", index=False)
    log(f"concept-level scorer reproduces the archive README for {sum(r['match_to_2dp'] for r in tool_rows)}/2 tools")

    rng = np.random.default_rng(0)
    test = [splits["test"][i] for i in rng.permutation(len(splits["test"]))[:S.test_docs]]
    channel_docs = [splits["dev"][i] for i in np.random.default_rng(1).permutation(len(splits["dev"]))[:S.channel_docs]]
    pre_docs = channel_docs[:S.pre_docs]
    by_id = {d.pmid: d for s in ("dev", "test") for d in splits[s]}
    log(f"test sample: {len(test)} abstracts, {sum(len(d.entities) for d in test)} mentions")

    # ------------------------------------------------------------------ OCR
    engine: list[DocTR] = []

    def get_engine() -> DocTR:
        if not engine:
            engine.append(DocTR(CACHE / "doctr", device=dev.type))
        return engine[0]

    def page(pmid: str, level: int, pipeline: tuple[str, ...] = ()) -> np.ndarray:
        d = by_id[pmid]
        img = np.asarray(render.scanned_page(d.text, d.title_end, level, seed=int(d.pmid)))
        return preprocess.apply(img, pipeline) if pipeline else img

    def ocr(split: str, docs: list[D.Doc], level: int, pipeline: tuple[str, ...] = ()) -> dict[str, str]:
        spec = {"level": asdict(render.LEVELS[level]), "pipeline": pipeline,
                "ocr": "doctr db_resnet50+crnn_vgg16_bn fp32", "render": "dejavu-serif-11pt-v1"}
        cache = TextCache(CACHE / "ocr" / f"{split}-L{level}-{'+'.join(pipeline) or 'raw'}-{key_of(spec)}.json")
        ids = [d.pmid for d in docs]
        todo = len(cache.missing(ids))
        stage = "OCR (preprocessed pages)" if pipeline else "OCR"
        out = timer.run(stage, lambda: ocr_condition(ids, lambda j: page(j, level, pipeline), cache, get_engine),
                        gpu=todo > 0)
        pages_ocr[stage] = pages_ocr.get(stage, 0) + todo
        return out

    channel_ocr = {lv: ocr("dev", channel_docs, lv) for lv in (1, 2, 3, 4)}
    sel_rows = []
    for lv in S.pre_levels:
        raw_cer = [cer_of(d.text, channel_ocr[lv][d.pmid]) for d in pre_docs]
        sel_rows.append({"level": lv, "pipeline": "none", "cer": float(np.mean(raw_cer))})
        for cand in S.pre_candidates:
            texts = ocr("dev", pre_docs, lv, cand)
            sel_rows.append({"level": lv, "pipeline": "+".join(cand),
                             "cer": float(np.mean([cer_of(d.text, texts[d.pmid]) for d in pre_docs]))})
    sel = pd.DataFrame(sel_rows)
    ranking = sel.groupby("pipeline").cer.mean().sort_values()
    chosen = tuple(ranking.index[0].split("+")) if ranking.index[0] != "none" else ()
    sel["chosen"] = sel.pipeline == "+".join(chosen)
    sel.to_csv(results / "preprocessing_selection.csv", index=False)
    log(f"preprocessing chosen on {len(pre_docs)} dev abstracts: {'+'.join(chosen) or 'none'} "
        f"(mean CER {ranking.iloc[0]:.4f} vs none {ranking['none']:.4f})")

    hyp: dict[str, dict[str, str]] = {"clean text": {d.pmid: d.text for d in test}}
    for lv in S.levels:
        hyp[condition_name(lv)] = ocr("test", test, lv)
        log(f"OCR level {lv}: mean CER per page {np.mean([cer_of(d.text, hyp[condition_name(lv)][d.pmid]) for d in test]):.4f}")
    if chosen:
        for lv in S.pre_levels:
            hyp[condition_name(lv, "preprocessing")] = ocr("test", test, lv, chosen)
    examples_pages = {f"L{lv}": page(test[0].pmid, lv) for lv in S.levels}
    if chosen:
        examples_pages[f"L{S.levels[-1]} + preprocessing"] = page(test[0].pmid, S.levels[-1], chosen)
    if engine:
        engine.clear()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ noise channel learned from dev OCR
    channel = noise.fit_channel([(d.text, channel_ocr[lv][d.pmid]) for lv in (1, 2, 3, 4) for d in channel_docs])
    visible = {" ": "<space>", "": ""}
    pd.DataFrame(noise.channel_table(channel, top=40)).assign(
        reference=lambda x: x.reference.map(lambda c: visible.get(c, c)),
        ocr=lambda x: x.ocr.map(lambda c: visible.get(c, c).replace(" ", "<space>"))).to_csv(
        results / "noise_channel.csv", index=False)
    chk_rows = []
    crng = np.random.default_rng(5)
    for target in (0.01, 0.05, 0.2):
        for d in splits["train"][:50]:
            noisy, ents, cmap = noise.corrupt(d.text, d.entities, channel, target / channel.expected_cer(), crng)
            spans = [(e.start, e.end) for e in d.entities]
            tracked = project(spans, cmap, noisy)
            aligned = project(spans, char_map(d.text, noisy), noisy)
            chk_rows.append({"target_cer": target, "pmid": d.pmid, "realized_cer": cer_of(d.text, noisy),
                             "mentions": len(spans), "alignment_agrees": sum(a == b for a, b in zip(tracked, aligned))})
    chk = pd.DataFrame(chk_rows)
    chk.groupby("target_cer").agg(realized_cer=("realized_cer", "mean"), mentions=("mentions", "sum"),
                                  alignment_agrees=("alignment_agrees", "sum")).reset_index().assign(
        agreement=lambda x: x.alignment_agrees / x.mentions).to_csv(results / "channel_check.csv", index=False)
    log(f"noise channel: expected CER at scale 1 = {channel.expected_cer():.4f}; "
        f"alignment agrees with the generator on {chk.alignment_agrees.sum() / chk.mentions.sum():.3f} of spans")

    # ------------------------------------------------------------------ NER training (clean and noise-augmented)
    train_docs = splits["train"][:S.clean.subsample] if S.clean.subsample else splits["train"]
    dev_docs = splits["dev"][:S.dev_docs]
    dev_pairs = [(d.text, d.entities) for d in dev_docs]

    def dev_f1(preds: list[list[D.Entity]]) -> float:
        counts = pd.DataFrame([doc_counts([((e.start, e.end), e.type) for e in d.entities], p)
                               for d, p in zip(dev_docs, preds)])
        return micro(counts, "strict")[2]

    def epoch_docs(cfg: ner.TrainConfig) -> Callable[[int], list[tuple[str, list[D.Entity]]]]:
        def make(epoch: int) -> list[tuple[str, list[D.Entity]]]:
            if cfg.noise_prob == 0:
                return [(d.text, d.entities) for d in train_docs]
            r = np.random.default_rng([epoch, 7])
            out = []
            for d in train_docs:
                if r.random() < cfg.noise_prob:
                    target = float(np.exp(r.uniform(np.log(cfg.noise_cer[0]), np.log(cfg.noise_cer[1]))))
                    text, ents, _ = noise.corrupt(d.text, d.entities, channel, target / channel.expected_cer(), r)
                    out.append((text, ents))
                else:
                    out.append((d.text, d.entities))
            return out
        return make

    model_cfg = {"clean-trained": S.clean, "noise-augmented": S.noisy}
    model_keys, history, model_rows = {}, [], []
    channel_key = hashlib.sha1(json.dumps(noise.channel_table(channel, top=10_000), sort_keys=True).encode()).hexdigest()[:8]
    for name, cfg in model_cfg.items():
        spec = {"cfg": asdict(cfg), "train": len(train_docs), "dev": len(dev_docs),
                "channel": channel_key if cfg.noise_prob else None}
        mkey = key_of(spec)
        model_keys[name] = mkey
        wpath = CACHE / "weights" / f"{name}-{mkey}.pt"
        if not wpath.exists():
            log(f"training the {name} tagger ({cfg.epochs} epochs on {len(train_docs)} abstracts)")
            t0 = time.perf_counter()
            state, hist = timer.run("NER training", lambda: ner.train(cfg, epoch_docs(cfg), dev_pairs, dev_f1, dev,
                                                                       seed=0, log=log), gpu=True)
            wpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": state, "history": hist.to_dict("records"), "spec": spec,
                        "seconds": time.perf_counter() - t0}, wpath)
        saved = torch.load(wpath, map_location="cpu", weights_only=False)
        h = pd.DataFrame(saved["history"]).assign(model=name)
        history.append(h)
        best = h.loc[h.dev_f1.idxmax()]
        model_rows.append({"model": name, "epochs_run": len(h), "best_epoch": int(best.epoch), "dev_f1": best.dev_f1,
                           "train_seconds": saved["seconds"], "noise_prob": cfg.noise_prob,
                           "noise_cer_range": f"{cfg.noise_cer[0]}-{cfg.noise_cer[1]}" if cfg.noise_prob else ""})
    pd.concat(history).to_csv(results / "training_history.csv", index=False)
    pd.DataFrame(model_rows).to_csv(results / "models.csv", index=False)

    # ------------------------------------------------------------------ LLM post-correction (subset)
    llm = OllamaCorrector(CACHE / "llm" / "replies.sqlite")
    llm_rows = []
    llm_sets = [(lv, test[:S.llm_docs]) for lv in S.llm_levels] + [(0, test[:S.llm_clean_docs])]
    for lv, docs in llm_sets:
        name = condition_name(lv, "LLM correction")
        hyp[name] = {}
        for d in docs:
            src = hyp[condition_name(lv)][d.pmid]
            t0 = time.perf_counter()
            fixed, secs, cached = llm.correct(src)
            if not cached:  # only new calls use the GPU
                timer.stages["LLM correction"] = timer.stages.get("LLM correction", 0.0) + time.perf_counter() - t0
                timer.gpu.add("LLM correction")
            fixed = " ".join(fixed.split())
            hyp[name][d.pmid] = fixed
            llm_rows.append({"level": lv, "pmid": d.pmid, "cer_before": cer_of(d.text, src),
                             "cer_after": cer_of(d.text, fixed), "wer_before": wer_of(d.text, src),
                             "wer_after": wer_of(d.text, fixed), "chars_before": len(src), "chars_after": len(fixed),
                             "model_seconds": secs, "cached": cached})
    log(f"LLM: {llm.misses} new calls ({llm.new_seconds:.0f} s), {llm.hits} cached")

    # ------------------------------------------------------------------ NER inference on every condition
    full_test = splits["test"]
    preds: dict[tuple[str, str], list[list[D.Entity]]] = {}
    for name in MODELS:
        cfg = model_cfg[name]
        loaded: list = []

        def get_model(name: str = name, cfg: ner.TrainConfig = cfg, loaded: list = loaded):
            if not loaded:
                state = torch.load(CACHE / "weights" / f"{name}-{model_keys[name]}.pt", map_location="cpu",
                                   weights_only=False)["state"]
                loaded.extend(ner.load_model(cfg, dev, state))
            return loaded

        jobs = {"full test (clean text)": [d.text for d in full_test]}
        for cond, texts in hyp.items():
            jobs[cond] = [texts[d.pmid] for d in test if d.pmid in texts]
        for cond, texts in jobs.items():
            path = CACHE / "preds" / f"{name}-{key_of({'model': model_keys[name], 'texts': texts})}.pkl"
            if path.exists():
                preds[(name, cond)] = pickle.loads(path.read_bytes())
                continue
            tok, model = get_model()
            out = timer.run("NER inference", lambda: ner.predict(model, tok, texts, cfg, dev), gpu=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(out))
            preds[(name, cond)] = out
        loaded.clear()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ scoring
    doc_tables, mention_tables, summary_rows = [], [], []
    conditions = list(hyp)
    for name in MODELS:
        for cond in conditions:
            docs = [d for d in test if d.pmid in hyp[cond]]
            df, men = score_condition(docs, [hyp[cond][d.pmid] for d in docs], preds[(name, cond)])
            meta = {"model": name, "condition": cond}
            doc_tables.append(df.assign(**meta))
            mention_tables.append(men.assign(**meta))
    docs_df = pd.concat(doc_tables, ignore_index=True)
    docs_df.to_csv(results / "doc_results.csv", index=False)
    mentions_df = pd.concat(mention_tables, ignore_index=True)

    def level_of(cond: str) -> float:
        return float(cond.split()[0][1:]) if cond.startswith("L") else np.nan

    for (name, cond), df in docs_df.groupby(["model", "condition"], sort=False):
        idx = bootstrap_indices(len(df), S.n_boot)
        summary_rows.append({"model": name, "condition": cond, "level": level_of(cond),
                             "variant": cond.split(" + ")[1] if " + " in cond else ("none" if cond != "clean text" else "clean text"),
                             **summarize(df, idx)})
    cond_df = pd.DataFrame(summary_rows)
    cond_df.to_csv(results / "condition_summary.csv", index=False)

    ms = (mentions_df.groupby(["model", "condition", "type", "status"])
          .agg(mentions=("found_strict", "size"), found_strict=("found_strict", "mean"),
               found_overlap=("found_overlap", "mean")).reset_index())
    ms_all = (mentions_df.groupby(["model", "condition", "status"])
              .agg(mentions=("found_strict", "size"), found_strict=("found_strict", "mean"),
                   found_overlap=("found_overlap", "mean")).reset_index().assign(type="all"))
    pd.concat([ms_all, ms], ignore_index=True).to_csv(results / "mention_status.csv", index=False)

    # paired comparisons (same documents), Holm-corrected together
    comps = []

    def compare(a: tuple[str, str], b: tuple[str, str], family: str) -> None:
        da = docs_df[(docs_df.model == a[0]) & (docs_df.condition == a[1])].set_index("pmid")
        db = docs_df[(docs_df.model == b[0]) & (docs_df.condition == b[1])].set_index("pmid")
        common = [p for p in da.index if p in db.index]
        da, db = da.loc[common], db.loc[common]
        idx = bootstrap_indices(len(common), S.n_boot, seed=1)
        for mode in ("strict", "overlap"):
            r = paired_diff(doc_arrays(da, mode), doc_arrays(db, mode), idx)
            comps.append({"family": family, "model_a": a[0], "condition_a": a[1], "model_b": b[0],
                          "condition_b": b[1], "metric": f"{mode}_f1", "documents": len(common),
                          "f1_a": micro(da, mode)[2], "f1_b": micro(db, mode)[2], **r})

    for lv in S.levels:
        compare(("clean-trained", condition_name(lv)), ("clean-trained", "clean text"), "OCR vs clean text")
    for cond in ["clean text"] + [condition_name(lv) for lv in S.levels]:
        compare(("noise-augmented", cond), ("clean-trained", cond), "noise-augmented vs clean-trained")
    for name in MODELS:
        if chosen:
            for lv in S.pre_levels:
                compare((name, condition_name(lv, "preprocessing")), (name, condition_name(lv)), "preprocessing vs raw OCR")
        for lv, _ in llm_sets:
            compare((name, condition_name(lv, "LLM correction")), (name, condition_name(lv)), "LLM correction vs raw OCR")
    if chosen:
        for lv in S.pre_levels:
            compare(("noise-augmented", condition_name(lv, "preprocessing")), ("clean-trained", condition_name(lv)),
                    "preprocessing + noise-augmented vs raw + clean-trained")
    comp_df = pd.DataFrame(comps)
    comp_df["p_holm"] = np.nan  # Holm over all comparisons of one metric (strict and overlap F1 separately)
    for metric in comp_df.metric.unique():
        rows = comp_df.metric == metric
        comp_df.loc[rows, "p_holm"] = holm(comp_df.loc[rows, "p"].to_numpy())
    strict = comp_df.metric == "strict_f1"
    comp_df.to_csv(results / "significance.csv", index=False)
    log(f"{int(strict.sum())} paired strict-F1 comparisons, {int((comp_df[strict].p_holm < 0.05).sum())} significant "
        "after Holm")

    # entity F1 against CER, pooling every raw-OCR page (document x level)
    raw_conds = [condition_name(lv) for lv in S.levels]
    bin_rows = []
    for name in MODELS:
        pool = docs_df[(docs_df.model == name) & docs_df.condition.isin(raw_conds)].reset_index(drop=True)
        bins = pd.cut(pool.cer, S.cer_bins, right=False, include_lowest=True)
        for b, g in pool.groupby(bins, observed=True):
            idx = bootstrap_indices(len(g), S.n_boot, seed=2)
            for mode in ("strict", "overlap"):
                lo, hi = bootstrap_f1(doc_arrays(g, mode), idx)
                bin_rows.append({"model": name, "metric": f"{mode}_f1", "cer_low": b.left, "cer_high": b.right,
                                 "pages": len(g), "cer": g.char_errors.sum() / g.chars.sum(),
                                 "f1": micro(g, mode)[2], "f1_low": lo, "f1_high": hi,
                                 "recall": micro(g, mode)[1], "precision": micro(g, mode)[0]})
    pd.DataFrame(bin_rows).to_csv(results / "f1_vs_cer_bins.csv", index=False)
    # simple slope: doc-level strict F1 regressed on CER below 10% (clean-trained), bootstrap CI over pages
    slope_rows = []
    for name in MODELS:
        pool = docs_df[(docs_df.model == name) & docs_df.condition.isin(raw_conds) & (docs_df.cer < 0.10)]
        f1_doc = 2 * pool.filter(like="strict_tp_").sum(1) / (2 * pool.filter(like="strict_tp_").sum(1)
                                                              + pool.filter(like="strict_fp_").sum(1)
                                                              + pool.filter(like="strict_fn_").sum(1)).clip(lower=1)
        x, y = pool.cer.to_numpy() * 100, f1_doc.to_numpy()
        b = np.polyfit(x, y, 1)[0]
        bs = [np.polyfit(x[i], y[i], 1)[0] for i in bootstrap_indices(len(x), S.n_boot, seed=3)]
        slope_rows.append({"model": name, "pages": len(x), "cer_max": 0.10, "f1_per_cer_point": b,
                           "ci_low": float(np.percentile(bs, 2.5)), "ci_high": float(np.percentile(bs, 97.5))})
    pd.DataFrame(slope_rows).to_csv(results / "f1_cer_slope.csv", index=False)

    # LLM details
    llm_df = pd.DataFrame(llm_rows)
    for name in MODELS:
        for lv, docs in llm_sets:
            sub = docs_df[docs_df.model == name].set_index(["condition", "pmid"])
            for d in docs:
                for tag, cond in (("before", condition_name(lv)), ("after", condition_name(lv, "LLM correction"))):
                    r = sub.loc[(cond, d.pmid)]
                    tp = r.filter(like="strict_tp_").sum()
                    denom = 2 * tp + r.filter(like="strict_fp_").sum() + r.filter(like="strict_fn_").sum()
                    llm_df.loc[(llm_df.level == lv) & (llm_df.pmid == d.pmid), f"f1_{tag}_{name}"] = 2 * tp / max(1, denom)
    llm_df.to_csv(results / "llm_correction.csv", index=False)
    llm_sum = []
    for lv, docs in llm_sets:
        ids = [d.pmid for d in docs]
        for tag, cond in (("raw OCR", condition_name(lv)), ("LLM-corrected", condition_name(lv, "LLM correction"))):
            row = {"level": lv, "documents": len(ids), "text": tag}
            for name in MODELS:
                df = docs_df[(docs_df.model == name) & (docs_df.condition == cond) & docs_df.pmid.isin(ids)]
                summ = summarize(df, bootstrap_indices(len(df), S.n_boot))
                if name == MODELS[0]:
                    row.update({k: summ[k] for k in ("cer", "wer", "mention_cer", "intact_share", "merged_share",
                                                     "corrupted_share", "lost_share")})
                row.update({f"{m}_f1_{name}": summ[f"{m}_f1"] for m in ("strict", "overlap")})
            llm_sum.append(row)
    pd.DataFrame(llm_sum).to_csv(results / "llm_summary.csv", index=False)

    # clean-text results on the full test set and the published comparison
    pub_rows = []
    for name in MODELS:
        df, _ = score_condition(full_test, [d.text for d in full_test], preds[(name, "full test (clean text)")])
        idx = bootstrap_indices(len(df), S.n_boot)
        for t in (*D.TYPES, None):
            p, r, f = micro(df, "strict", t)
            lo, hi = bootstrap_f1(doc_arrays(df, "strict", t), idx)
            pub_rows.append({"system": f"{name} PubMedBERT (this project)", "type": t or "both", "precision": p,
                             "recall": r, "f1": f, "f1_low": lo, "f1_high": hi, "source": "computed"})
    tool_df = pd.DataFrame([{"pmid": d.pmid, **doc_counts([((e.start, e.end), e.type) for e in d.entities],
                                                         tool_preds.get(d.pmid, []))} for d in full_test])
    for tool, t in D.TOOL_TYPE.items():
        p, r, f = micro(tool_df, "strict", t)
        pub_rows.append({"system": f"{tool} output shipped with the corpus", "type": t, "precision": p, "recall": r,
                         "f1": f, "source": "computed from the archive"})
    for (system, t), f in PUBLISHED_F1.items():
        pub_rows.append({"system": system, "type": t, "f1": f, "source": "published (separate model per type)"})
    for t, a in IAA.items():
        pub_rows.append({"system": "Inter-annotator agreement, test set (Li et al. 2016)", "type": t, "f1": a,
                         "source": "published"})
    pd.DataFrame(pub_rows).to_csv(results / "published_comparison.csv", index=False)

    # example snippets and page crops
    ex_doc = test[0]
    ex_rows = []
    worst = condition_name(S.levels[-1])
    for cond in [c for c in ["clean text", condition_name(2), worst, f"{worst} + preprocessing",
                             f"{worst} + LLM correction"] if c in hyp]:
        if ex_doc.pmid not in hyp[cond]:
            continue
        text = hyp[cond][ex_doc.pmid]
        proj = project([(e.start, e.end) for e in ex_doc.entities], char_map(ex_doc.text, text), text)
        for name in MODELS:
            k = [d.pmid for d in test if d.pmid in hyp[cond]].index(ex_doc.pmid)
            pred = preds[(name, cond)][k]
            ex_rows.append({"condition": cond, "model": name, "cer": cer_of(ex_doc.text, text), "text": text[:600],
                            "gold": json.dumps([[p[0], p[1], e.type] if p else None for p, e in zip(proj, ex_doc.entities)]),
                            "pred": json.dumps([[e.start, e.end, e.type] for e in pred])})
    pd.DataFrame(ex_rows).to_csv(results / "examples.csv", index=False)
    crops = {}
    for k, img in examples_pages.items():
        w = img.shape[1]
        dpi = w / 8.5  # pixels per inch of this page (letter width)
        crops[k] = img[int(0.62 * dpi): int(1.72 * dpi), int(0.9 * dpi): int(4.75 * dpi)]
    np.savez_compressed(results / "page_examples.npz", **{f"page{i}": v for i, v in enumerate(crops.values())},
                        names=np.array(list(crops)), pmid=np.array(ex_doc.pmid))

    # ------------------------------------------------------------------ tracking and metadata
    init_mlflow(out_root)
    for r in summary_rows:
        log_result(f"{r['model']} | {r['condition']}", {"model": r["model"], "condition": r["condition"],
                                                         "test_docs": S.test_docs},
                   {k: v for k, v in r.items() if isinstance(v, (int, float)) and k != "level"})
    n_llm, llm_secs = llm.total_seconds()
    gpu_s = sum(v for k, v in timer.stages.items() if k in timer.gpu)
    meta = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick, "settings": asdict(S), "preprocessing_chosen": list(chosen),
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()}, "gpu_stages": sorted(timer.gpu),
        "gpu_seconds_this_run": round(gpu_s, 1), "pages_ocr_this_run": pages_ocr,
        "llm": {"model": llm.model, "new_calls": llm.misses, "cached_calls": llm.hits,
                "new_model_seconds": round(llm.new_seconds, 1), "cache_total_replies": n_llm,
                "cache_total_model_seconds": round(llm_secs, 1)},
        "noise_channel_expected_cer": channel.expected_cer(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "doctr": __import__("doctr").__version__,
                     "transformers": __import__("transformers").__version__},
        "data": {"source": D.URL, "sha256": D.SHA256, "license": D.LICENSE},
        "model_keys": model_keys,
    }
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # every run (smoke, full, cached re-runs) is appended to one compute log, so total compute is auditable
    log_path = CACHE / "compute_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"started_utc": meta["started_utc"], "finished_utc": meta["finished_utc"],
                             "kind": "smoke" if args.quick else ("full" if gpu_s > 60 else "cached re-run"),
                             "timings_s": meta["timings_s"], "gpu_stages": meta["gpu_stages"],
                             "gpu_seconds": meta["gpu_seconds_this_run"], "pages_ocr": pages_ocr,
                             "llm_new_calls": llm.misses, "llm_new_seconds": round(llm.new_seconds, 1)}) + "\n")
    runs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pd.DataFrame([{"started_utc": r["started_utc"], "kind": r["kind"],
                   "wall_seconds": (datetime.fromisoformat(r["finished_utc"])
                                    - datetime.fromisoformat(r["started_utc"])).total_seconds(),
                   "gpu_stage_seconds": r["gpu_seconds"], "pages_ocr": sum(r["pages_ocr"].values()),
                   "llm_new_calls": r["llm_new_calls"]} for r in runs]).to_csv(results / "compute_log.csv", index=False)
    for p in make_all(results, figures):
        log(f"wrote {p.name}")
    log(f"done; GPU stages {gpu_s / 60:.1f} min; all stages {sum(timer.stages.values()) / 60:.1f} min")


if __name__ == "__main__":
    main()
