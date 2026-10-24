"""Transformer NER for chemicals and diseases: word-level BIO tagging with a fine-tuned biomedical BERT.

Text is split into words with ``\\w+|[^\\w\\s]`` (every punctuation mark is its own word), cut into windows of at
most ``max_words`` words (preferably after a full stop, so clean and OCR text are windowed the same way), and each
word is labelled through its first sub-word token. Predicted mentions are maximal runs of B/I labels of one type,
mapped back to character offsets.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .data import Entity

WORD = re.compile(r"\w+|[^\w\s]")
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
L2I = {lab: i for i, lab in enumerate(LABELS)}


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning settings (fixed a priori; the epoch with the best development F1 is kept)."""

    model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    revision: str = "d673b8835373c6fa116d6d8006b33d48734e305d"
    epochs: int = 2
    lr: float = 5e-5
    batch: int = 16
    warmup: float = 0.1
    weight_decay: float = 0.01
    max_words: int = 80
    max_length: int = 256
    noise_prob: float = 0.0  # share of training documents replaced by a noisy copy in every epoch
    noise_cer: tuple[float, float] = (0.005, 0.25)  # target CER range of noisy copies (log-uniform)
    subsample: int = 0  # quick mode: number of training documents


def words(text: str) -> list[tuple[int, int]]:
    """Character spans of the words of ``text``."""
    return [m.span() for m in WORD.finditer(text)]


def bio(spans: list[tuple[int, int]], entities: list[Entity]) -> list[int]:
    """Label ids per word: the first word overlapping a mention gets B-, the following overlapping words I-."""
    labels = [0] * len(spans)
    j = 0
    for e in sorted(entities, key=lambda e: e.start):
        first = True
        while j < len(spans) and spans[j][1] <= e.start:
            j += 1
        k = j
        while k < len(spans) and spans[k][0] < e.end:
            if labels[k] == 0:
                labels[k] = L2I[("B-" if first else "I-") + e.type]
            first = False
            k += 1
    return labels


def decode(spans: list[tuple[int, int]], labels: list[int]) -> list[Entity]:
    """Mentions from word labels (an I- label that does not continue a mention of its type starts one)."""
    out: list[Entity] = []
    cur: list | None = None
    for (s, e), lab in zip(spans, labels):
        name = LABELS[lab]
        if name == "O":
            cur = None
            continue
        tag, typ = name.split("-")
        if tag == "I" and cur is not None and cur[2] == typ:
            cur[1] = e
        else:
            cur = [s, e, typ]
            out.append(cur)
    return [Entity(s, e, t) for s, e, t in out]


def boundary_alignment(text: str, entities: list[Entity]) -> float:
    """Share of mentions whose both edges fall on word boundaries (the ceiling for exact-match word tagging)."""
    starts = {s for s, _ in words(text)}
    ends = {e for _, e in words(text)}
    return float(np.mean([e.start in starts and e.end in ends for e in entities])) if entities else 1.0


def windows(spans: list[tuple[int, int]], text: str, max_words: int) -> list[tuple[int, int]]:
    """Cut word indices into [i, j) windows of at most ``max_words``, preferably right after a full stop."""
    out, i, n = [], 0, len(spans)
    while i < n:
        j = min(n, i + max_words)
        if j < n:
            stops = [k + 1 for k in range(i + max_words // 3, j) if text[spans[k][0]:spans[k][1]] == "."]
            if stops:
                j = stops[-1]
        out.append((i, j))
        i = j
    return out


def encode(tokenizer, text: str, entities: list[Entity] | None, max_words: int, max_length: int) -> list[dict]:
    """Tokenised windows of one document: input ids, the word index of each first sub-word, and labels."""
    spans = words(text)
    labels = bio(spans, entities) if entities is not None else None
    out = []
    for i, j in windows(spans, text, max_words):
        enc = tokenizer([text[s:e] for s, e in spans[i:j]], is_split_into_words=True, truncation=True,
                        max_length=max_length)
        word_ids = enc.word_ids()
        first = [-1] * len(word_ids)
        seen = set()
        for t, w in enumerate(word_ids):
            if w is not None and w not in seen:
                seen.add(w)
                first[t] = i + w
        lab = [labels[w] if (labels is not None and w >= 0) else -100 for w in first]
        out.append({"input_ids": enc["input_ids"], "first": first, "labels": lab, "n_words": j - i,
                    "covered": len(seen)})
    return out


def _batches(examples: list[dict], batch: int, shuffle: bool, rng: np.random.Generator | None = None):
    order = np.arange(len(examples))
    if shuffle:
        rng.shuffle(order)
    else:
        order = sorted(order, key=lambda k: len(examples[k]["input_ids"]))
    for b in range(0, len(order), batch):
        part = [examples[k] for k in order[b:b + batch]]
        width = max(len(x["input_ids"]) for x in part)
        ids = torch.zeros((len(part), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for r, x in enumerate(part):
            n = len(x["input_ids"])
            ids[r, :n] = torch.tensor(x["input_ids"])
            mask[r, :n] = 1
            labels[r, :n] = torch.tensor(x["labels"])
        yield part, ids, mask, labels


def load_model(cfg: TrainConfig, device: torch.device, state: dict | None = None):
    """Tokenizer and token-classification model (pretrained encoder, new head, or a saved fine-tuned state)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model, revision=cfg.revision, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    if state is not None:
        model.load_state_dict(state)
    return tok, model.to(device)


@torch.no_grad()
def predict(model, tok, texts: list[str], cfg: TrainConfig, device: torch.device, batch: int = 64) -> list[list[Entity]]:
    """Predicted mentions (character offsets) for every text."""
    model.eval()
    examples, owner = [], []
    for d, text in enumerate(texts):
        for x in encode(tok, text, None, cfg.max_words, cfg.max_length):
            examples.append(x)
            owner.append(d)
    for k, x in enumerate(examples):
        x["k"] = k
    word_labels = [np.zeros(len(words(t)), dtype=int) for t in texts]
    for part, ids, mask, _ in _batches(examples, batch, shuffle=False):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        pred = logits.float().argmax(-1).cpu().numpy()
        for r, x in enumerate(part):
            d = owner[x["k"]]
            for t, w in enumerate(x["first"]):
                if w >= 0:
                    word_labels[d][w] = pred[r, t]
    return [decode(words(t), list(lab)) for t, lab in zip(texts, word_labels)]


def train(cfg: TrainConfig, make_epoch: Callable[[int], list[tuple[str, list[Entity]]]],
          dev: list[tuple[str, list[Entity]]], evaluate: Callable[[list[list[Entity]]], float], device: torch.device,
          seed: int = 0, log: Callable[[str], None] = print) -> tuple[dict, pd.DataFrame]:
    """Fine-tune with AdamW, linear warm-up/decay and bfloat16 autocast; keep the epoch with the best dev F1.

    ``make_epoch(epoch)`` returns the (text, mentions) training documents of that epoch, which lets noise
    augmentation draw fresh noisy copies every epoch with the same number of optimisation steps as clean training.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, model = load_model(cfg, device)
    docs0 = make_epoch(0)
    steps_per_epoch = math.ceil(sum(len(encode(tok, t, e, cfg.max_words, cfg.max_length)) for t, e in docs0) / cfg.batch)
    total = steps_per_epoch * cfg.epochs
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    warm = max(1, int(cfg.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    best, best_f1, hist = None, -1.0, []
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        docs = docs0 if epoch == 0 else make_epoch(epoch)
        examples = [x for t, e in docs for x in encode(tok, t, e, cfg.max_words, cfg.max_length)]
        losses = []
        for _, ids, mask, labels in _batches(examples, cfg.batch, shuffle=True, rng=rng):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
        f1 = evaluate(predict(model, tok, [t for t, _ in dev], cfg, device))
        hist.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_f1": f1, "steps": len(losses),
                     "seconds": time.perf_counter() - t0})
        log(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f}, dev strict F1 {f1:.4f} ({time.perf_counter() - t0:.0f} s)")
        if f1 > best_f1:
            best_f1 = f1
            best = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    return best, pd.DataFrame(hist)
