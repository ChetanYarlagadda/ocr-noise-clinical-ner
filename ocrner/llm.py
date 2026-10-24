"""OCR post-correction with a local LLM (Ollama ``qwen3:4b``), with every reply cached in SQLite.

The Ollama build of qwen3:4b is a thinking model whose chat template always opens a reasoning block; with
``think: false`` and free-form output it still writes its reasoning into the answer. Constraining the reply to a
JSON schema (``{"corrected_text": ...}``) makes it answer directly, so that is how thinking is switched off here.
Decoding is greedy (temperature 0, fixed seed).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SYSTEM = ("You correct OCR errors in scanned biomedical abstracts. Fix misrecognised characters, broken or merged "
          "words and punctuation. Do not paraphrase, summarise, reorder or add content; keep every word that is "
          "already correct. Return only the corrected text.")
SCHEMA = {"type": "object", "properties": {"corrected_text": {"type": "string"}}, "required": ["corrected_text"]}


class OllamaCorrector:
    """Calls ``/api/chat`` with the JSON schema; replies are cached under sha256 of the full request."""

    def __init__(self, cache_path: Path, model: str = "qwen3:4b", host: str = "http://localhost:11434",
                 num_ctx: int = 4096, timeout: int = 300) -> None:
        self.model, self.host, self.timeout = model, host, timeout
        self.options = {"temperature": 0.0, "seed": 0, "num_ctx": num_ctx}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, request TEXT, reply TEXT, "
                        "latency_s REAL, output_tokens INTEGER, created_utc TEXT)")
        self.hits = self.misses = 0
        self.new_seconds = 0.0

    def request(self, text: str) -> dict:
        """The Ollama request body for one OCR text (the cache key is computed from it)."""
        max_tokens = 64 + 2 * len(text) // 3  # generous: about 4 characters per token, plus the JSON wrapper
        return {"model": self.model, "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": text}],
                "format": SCHEMA, "think": False, "stream": False,
                "options": {**self.options, "num_predict": max_tokens}}

    def correct(self, text: str) -> tuple[str, float, bool]:
        """Corrected text, model seconds of the original call, and whether it came from the cache.

        If the reply is not valid JSON (for example cut off at the token limit) the OCR text is returned unchanged.
        """
        body = self.request(text)
        req = json.dumps(body, sort_keys=True)
        key = hashlib.sha256(req.encode()).hexdigest()
        row = self.db.execute("SELECT reply, latency_s FROM replies WHERE key = ?", (key,)).fetchone()
        if row is None:
            t0 = time.perf_counter()
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            out = r.json()
            latency = time.perf_counter() - t0
            self.db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?, ?)",
                            (key, req, out["message"]["content"], latency, out.get("eval_count"),
                             datetime.now(timezone.utc).isoformat(timespec="seconds")))
            self.db.commit()
            self.misses += 1
            self.new_seconds += latency
            row, cached = (out["message"]["content"], latency), False
        else:
            self.hits += 1
            cached = True
        try:
            fixed = json.loads(row[0])["corrected_text"]
        except (json.JSONDecodeError, KeyError, TypeError):
            fixed = text
        return fixed, float(row[1]), cached

    def total_seconds(self) -> tuple[int, float]:
        """Number of stored replies and their summed model seconds (all runs that used this cache)."""
        n, s = self.db.execute("SELECT count(*), coalesce(sum(latency_s), 0) FROM replies").fetchone()
        return int(n), float(s)
