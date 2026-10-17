"""
Ablation benchmark for optimized_embedder.py: adds one optimization at a time
and records the wall-clock time of each step to logs/ablation.json.

Run from the project root (needs GPU_MODEL_SERVER_API_KEY):
  python benchmarks/ablation.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optimized_embedder as oe  # noqa: E402

texts_all = pd.read_csv("data/endpoints_data.csv")["anchor"].tolist()
texts_unique = list(dict.fromkeys(texts_all))
make_session = oe.make_session


def without_gzip(api_key, pool_size):
    session = make_session(api_key, pool_size)
    session.headers["Accept-Encoding"] = "identity"
    return session


steps = [
    # (label, texts, batch_size, workers, gzip)
    ("Keep-alive + no sleep (1 text/request, sequential, no dedup)", texts_all, 1, 1, True),
    ("+ deduplication", texts_unique, 1, 1, True),
    ("+ batching (128 texts/request, sequential)", texts_unique, 128, 1, True),
    ("+ concurrency (8 parallel requests) = final", texts_unique, 128, 8, True),
    ("final without gzip (for comparison)", texts_unique, 128, 8, False),
]

results = []
for label, texts, batch_size, workers, gzip in steps:
    oe.make_session = make_session if gzip else without_gzip
    t0 = time.perf_counter()
    _, n_requests = oe.embed_texts(texts, batch_size=batch_size, workers=workers)
    elapsed = time.perf_counter() - t0
    results.append(dict(step=label, texts=len(texts), requests=n_requests, batch_size=batch_size,
                        workers=workers, gzip=gzip, seconds=round(elapsed, 2)))
    print(f"{elapsed:8.2f} s  {label}")

Path("logs").mkdir(exist_ok=True)
Path("logs/ablation.json").write_text(json.dumps(results, indent=2))
