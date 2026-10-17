"""
Profiles the embedding API to explain where time goes and how batching affects results.
Saves everything to logs/profile.json, which the Part 1B notebook displays.

Run from the project root (needs GPU_MODEL_SERVER_API_KEY), with no other traffic to the API:
  python benchmarks/profile_api.py
"""

import gzip
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optimized_embedder as oe  # noqa: E402

API_KEY = os.environ["GPU_MODEL_SERVER_API_KEY"]
df = pd.read_csv("data/endpoints_data.csv")
unique_texts = list(dict.fromkeys(df["anchor"]))
naive = np.load("embeddings/naive/anchor.npy")       # one text per request, sequential
first_row = {text: i for i, text in reversed(list(enumerate(df["anchor"])))}
profile = {}


def post(session_or_module, texts, encoding="gzip"):
    """One request; returns (embeddings, round-trip seconds, server metadata, bytes on the wire)."""
    t0 = time.perf_counter()
    r = session_or_module.post(
        oe.API_URL,
        json={"input": texts, "normalize": True, "model": oe.MODEL_NAME, "truncation_dimension": 768},
        headers={"X-API-Key": API_KEY, "Accept-Encoding": encoding}, timeout=60, stream=True)
    raw = r.raw.read(decode_content=False)                       # bytes as transferred on the wire
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    body = gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
    data = json.loads(body)
    return np.array(data["result"]), elapsed, data["metadata"], len(raw)


session = oe.make_session(API_KEY, pool_size=64)

# 1. Where does the time of a single request go?
post(session, [unique_texts[0]])                                 # warm-up (opens the connection)
rows = []
for text in unique_texts[1:11]:
    _, t_keepalive, meta, _ = post(session, [text])
    _, t_new_conn, _, _ = post(requests, [text])                 # module-level post = new connection
    rows.append(dict(round_trip_keepalive_s=t_keepalive, round_trip_new_connection_s=t_new_conn,
                     server_total_s=meta["total_time"], model_inference_s=meta["model_inference_time"]))
profile["single_request"] = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
print("1. single request (median of 10):", profile["single_request"])

# 2. Payload size and server time by batch size
rows = []
for n in [1, 32, 128, 256]:
    batch = unique_texts[:n]
    _, t_gzip, meta, bytes_gzip = post(session, batch, "gzip")
    _, t_plain, _, bytes_plain = post(session, batch, "identity")
    rows.append(dict(batch_size=n, server_total_s=meta["total_time"], round_trip_gzip_s=t_gzip,
                     round_trip_plain_s=t_plain, kb_gzip=bytes_gzip / 1024, kb_plain=bytes_plain / 1024))
profile["by_batch_size"] = rows
per_text_kb = rows[-1]["kb_plain"] / 256
profile["payload"] = dict(kb_per_embedding_plain=per_text_kb,
                          mb_total_plain=per_text_kb * len(unique_texts) / 1024,
                          gzip_ratio=rows[-1]["kb_plain"] / rows[-1]["kb_gzip"])
print("2. payload:", profile["payload"])

# 3. Batch size x workers sweep on the full set of unique anchors
rows = []
for batch_size, workers in [(32, 8), (64, 16), (128, 4), (128, 8), (128, 16), (256, 8)]:
    t0 = time.perf_counter()
    oe.embed_texts(unique_texts, batch_size=batch_size, workers=workers)
    rows.append(dict(batch_size=batch_size, workers=workers, seconds=time.perf_counter() - t0))
    print(f"3. sweep batch {batch_size:>3} x {workers:>2} workers: {rows[-1]['seconds']:.1f} s")
profile["sweep"] = rows

# 4. Determinism: when are embeddings bit-identical to the naive (single, sequential) ones?
sample = unique_texts[:200]
reference = naive[[first_row[t] for t in sample]]


def compare(embeddings):
    diff = np.abs(embeddings - reference).max(axis=1)
    return dict(identical_rows_pct=float((diff == 0).mean() * 100), max_abs_diff=float(diff.max()))


singles_sequential = np.stack([post(session, [t])[0][0] for t in sample])
in_batches = np.concatenate([post(session, sample[i:i + 100])[0] for i in range(0, len(sample), 100)])
with ThreadPoolExecutor(max_workers=64) as pool:
    singles_concurrent = np.stack(list(pool.map(lambda t: post(session, [t])[0][0], sample)))
profile["determinism"] = {
    "single text per request, sequential": compare(singles_sequential),
    "batches of 100": compare(in_batches),
    "single text per request, 64 concurrent": compare(singles_concurrent),
}
print("4. determinism vs naive:", json.dumps(profile["determinism"], indent=2))

Path("logs/profile.json").write_text(json.dumps(profile, indent=2))
print("Saved logs/profile.json")
