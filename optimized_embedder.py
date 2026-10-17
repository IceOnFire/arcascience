"""
Optimized embedding script for clinical endpoints.

Improvements over naive_embedder.py:
  1. Deduplication  - each unique text is embedded once, results are mapped back to rows.
  2. Batching       - many texts per request instead of one (the API accepts lists).
  3. Concurrency    - several batches in flight at once (the server is multi-GPU).
  4. Keep-alive     - a shared requests.Session reuses TCP/TLS connections.
  5. No sleep       - the artificial 50 ms delay per request is removed.
  6. Compression    - gzip responses (embeddings as JSON floats are ~16 KB each).
  7. Resilience     - timeouts, retries with exponential backoff on 429/5xx and
                      connection errors, and validation of every response.

Usage:
  python optimized_embedder.py                      # anchors -> embeddings/anchor.npy
  python optimized_embedder.py --include-triplets   # + positive.npy, negative.npy
  python optimized_embedder.py --exact              # bit-identical to the naive script (slower)
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://model-gpu-test.arcascience.ai/text-embedding"
MODEL_NAME = "arca-endpoint-embedder"


def make_session(api_key, pool_size):
    """HTTP session with connection pooling and automatic retries."""
    retry = Retry(
        total=5,
        backoff_factor=0.5,                       # 0.5s, 1s, 2s, 4s, ... between attempts
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),      # embedding requests are idempotent, safe to retry
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session = requests.Session()
    session.mount("https://", adapter)
    # Each embedding is ~16 KB of JSON text; gzip roughly halves transfer time, which is
    # the real bottleneck once batching removes per-request overhead. (requests sends
    # this by default; set explicitly so the optimization is visible and preserved.)
    session.headers.update({"X-API-Key": api_key, "Accept-Encoding": "gzip"})
    return session


def embed_batch(session, texts, dimension, timeout):
    """Embed one batch of texts. Raises on HTTP errors or malformed responses."""
    response = session.post(
        API_URL,
        json={
            "input": texts,
            "normalize": True,
            "model": MODEL_NAME,
            "truncation_dimension": dimension,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()["result"]
    if len(result) != len(texts):
        raise RuntimeError(f"API returned {len(result)} embeddings for {len(texts)} inputs")
    return result


def embed_texts(texts, dimension=768, batch_size=128, workers=8, timeout=60):
    """Embed a list of unique texts with batched, concurrent requests. Returns an array in input order."""
    api_key = os.getenv("GPU_MODEL_SERVER_API_KEY")
    if not api_key:
        raise ValueError("GPU_MODEL_SERVER_API_KEY environment variable not set")

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    with make_session(api_key, pool_size=workers) as session, ThreadPoolExecutor(max_workers=workers) as pool:
        # pool.map preserves batch order, so results line up with the input texts
        results = pool.map(lambda batch: embed_batch(session, batch, dimension, timeout), batches)
        embeddings = [emb for batch_result in results for emb in batch_result]
    return np.asarray(embeddings, dtype=np.float64), len(batches)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for clinical endpoints (optimized)")
    parser.add_argument("--csv", "-c", default="data/endpoints_data.csv", help="Path to endpoints CSV")
    parser.add_argument("--dimension", "-d", type=int, default=768, choices=[768, 512, 256, 128, 64],
                        help="Embedding dimension")
    parser.add_argument("--output-dir", "-o", default="embeddings", help="Directory for the .npy files")
    parser.add_argument("--include-triplets", action="store_true",
                        help="Also embed the positive and negative columns")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Texts per API request")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--timeout", type=float, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--exact", action="store_true",
                        help="Bit-identical output to naive_embedder.py: one text per request, one at a time. "
                             "The GPU server computes batched inputs (including concurrent requests it merges) "
                             "with slightly different floating-point results (cosine similarity > 0.999).")
    args = parser.parse_args()
    if args.exact:
        args.batch_size, args.workers = 1, 1

    t_start = time.perf_counter()
    df = pd.read_csv(args.csv)
    columns = ["anchor", "positive", "negative"] if args.include_triplets else ["anchor"]

    # Deduplicate across all requested columns: a text shared by an anchor and a
    # positive elsewhere is still embedded only once.
    all_texts = pd.concat([df[col] for col in columns], ignore_index=True)
    unique_texts = all_texts.drop_duplicates().tolist()
    t_loaded = time.perf_counter()

    print(f"Rows: {len(df)} | columns: {', '.join(columns)} | texts: {len(all_texts)} "
          f"| unique: {len(unique_texts)} ({1 - len(unique_texts) / len(all_texts):.0%} saved by dedup)")
    print(f"Dimension: {args.dimension} | batch size: {args.batch_size} | workers: {args.workers}")

    unique_embeddings, n_requests = embed_texts(unique_texts, args.dimension, args.batch_size,
                                                args.workers, args.timeout)
    t_embedded = time.perf_counter()

    # Map every row back to its text's embedding and save one file per column
    index_of = {text: i for i, text in enumerate(unique_texts)}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        rows = df[col].map(index_of).to_numpy()
        path = out_dir / f"{col}.npy"
        np.save(path, unique_embeddings[rows])
        print(f"Saved {path} {unique_embeddings[rows].shape}")
    t_end = time.perf_counter()

    api_time = t_embedded - t_loaded
    print("=" * 70)
    print("TIMING")
    print(f"  Load + dedup:   {t_loaded - t_start:8.2f} s")
    print(f"  API requests:   {api_time:8.2f} s  ({n_requests} requests, "
          f"{api_time / n_requests * args.workers:.2f} s avg per request)")
    print(f"  Map + save:     {t_end - t_embedded:8.2f} s")
    print(f"  TOTAL:          {t_end - t_start:8.2f} s")
    print(f"  Throughput:     {len(all_texts) / (t_end - t_start):8.0f} rows/s "
          f"({(t_end - t_start) / len(all_texts) * 1000:.2f} ms per row)")


if __name__ == "__main__":
    main()
