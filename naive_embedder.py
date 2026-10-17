"""
Embedding script for clinical endpoints.

This is the BASELINE (naive) implementation. Do not modify this file.
Create optimized_embedder.py with your improvements.

This script generates embeddings for clinical trial endpoints using the
Arca endpoint embedding API.
"""

import os
import time

import numpy as np
import pandas as pd
import requests

# API configuration
API_URL = "https://model-gpu-test.arcascience.ai/text-embedding"
MODEL_NAME = "arca-endpoint-embedder"
API_KEY = os.getenv("GPU_MODEL_SERVER_API_KEY")

# Note: API server uses multi-GPU setup and handles concurrent requests efficiently


def get_embedding(endpoint_text, dimension=768):
    if not API_KEY:
        raise ValueError("GPU_MODEL_SERVER_API_KEY environment variable not set")

    headers = {"X-API-Key": API_KEY}
    response = requests.post(
        API_URL,
        json={
            "input": [endpoint_text],
            "normalize": True,
            "model": MODEL_NAME,
            "truncation_dimension": dimension
        },
        headers=headers
    )

    embeddings = response.json()["result"]
    return np.array(embeddings[0])


def embed_all_endpoints(csv_path, dimension=768, output_path="embeddings_naive.npy"):
    df = pd.read_csv(csv_path)
    embeddings = []

    print(f"Starting to embed {len(df)} endpoints...")
    print(f"Dimension: {dimension}")
    print("=" * 70)
    start_time = time.time()

    # Process each endpoint
    for idx, row in df.iterrows():
        endpoint = row['anchor']

        # Show progress for each endpoint
        print(f"Processing endpoint {idx+1}/{len(df)}: {endpoint[:60]}...")

        # Get embedding for this endpoint
        emb = get_embedding(endpoint, dimension)
        embeddings.append(emb)

        # Small delay to avoid overwhelming the API
        time.sleep(0.05)

        # Progress update every 10 endpoints
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            print(f"  → Progress: {idx+1}/{len(df)} ({elapsed:.1f}s elapsed, {avg_time:.3f}s per endpoint)")

    embeddings = np.array(embeddings)
    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"COMPLETED in {elapsed:.2f} seconds")
    print(f"Average time per endpoint: {elapsed/len(df):.3f} seconds")
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")

    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for clinical endpoints")
    parser.add_argument("--csv", "-c", default="data/endpoints_data.csv", help="Path to endpoints CSV")
    parser.add_argument("--dimension", "-d", type=int, default=768, choices=[768, 512, 256, 128, 64],
                       help="Embedding dimension")
    parser.add_argument("--output", "-o", default="embeddings_naive.npy", help="Output path for embeddings")

    args = parser.parse_args()

    embeddings = embed_all_endpoints(args.csv, args.dimension, args.output)
    print("\nEmbedding generation complete!")
