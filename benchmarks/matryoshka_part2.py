"""
Matryoshka check for Part 2: retrains the final classifier on embeddings truncated to
768, 512, 256, 128 and 64 dimensions (rescaled to length 1) and reports test scores.

Same protocol as Part 2's robustness check (grouped splits, anchors + positives, 5 seeds),
from benchmarks/part2_protocol.py. No API calls: it reads the saved embeddings.
Results are discussed in addendum_matryoshka.ipynb.

Run from the project root:
  python benchmarks/matryoshka_part2.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.part2_protocol import X_anchor, X_positive, make_splits, train_and_test, y_all  # noqa: E402

DIMS = [768, 512, 256, 128, 64]


def truncate(E, d):
    """Keep the first d numbers of each embedding and rescale it to length 1, as the API would."""
    return np.ascontiguousarray(E[:, :d] / np.linalg.norm(E[:, :d], axis=1, keepdims=True))


def run_experiment(dims=DIMS, seeds=range(5)):
    """One row per (dimension, seed): test scores of the network and of kNN (k=10)."""
    rows = []
    for d in dims:
        X_all, X_pos = truncate(X_anchor, d), truncate(X_positive, d)
        for s in seeds:
            tr, va, te = make_splits(s)
            model, acc, f1, _ = train_and_test(tr, va, te, s, X_all, X_pos)
            knn = KNeighborsClassifier(10, metric="cosine", weights="distance").fit(X_all[tr], y_all[tr])
            knn_pred = knn.predict(X_all[te])
            rows.append(dict(dim=d, seed=s, nn_acc=acc, nn_f1=f1,
                             knn_acc=accuracy_score(y_all[te], knn_pred),
                             knn_f1=f1_score(y_all[te], knn_pred, average="macro"),
                             nn_weights=sum(p.numel() for p in model.parameters())))
    return pd.DataFrame(rows)


def summarize(res):
    return res.groupby("dim", sort=False).agg(
        nn_acc=("nn_acc", "mean"), nn_acc_std=("nn_acc", "std"), nn_f1=("nn_f1", "mean"),
        nn_f1_std=("nn_f1", "std"), knn_acc=("knn_acc", "mean"), knn_f1=("knn_f1", "mean"),
        nn_weights=("nn_weights", "first"))


if __name__ == "__main__":
    print(summarize(run_experiment()).round(3).to_string())
