"""
Label-noise check for Part 2: removes suspicious rows from the training set only and
measures the effect on the untouched test sets. Also runs one deliberately wrong variant
that cleans validation and test too, to show how much that would inflate the scores.

Same protocol as Part 2's robustness check (grouped splits, anchors + positives, 5 seeds),
from benchmarks/part2_protocol.py. No API calls: it reads the saved embeddings.
Results are discussed in addendum_label_cleaning.ipynb.

Run from the project root:
  python benchmarks/clean_training_rows.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.part2_protocol import CLASSES, X_anchor, df, groups, make_splits, train_and_test, y_all  # noqa: E402

# Rows whose anchor text (lowercased, trimmed) carries more than one category (Part 1A, Section 1)
conflicting = df.groupby(groups)["case"].transform("nunique").to_numpy() > 1


def knn_outliers(ix, k=10):
    """Mark the rows of `ix` with no same-category row among their k nearest neighbours in `ix`.
    Copies of the row's own anchor text are skipped, otherwise a repeated anchor would count as its own neighbour."""
    _, nb = NearestNeighbors(n_neighbors=k + 20, metric="cosine").fit(X_anchor[ix]).kneighbors(X_anchor[ix])
    out = np.zeros(len(ix), bool)
    for r, row in enumerate(nb):
        others = [j for j in row if groups[ix[j]] != groups[ix[r]]][:k]
        out[r] = not (y_all[ix[others]] == y_all[ix[r]]).any()
    return out


def variants(tr, va, te):
    """(train, validation, test) row indices of each variant, for one split."""
    out_tr = knn_outliers(tr)
    return {
        "baseline (Part 2 final)": (tr, va, te),
        "train: drop conflicting labels": (tr[~conflicting[tr]], va, te),
        "train: drop kNN outliers": (tr[~out_tr], va, te),
        "train: drop both": (tr[~conflicting[tr] & ~out_tr], va, te),
        "WRONG: also drop kNN outliers from val + test": (tr[~out_tr], va[~knn_outliers(va)], te[~knn_outliers(te)]),
    }


def run_experiment(seeds=range(5)):
    """One row per (variant, seed): test scores, set sizes and the F1 of each category."""
    rows = []
    for s in seeds:
        for name, (tr, va, te) in variants(*make_splits(s)).items():
            _, acc, f1, pred = train_and_test(tr, va, te, s)
            per_class = f1_score(y_all[te], pred, average=None, labels=range(len(CLASSES)))
            rows.append(dict(variant=name, seed=s, acc=acc, f1=f1, train_rows=len(tr), test_items=len(te),
                             **{f"f1 {c}": v for c, v in zip(CLASSES, per_class)}))
    return pd.DataFrame(rows)


def summarize(res):
    return res.groupby("variant", sort=False).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"), f1=("f1", "mean"), f1_std=("f1", "std"),
        train_rows=("train_rows", "mean"), test_items=("test_items", "mean"))


def gains(res):
    """Per-seed accuracy change versus the baseline, in points."""
    acc = res.pivot(index="seed", columns="variant", values="acc")[res["variant"].unique()]
    return (acc.sub(acc.iloc[:, 0], axis=0) * 100).iloc[:, 1:]


if __name__ == "__main__":
    res = run_experiment()
    print(summarize(res).round(3).to_string())
    print("\nPer-seed accuracy change vs baseline (points):")
    print(gains(res).round(1).to_string())
