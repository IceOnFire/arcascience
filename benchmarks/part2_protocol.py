"""
Part 2's data, splits, model and training loop, shared by the addendum experiments
(benchmarks/matryoshka_part2.py, benchmarks/clean_training_rows.py and their notebooks).

The code is copied from part2.ipynb; the only change is that the network's input width
follows the data (in_dim), so it also works on truncated embeddings.
Paths are relative: import it with the project root as the working directory.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

torch.set_num_threads(4)

df = pd.read_csv("data/endpoints_data.csv")
X_anchor = np.load("embeddings/anchor.npy").astype(np.float32)
X_positive = np.load("embeddings/positive.npy").astype(np.float32)
CLASSES = df["case"].value_counts().index.tolist()             # most frequent first
y_all = df["case"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups = df["anchor"].str.lower().str.strip().to_numpy()        # rows with the same group stay together
pos_groups = df["positive"].str.lower().str.strip().to_numpy()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def grouped_split(idx, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(sgkf.split(idx, y_all[idx], groups[idx]))
    return idx[rest], idx[held]


def dedup(ix):
    """Keep one row per distinct (anchor, category)."""
    return df.iloc[ix].drop_duplicates(["anchor", "case"]).index.to_numpy()


def make_splits(seed):
    """Part 2, Section 1: grouped train / validation / test row indices."""
    trval, test = grouped_split(np.arange(len(df)), n_splits=7, seed=seed)
    train, val = grouped_split(trval, n_splits=6, seed=seed)
    return train, dedup(val), dedup(test)


class EndpointClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 128), n_classes=9, dropout=0.3, input_dropout=0.1):
        super().__init__()
        layers, prev = [nn.Dropout(input_dropout)], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y, power=0.5):
    w = 1.0 / np.bincount(y, minlength=len(CLASSES)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


DEFAULT_CFG = dict(hidden=(512, 128), dropout=0.3, input_dropout=0.1, lr=1e-3, weight_decay=1e-2,
                   batch_size=64, max_epochs=60, patience=10, weight_power=0.5, label_smoothing=0.05)


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    """Return (loss, accuracy, macro-F1, predictions) of `model` on X, y."""
    model.eval()
    logits = model(torch.from_numpy(X))
    loss = loss_fn(logits, torch.from_numpy(y)).item()
    pred = logits.argmax(1).numpy()
    return loss, accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


def train_model(X_tr, y_tr, X_va, y_va, cfg=DEFAULT_CFG, seed=42):
    """Part 2, Section 4: AdamW, early stopping on validation macro-F1, best epoch restored."""
    set_seed(seed)
    model = EndpointClassifier(in_dim=X_tr.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"],
                               input_dropout=cfg["input_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr, cfg["weight_power"]),
                                        label_smoothing=cfg["label_smoothing"])
    eval_loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    gen = torch.Generator().manual_seed(seed)
    best_f1, best_state, bad_epochs = -1, None, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for batch in torch.randperm(len(Xt), generator=gen).split(cfg["batch_size"]):
            opt.zero_grad()
            train_loss_fn(model(Xt[batch]), yt[batch]).backward()
            opt.step()
        va_loss, _, va_f1, _ = evaluate(model, X_va, y_va, eval_loss_fn)
        sched.step(va_loss)
        if va_f1 > best_f1:
            best_f1, best_state, bad_epochs = va_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model


def train_and_test(tr, va, te, seed, X_all=X_anchor, X_pos=X_positive):
    """Part 2's final setup: train on anchors + positives of `tr`, early-stop on `va`, score on `te`.
    Positives whose text is a validation/test anchor are dropped (Part 2, Section 8).
    Returns (model, test accuracy, test macro-F1, test predictions)."""
    held_out = set(groups[np.concatenate([va, te])])
    keep = tr[~np.isin(pos_groups[tr], list(held_out))]
    Xt = np.concatenate([X_all[tr], X_pos[keep]])
    yt = np.concatenate([y_all[tr], y_all[keep]])
    model = train_model(Xt, yt, X_all[va], y_all[va], seed=seed)
    _, acc, f1, pred = evaluate(model, X_all[te], y_all[te], nn.CrossEntropyLoss())
    return model, acc, f1, pred
