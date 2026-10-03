# select_prototypes.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def allocate_per_class_counts(y: np.ndarray, M: int) -> Dict[int, int]:
    """
    Allocate M prototypes across existing classes as evenly as possible.
    """
    classes = sorted(np.unique(y).tolist())
    num_classes = len(classes)
    base = M // num_classes
    rem = M % num_classes
    counts = {}
    for i, c in enumerate(classes):
        counts[c] = base + (1 if i < rem else 0)
    return counts


def select_random(X: np.ndarray, M: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    if M >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=M, replace=False).astype(np.int64)


def select_kmeans_per_class(
    X: np.ndarray,
    y: np.ndarray,
    M: int,
    seed: int,
    kmeans_batch: int = 2048,
) -> np.ndarray:
    """
    Per-class MiniBatchKMeans with k clusters per class; pick nearest point to each centroid.
    Returns indices into original X/y of size M (trim/pad if needed).
    """
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(y).tolist())
    counts = allocate_per_class_counts(y, M)

    selected: List[int] = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        X_c = X[idx_c]
        k = counts[c]
        if k <= 0:
            continue
        if k >= len(idx_c):
            selected.extend(idx_c.tolist())
            continue

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=kmeans_batch,
            n_init=10,
            reassignment_ratio=0.01,
            max_iter=200,
        )
        labels = km.fit_predict(X_c)
        centers = km.cluster_centers_

        for j in range(k):
            members_local = np.where(labels == j)[0]
            if len(members_local) == 0:
                # rare fallback
                selected.append(int(rng.choice(idx_c, size=1, replace=False)[0]))
                continue

            X_mem = X_c[members_local]
            center = centers[j:j+1]  # (1, d)
            d2 = np.sum((X_mem - center) ** 2, axis=1)
            best_local = members_local[int(np.argmin(d2))]
            selected.append(int(idx_c[best_local]))

    # unique, preserve order
    selected = list(dict.fromkeys(selected))

    if len(selected) > M:
        selected = selected[:M]
    elif len(selected) < M:
        remaining = np.setdiff1d(np.arange(len(X)), np.array(selected, dtype=np.int64), assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(M - len(selected), len(remaining)), replace=False)
            selected.extend(extra.tolist())

    return np.array(selected, dtype=np.int64)
