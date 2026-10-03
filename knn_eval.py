# knn_eval.py
from __future__ import annotations
import numpy as np


def knn_1nn_accuracy(
    X_proto: np.ndarray,
    y_proto: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch: int = 512,
) -> float:
    """
    1-NN accuracy with squared L2 distance using batching:
      ||x - p||^2 = ||x||^2 + ||p||^2 - 2 x·p
    """
    Xp = X_proto.astype(np.float32, copy=False)
    Xt = X_test.astype(np.float32, copy=False)

    proto_norm = np.sum(Xp * Xp, axis=1)  # (M,)
    correct = 0
    ntest = len(Xt)

    for i in range(0, ntest, batch):
        xb = Xt[i:i + batch]
        yb = y_test[i:i + batch]

        xb_norm = np.sum(xb * xb, axis=1)  # (B,)
        dots = xb @ Xp.T                    # (B, M)
        d2 = xb_norm[:, None] + proto_norm[None, :] - 2.0 * dots
        nn = np.argmin(d2, axis=1)
        pred = y_proto[nn]
        correct += int(np.sum(pred == yb))

    return correct / ntest
