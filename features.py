# features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FeatureTransform:
    """
    Optional PCA transform fitted on training data only.
    """
    pca: Optional[PCA] = None

    def fit(self, X_train: np.ndarray, pca_dim: int, seed: int) -> None:
        if pca_dim and pca_dim > 0:
            self.pca = PCA(
                n_components=pca_dim,
                svd_solver="randomized",
                random_state=seed
            )
            self.pca.fit(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return X.astype(np.float32, copy=False)
        return self.pca.transform(X).astype(np.float32, copy=False)
