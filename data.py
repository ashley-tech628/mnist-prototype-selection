# data.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_mnist_torchvision(root: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads MNIST via torchvision.
    Returns:
      X_train: (60000, 784), y_train: (60000,)
      X_test:  (10000, 784), y_test:  (10000,)
    """
    from torchvision.datasets import MNIST
    from torchvision import transforms

    tfm = transforms.ToTensor()
    train = MNIST(root=root, train=True, download=True, transform=tfm)
    test = MNIST(root=root, train=False, download=True, transform=tfm)

    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    return X_train, y_train, X_test, y_test


def load_mnist_csv(
    train_csv: str,
    test_csv: Optional[str] = None,
    test_size: int = 10000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supports Kaggle-style CSV:
      - train.csv: label + 784 pixel columns
      - test.csv:  784 pixel columns (often no labels). If no labels, we split from train.

    If test_csv exists and contains 'label', we'll use it directly.
    Otherwise we stratified-split from train.
    """
    df_tr = pd.read_csv(train_csv)
    if "label" not in df_tr.columns:
        raise ValueError("train_csv must contain a 'label' column")

    y = df_tr["label"].to_numpy(dtype=np.int64)
    X = df_tr.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0

    if test_csv is not None:
        df_te = pd.read_csv(test_csv)
        if "label" in df_te.columns:
            y_test = df_te["label"].to_numpy(dtype=np.int64)
            X_test = df_te.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
            return X, y, X_test, y_test

    # split from train
    test_frac = min(0.3, test_size / len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    if len(X_test) > test_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_test), size=test_size, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
    return X_train, y_train, X_test, y_test


def load_data(
    mode: str,
    data_root: str = "./data",
    csv_train: Optional[str] = None,
    csv_test: Optional[str] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified entry.
    mode: 'torchvision' or 'csv'
    """
    if mode == "torchvision":
        return load_mnist_torchvision(root=data_root)
    if mode == "csv":
        if not csv_train:
            raise ValueError("csv_train is required when mode='csv'")
        return load_mnist_csv(train_csv=csv_train, test_csv=csv_test, seed=seed)
    raise ValueError(f"Unknown mode: {mode}")
