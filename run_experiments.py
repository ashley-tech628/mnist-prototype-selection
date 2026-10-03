# run_experiments.py
from __future__ import annotations
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data
from features import FeatureTransform
from select_prototypes import select_random, select_kmeans_per_class
from knn_eval import knn_1nn_accuracy


def mean_std_ci95(accs: List[float]) -> Tuple[float, float, float]:
    """
    Return (mean, std, half_width_95ci) using t-interval.
    """
    a = np.array(accs, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0

    # 95% two-sided t critical values for small df; fallback to 1.96.
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086
    }
    n = len(a)
    if n <= 1:
        return mean, std, 0.0
    df = n - 1
    tcrit = t_table.get(df, 1.96)
    half = float(tcrit * std / np.sqrt(n))
    return mean, std, half


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["torchvision", "csv"], default="torchvision")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--csv_train", type=str, default=None)
    p.add_argument("--csv_test", type=str, default=None)

    p.add_argument("--Ms", type=int, nargs="+", default=[1000, 5000, 10000])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pca_dim", type=int, default=50, help="0 to disable PCA")
    p.add_argument("--batch", type=int, default=512, help="test batch size for 1-NN")
    p.add_argument("--kmeans_batch", type=int, default=2048, help="MiniBatchKMeans batch size")

    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--limit_train", type=int, default=0, help=">0 to subsample training set (debug)")
    p.add_argument("--limit_test", type=int, default=0, help=">0 to subsample test set (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(
        mode=args.data,
        data_root=args.data_root,
        csv_train=args.csv_train,
        csv_test=args.csv_test,
        seed=args.seed,
    )

    # optional subsample for debug speed
    rng = np.random.default_rng(args.seed)
    if args.limit_train and args.limit_train < len(X_train):
        idx = rng.choice(len(X_train), size=args.limit_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if args.limit_test and args.limit_test < len(X_test):
        idx = rng.choice(len(X_test), size=args.limit_test, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    # Fit transform on train only
    ft = FeatureTransform()
    ft.fit(X_train, pca_dim=args.pca_dim, seed=args.seed)
    Xtr = ft.transform(X_train)
    Xte = ft.transform(X_test)

    rows = []
    for M in args.Ms:
        for t in range(args.trials):
            trial_seed = args.seed + 1000 * t + M

            # Random baseline
            ridx = select_random(Xtr, M, seed=trial_seed)
            acc_r = knn_1nn_accuracy(Xtr[ridx], y_train[ridx], Xte, y_test, batch=args.batch)
            rows.append({"method": "random", "M": M, "trial": t, "seed": trial_seed, "acc": acc_r})

            # Our method
            kidx = select_kmeans_per_class(Xtr, y_train, M, seed=trial_seed, kmeans_batch=args.kmeans_batch)
            acc_k = knn_1nn_accuracy(Xtr[kidx], y_train[kidx], Xte, y_test, batch=args.batch)
            rows.append({"method": "kmeans_per_class", "M": M, "trial": t, "seed": trial_seed, "acc": acc_k})

            print(f"[M={M} trial={t}] random={acc_r:.4f} | kmeans_per_class={acc_k:.4f}")

    df = pd.DataFrame(rows)
    results_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(results_path, index=False)

    # summary
    summary_rows = []
    for (method, M), g in df.groupby(["method", "M"]):
        accs = g["acc"].tolist()
        mean, std, half = mean_std_ci95(accs)
        summary_rows.append({
            "method": method,
            "M": int(M),
            "mean_acc": mean,
            "std_acc": std,
            "ci95_halfwidth": half,
            "n_trials": len(accs),
        })
    sdf = pd.DataFrame(summary_rows).sort_values(["M", "method"])
    summary_path = os.path.join(args.outdir, "summary.csv")
    sdf.to_csv(summary_path, index=False)

    # plot
    plt.figure()
    for method in ["random", "kmeans_per_class"]:
        sub = sdf[sdf["method"] == method].sort_values("M")
        plt.errorbar(
            sub["M"].to_numpy(),
            sub["mean_acc"].to_numpy(),
            yerr=sub["ci95_halfwidth"].to_numpy(),
            marker="o",
            capsize=4,
            label=method
        )
    plt.xscale("log")
    plt.xlabel("Number of prototypes M (log scale)")
    plt.ylabel("Test accuracy (1-NN)")
    plt.title("Prototype Selection for 1-NN on MNIST")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "plot.png")
    plt.savefig(plot_path, dpi=200)

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
