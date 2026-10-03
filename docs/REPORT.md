# Technical report

The project examines reference-set compression for Euclidean 1-nearest-neighbor classification. Per-class MiniBatchKMeans selects actual observed examples, providing geometric coverage with approximately balanced class quotas. Random selection is the simpler baseline.

## Historical evidence

There are 60 saved records: two methods, three budgets and ten trials. Summary means, sample standard deviations and t intervals recompute from raw records. The 1,000-prototype clustering mean is 93.711% versus 90.117% random, a 3.594 percentage-point gain. Paired trial-seed differences yield the interval recorded in paired_analysis.json. Shared test images mean that repetition measures seed sensitivity, not independent dataset sampling.

The valid nested source is selected because the outer runner has a syntax error. The selected code defaults to train-only PCA with 50 components; saved output lacks a configuration manifest, so its exact provenance cannot be fully recovered. The source-to-output association is therefore historical evidence, not an exact reproducibility claim.

## Confounds and statistical limits

Class-balanced clustering versus global random combines the effect of quotas with the effect of geometry. A class-balanced random comparator is needed to isolate the clustering contribution. Reusing the test split across budgets provides an exploratory comparison, not an untouched final holdout after model selection. Approximate t intervals assume a reasonably well-behaved distribution across seed repetitions and do not account for multiple budget comparisons.

Prototype feature storage is linear in budget. Batched 1-NN bounds the size of an individual distance matrix but still creates additional dot-product and arithmetic temporaries. No claim of process memory, inference speedup or retained full-set accuracy follows from prototype count alone.

## New diagnostic

A NumPy-only reader supports local uint8 IDX files and rejects malformed lengths. The new class-balanced random selector ensures unique indices and refills quotas when a class is short. On a seed-42 subset of 5,000 training and 1,000 test images, with 300 prototypes and no PCA, uniform random scored 82.1% and balanced random 81.4%. This is a one-seed diagnostic; it gives no robust ranking and uses a different setting from the historical sweep. Its unfavorable result is retained rather than selectively omitted.

Five tests passed for 1-NN equivalence, selection uniqueness/reproducibility, fallback behavior and file contracts. The original clustering pipeline was syntax checked but not executed during packaging because scikit-learn/Matplotlib are absent in the available runtime. GitHub CI was configured, not run here.

## Follow-up protocol

Use a fixed train/validation/test partition to compare global random, class-balanced random and per-class clustering at matched budgets and feature dimensions. Select PCA/cluster settings on validation. Report paired seed differences, confusion/error slices, selection runtime, repeated inference timing, peak memory and full-reference 1-NN results. None of these future comparisons is presented as completed.
