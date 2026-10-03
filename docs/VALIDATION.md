# Validation and attribution

Checked locally on 2026-10-02 with Python 3.12.14 and NumPy 2.3.5:

- All five offline tests passed.
- All 60 historical trial records audited; saved means, standard deviations and interval widths match recomputation.
- Paired seed analysis generated for all three budgets.
- A real-data subset diagnostic ran end to end with dataset and selected-index hashes.
- Selected Python sources parsed successfully.

Not checked: full historical PCA/clustering reproduction, original environment versions, dataset-wide prototype timing, accelerator behavior or hosted CI.

Ashley Liu (Xinying Liu) confirmed independent completion of the original project. IDX utilities, class-balance diagnostic, tests, paired audit and documentation were added with AI assistance during portfolio preparation. Original code hashes and preserved results are in source-manifest.json.
