# Task 3 data split

- Training retrospective volumes: all records listed in `manifests/data_split.json` with split `train`.
- Training prospective subjects: `0006`, `0007`, `0009` (all available slices and fields).
- Validation subjects: none.
- The 97 deterministic retrospective reserved volumes reproduce the historical cache layout but are not read during training or validation.
- No validation, early stopping, hyperparameter tuning, or checkpoint selection is performed.
- Withdrawn paired validation data, `Validating_prospective`, and `Testing_prospective` are never read by preprocessing or training.
