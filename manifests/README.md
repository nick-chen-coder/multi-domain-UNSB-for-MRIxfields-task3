# Public data manifests

These files are copies of the records supplied with the final Task 3 audit materials.

| File | Contents |
| --- | --- |
| [data_split.json](data_split.json) | All 1,887 training-volume records and the 97 retrospective volumes excluded from training and validation of the present method. |
| [retrospective_reserved_volumes.csv](retrospective_reserved_volumes.csv) | The 97 reserved retrospective filenames, unchanged from the audit CSV. |
| [preprocessing_complete.json](preprocessing_complete.json) | Completed preprocessing inventory, including slice counts, foreground thresholds and per-volume measurements. |
| [preprocessing_20260904_183636.json](preprocessing_20260904_183636.json) | The timestamped copy of the same completed preprocessing inventory. Both names are retained from the audit materials. |

## Path conversion

- Each JSON `source_path` is relative to `paths.raw_data_root` in [configs/paths.yaml](../configs/paths.yaml), for example `Training_retrospective/T1W/0.1T/R_T1W_0.1T_0001.nii.gz`.
- The preprocessing JSON `config` path is relative to the repository root: `configs/preprocess.yaml`.
- All other JSON values are preserved, including the original configuration hash, volume assignments, historical cache labels, measurements and training-pipeline validation-status fields. The reserved-volume CSV is byte-identical to the source audit file.

The preprocessing inventories include the 97 unused volumes and therefore contain 1,984 volumes and 722,176 slices. Training used 1,842 retrospective volumes and 45 prospective volumes. The historical `val` label on the 97 reserved volumes is a cache label, not evidence that these volumes were used for model validation.

See [data-split.md](../data-split.md) for the training subsets, the history of the 97 reserved volumes, and the separate validation/model-selection procedures.
