# Task 3 data split and model selection

## Training data

| Stage | Official data | Unique subjects | Volumes |
| --- | --- | ---: | ---: |
| Unpaired training | Training_retrospective | 1,029 | 1,842 |
| Paired fine-tuning | Training_prospective, subjects 0006, 0007 and 0009 | 3 | 45 |

Both training subsets cover T1W, T2W and T2FLAIR at 0.1, 1.5, 3, 5 and 7 T. The same data are reused across translation directions; counts are not multiplied by the number of directions.

Another **97 retrospective volumes** had been reserved for validation during earlier diffusion-model research. The present method did not use unpaired data for validation, but the earlier split was inadvertently retained instead of being updated. These 97 volumes were therefore used for **neither training nor validation of the present method**.

The complete volume assignments are in [manifests/data_split.json](manifests/data_split.json). The 97 filenames are also listed in [manifests/retrospective_reserved_volumes.csv](manifests/retrospective_reserved_volumes.csv).

## Split configuration and preprocessing records

[configs/preprocess.yaml](configs/preprocess.yaml) specifies the split seed, **20260809**, the retrospective reserved-volume quotas by contrast and field, and the three prospective training subjects. The selection algorithm is implemented in [mrixfields_unsb/preprocessing/pipeline.py](mrixfields_unsb/preprocessing/pipeline.py).

The reserved volumes retain the historical `split: "val"` cache label. This label identifies their preprocessing destination; it does not mean they were used to validate the present method. Training reads only `all_slices_train/retrospective` and `all_slices_train/prospective`, as specified in the two training configurations.

The preprocessing records cover **1,984 volumes and 722,176 axial slices**, including the 97 unused volumes. These preprocessing totals must not be interpreted as training-data totals.

## Validation and model selection

- **Unpaired stage:** Prospective subjects **0006, 0007 and 0009** (3 subjects, 45 volumes) were used to evaluate the unpaired model. The training length of **30,000 updates** was selected by considering nRMSE, SSIM, LPIPS and visual quality. These subjects were subsequently used for paired fine-tuning.
- **Paired stage:** The official **Validating_prospective** set contains **15 subjects**, yielding **180 prediction files**. Its subject IDs are 0001, 0002, 0003, 0004, 0005, 0008, 0010, 0011, 0012, 0013, 0014, 0015, 0016, 0017 and 0018, with no overlap with the three paired-training subjects. The EMA weights at paired fine-tuning step **20,000** were selected as a compromise across nRMSE, SSIM and LPIPS.

The final recorded training run followed the selected **30,000 unpaired + 20,000 paired** schedule. Its training entry points do not execute in-loop validation, early stopping or metric-based checkpoint selection. This execution detail is distinct from the experimental comparisons used to choose the training schedule.

The audit fields `validation_subjects: []` and `validation_status` in the original split record describe the absence of validation executed by that training pipeline; they do not document the separate model-selection experiments described above. Their original values are retained in the published record.

## Public manifest paths

The JSON and CSV records in [manifests/](manifests/) are public copies of the submitted audit records. Machine-specific absolute paths have been converted to paths relative to either the configured official raw-data root (`source_path`) or the repository root (`config`). Volume assignments, filenames, counts, recorded measurements and other metadata are unchanged. No MRI images are included.

The local source audit materials are preserved unchanged. See [manifests/README.md](manifests/README.md) for the file inventory and verification details.
