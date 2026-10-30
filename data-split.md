# Task 3 data split

- Unpaired training: 1,842 official retrospective volumes from 1,029 subjects, listed under `training` with cohort `R` in `manifests/data_split.json`.
- Paired fine-tuning: 45 official prospective volumes from subjects `0006`, `0007`, and `0009`.
- The 97 reserved retrospective volumes were used for neither training nor validation of the present method.

## Validation and model selection

- The 30,000-step unpaired schedule was selected using evaluations on prospective subjects `0006`, `0007`, and `0009`, considering nRMSE, SSIM, LPIPS, and visual quality. These subjects were subsequently used for paired fine-tuning.
- The 20,000-step paired schedule was selected using the official `Validating_prospective` set of 15 subjects, with no subject overlap with paired training, balancing nRMSE, SSIM, and LPIPS.
- The final recorded training run followed these selected schedules without in-loop validation or early stopping and exported the EMA generator at paired step 20,000.
- The empty `validation_subjects` list and `validation_status` fields in the audit manifests/configurations describe this final training run.
