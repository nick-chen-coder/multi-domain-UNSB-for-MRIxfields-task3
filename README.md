# Multi-domain UNSB for MRIxFields 2026 Task 3

Code, data manifests and model weights for team **Tryer**, submission **UID 47**.

The method uses a field- and contrast-conditioned 2D gated residual U-Net with a five-step Unpaired Neural Schrödinger Bridge (UNSB). One shared generator handles T1W, T2W and T2FLAIR translation between 0.1, 1.5, 3, 5 and 7 T. Training consists of 30,000 unpaired updates followed by 20,000 paired fine-tuning updates. The final model uses the generator EMA weights.

## Repository structure

```text
code/                  Original training, preprocessing and inference code
  configs/             Configuration files
  mrixfields_unsb/      Models, losses, samplers and training implementation
  scripts/             Python entry points
  sh/                  Workflow scripts
  tests/               Existing core tests
  README.md            Original reproduction instructions
manifests/             Original data-split and preprocessing records
data-split.md          Original audit data-split note
README.md              This overview
```

The files under `code/`, `manifests/` and `data-split.md` preserve the local audit materials' directory structure and contents.

## Environment and paths

The recorded environment uses Linux, Python 3.12, PyTorch 2.12.1 and CUDA 13.0. From the repository root:

```bash
cd code
python -m pip install -r requirements.txt
python -m pip install -e .
```

Edit [code/configs/paths.yaml](code/configs/paths.yaml) for your machine. In particular:

- `raw_data_root`: the directory containing the official `Training_retrospective` and `Training_prospective` folders.
- `inference_input_root`: the official public validation input directory.
- `task_root` and `slices_root`: your experiment-output and slice-cache directories.

The checked-in configurations and manifests retain the original recorded paths. Official MRI data must be obtained separately. Input volumes have shape `(364, 436, 364)` with intensities in `[0,1]`.

## Final model weights

Download the checkpoint from [the UID 47 release](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/tag/uid47-final):

- [paired_ema.pt](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/download/uid47-final/paired_ema.pt), 151,603,859 bytes.
- [SHA256SUMS](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/download/uid47-final/SHA256SUMS).

```text
8dcc749a314aa85070271c04cb21d5f72e18935c5fc84f1116097f9f083996eb  paired_ema.pt
```

Save the checkpoint as `exported/paired_ema.pt` under the configured `paths.artifacts_root`. It is an inference export of the final EMA generator. It is identical to `/app/model/final.pt` in the submitted Docker image:

```text
docker.synapse.org/syn76892337/task3:v2
```

## Inference with the released checkpoint

After configuring paths and downloading the checkpoint, run from `code/`:

```bash
python scripts/inference.py --config configs/inference.yaml --device cuda
python scripts/validate_predictions.py --config configs/inference.yaml
```

This entry point performs full-volume inference for the official public validation input layout, producing 180 prediction files. Output locations are specified in [code/configs/inference.yaml](code/configs/inference.yaml). The submitted Docker image uses the challenge's separate manifest-driven test interface.

## Training

From `code/`, run the complete workflow:

```bash
bash sh/run_all.sh fresh
```

To resume:

```bash
bash sh/run_all.sh resume
```

The workflow runs preprocessing, unpaired training, paired fine-tuning, public validation inference and submission packaging. Stage settings are recorded in [code/configs/train_unsb.yaml](code/configs/train_unsb.yaml) and [code/configs/train_paired.yaml](code/configs/train_paired.yaml).

Unpaired training used 1,842 retrospective volumes from 1,029 subjects. Paired fine-tuning used 45 prospective volumes from subjects 0006, 0007 and 0009. An earlier reserved split of 97 retrospective volumes was retained; these volumes were used for neither training nor validation of the present method. Volume assignments are recorded in [manifests/data_split.json](manifests/data_split.json), with the reserved filenames in [manifests/retrospective_reserved_volumes.csv](manifests/retrospective_reserved_volumes.csv).

The 30,000-update unpaired schedule was selected using evaluations on prospective subjects 0006, 0007 and 0009, considering nRMSE, SSIM, LPIPS and visual quality. The 20,000-update paired schedule was selected using the official `Validating_prospective` set of 15 non-overlapping subjects, balancing nRMSE, SSIM and LPIPS. These experimental comparisons are separate from the final recorded training run, whose training entry points execute the selected schedule without in-loop validation.
