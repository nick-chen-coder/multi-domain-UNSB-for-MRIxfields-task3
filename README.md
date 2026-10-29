# Multi-domain UNSB for MRIxFields 2026 Task 3

Training and inference code for team **Tryer**, Task 3, submission UID **47**.

The method uses a field- and contrast-conditioned 2D gated residual U-Net with a five-step neural Schrodinger bridge. A shared generator handles T1W, T2W and T2FLAIR across 0.1T, 1.5T, 3T, 5T and 7T. Training consists of 30,000 unpaired steps followed by 20,000 paired fine-tuning steps. Inference uses the final EMA generator.

## Repository layout

```text
configs/             Training, preprocessing and inference configuration
mrixfields_unsb/     Models, losses, data handling and training implementation
scripts/             Python command-line entry points
sh/                  Workflow scripts
tests/               Existing core tests
submission/          Manifest-driven final Docker inference implementation
manifests/           Public copies of data-split and preprocessing records
data-split.md        Training subsets, reserved volumes and model selection
```

The source code from the audit package's `task3/code/` directory is stored directly at this repository root. For example, audit path `code/configs/train_unsb.yaml` corresponds to repository path `configs/train_unsb.yaml`.

## Data split and model selection

Unpaired training used 1,842 official retrospective volumes from 1,029 subjects; paired fine-tuning used 45 prospective volumes from subjects 0006, 0007 and 0009. Another 97 retrospective volumes retained an earlier reserved split and were used for neither training nor validation of the present method.

See [data-split.md](data-split.md) for the complete data-use and model-selection description, and [manifests/](manifests/) for the public volume lists and preprocessing records. Public manifest paths are relative; the original local audit materials are unchanged.

## Final model weights

The final EMA checkpoint is available as a release attachment:

- [UID 47 final-model release](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/tag/uid47-final)
- [Download paired_ema.pt](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/download/uid47-final/paired_ema.pt) (151,603,859 bytes)
- [SHA256SUMS](https://github.com/nick-chen-coder/multi-domain-UNSB-for-MRIxfields-task3/releases/download/uid47-final/SHA256SUMS)

```text
8dcc749a314aa85070271c04cb21d5f72e18935c5fc84f1116097f9f083996eb  paired_ema.pt
```

For the training-code inference workflow, save the checkpoint as `exported/paired_ema.pt` under your configured `paths.artifacts_root`. This is the same checkpoint stored as `/app/model/final.pt` in the final submitted Docker image. The release checkpoint contains the inference EMA generator rather than a complete optimizer checkpoint for resuming training.

## Environment

The recorded training environment uses Python 3.12, PyTorch 2.12.1 and CUDA 13.0. Install the pinned dependencies and package from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Alternatively, use the supplied `environment.yml`.

## Configure data and output paths

Edit `configs/paths.yaml` for your machine. The checked-in paths retain the training configuration; replace them with your own directories.

The training pipeline expects the official `Training_retrospective` and `Training_prospective` directories under `paths.raw_data_root`. Paired fine-tuning uses prospective training subjects `0006`, `0007` and `0009`. Configure `paths.inference_input_root` separately for the official public validation inputs.

The pipeline expects native volumes of shape `(364, 436, 364)` with finite intensities in `[0, 1]`. It extracts axial slices and pads them for the generator. A complete uncompressed slice cache requires substantial disk space; the preprocessing script checks available capacity before a full fresh run.

## Run training and public validation inference

From the repository root:

```bash
bash sh/run_all.sh fresh
```

To resume an interrupted workflow:

```bash
bash sh/run_all.sh resume
```

Or run individual stages:

```bash
bash sh/01_preprocess.sh fresh
bash sh/02_train.sh fresh
bash sh/03_infer.sh fresh
bash sh/04_package.sh
```

The main output paths, relative to `paths.task_root`, are:

```text
artifacts/weights/unsb/final.pt
artifacts/weights/paired/final.pt
artifacts/exported/paired_ema.pt
artifacts/inference_outputs/
artifacts/packages/task3.zip
```

The ZIP workflow targets the public validation submission. The final hidden-test submission uses the manifest-driven entry point below.

## Final submitted Docker inference

Image:

```text
docker.synapse.org/syn76892337/task3:v2
```

Verified Linux/amd64 manifest digest:

```text
sha256:7be4b9ebdc0e53baaa5dd24a1b0a0ec739c5099b6bc8cb4c603e4fed3518f74f
```

The `submission/` directory contains the application Python files extracted from this image. Its entry point is `submission/inference.py`, corresponding to `/app/inference.py` in the image. It reads `/input/manifest.json` and writes the exact requested Task 3 predictions to `/output`. The default manifest contains 120 full-volume requests. Production inference requires an NVIDIA GPU.

With Synapse pull access and the official input directory available:

```bash
docker pull docker.synapse.org/syn76892337/task3:v2
docker run --rm --gpus all \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  docker.synapse.org/syn76892337/task3:v2
```

Input and output directories must follow the official manifest layout. Model weights are distributed in the submitted image and as a release attachment. The Git history contains code, configurations, documentation and data manifests; MRI datasets, generated predictions, binary checkpoints and audit logs are not stored in the Git tree.

The image checkpoint `/app/model/final.pt` and the audit export `paired_ema.pt` have the same SHA256:

```text
8dcc749a314aa85070271c04cb21d5f72e18935c5fc84f1116097f9f083996eb
```

## Main references

1. Kim Y, et al. DMD-augmented Unpaired Neural Schrodinger Bridge for Ultra-Low Field MRI Enhancement. 2026. https://arxiv.org/abs/2603.03769
2. Kim B, Kwon G, Kim K, Ye JC. Unpaired Image-to-Image Translation via Neural Schrodinger Bridge. ICLR 2024. https://arxiv.org/abs/2305.15086
3. Park T, Efros AA, Zhang R, Zhu JY. Contrastive Learning for Unpaired Image-to-Image Translation. ECCV 2020. https://arxiv.org/abs/2007.15651
4. Zhang R, et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018. https://arxiv.org/abs/1801.03924
5. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
6. Belghazi MI, et al. Mutual Information Neural Estimation. ICML 2018. https://proceedings.mlr.press/v80/belghazi18a.html

The final training implementation uses anatomical structure preservation constraints inspired by reference 1; its loss configuration is documented in `configs/train_unsb.yaml` and `configs/train_paired.yaml`.
