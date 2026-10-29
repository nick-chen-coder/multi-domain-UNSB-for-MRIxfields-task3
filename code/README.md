# MRIxFields 2026 Task 3 复现代码

## 路径配置

只修改 `configs/paths.yaml`：

```yaml
paths:
  task_root: /root/autodl-tmp/UNSB_real_final/task3
  raw_data_root: /root/autodl-tmp/data/Training_data/release_20260414
  slices_root: /root/autodl-tmp/data/Slices
  inference_input_root: /root/autodl-tmp/data/Validating_prospective-encrypt/Validating_prospective
  logs_root: ${paths.task_root}/logs
  artifacts_root: ${paths.task_root}/artifacts
  manifests_root: ${paths.task_root}/manifests
```

训练只允许读取 `raw_data_root/Training_retrospective` 和官方训练集中的 `Training_prospective`。配对微调固定使用 `0006/0007/0009`。`Validating_prospective` 只在训练结束后用于生成在线验证提交，不参与训练、选模或早停；代码不读取撤回的20名志愿者全配对数据。

## 环境

复现实验使用 Python 3.12、PyTorch 2.12.1、CUDA 13.0。安装命令：

```bash
cd /root/autodl-tmp/UNSB_real_final/task3/code && python -m pip install -r requirements.txt && python -m pip install -e .
```

也可以使用：

```bash
cd /root/autodl-tmp/UNSB_real_final/task3/code && conda env create -f environment.yml && conda activate unsb_real_final
```

## 一键运行

流程依次执行：不压缩切片预处理、UNSB 训练30000步、配对微调20000步、公开验证集全体积推理、结果检查和验证提交包打包。

首次运行：

```bash
nohup bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/run_all.sh fresh >/dev/null 2>&1 &
```

中断后继续；已经完成的预处理或训练阶段会自动跳过：

```bash
nohup bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/run_all.sh resume >/dev/null 2>&1 &
```

实时日志：

```bash
tail -F /root/autodl-tmp/UNSB_real_final/task3/logs/workflow/latest.log
```

## 分步运行

```bash
bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/01_preprocess.sh fresh
bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/02_train.sh fresh
bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/03_infer.sh fresh
bash /root/autodl-tmp/UNSB_real_final/task3/code/sh/04_package.sh
```

日志目录：

```text
task3/logs/workflow/latest.log
task3/logs/unsb/full.log
task3/logs/paired/full.log
task3/logs/audit-logs/
```

正式训练启动后，不得删除、筛选、合并或改写 `task3/logs` 中的任何日志。预处理完成后自动生成 `task3/data-split.md` 和 `task3/manifests/data_split.json`。

## Checkpoint

UNSB 的 `latest.pt` 每2000步更新，永久保留10000、20000和最终30000步；配对微调的 `latest.pt` 每2500步更新，永久保留5000、10000、15000和最终20000步。`latest.pt` 保存完整优化器、调度器、采样器及随机状态，可精确断点续训。

## 输出

```text
task3/artifacts/weights/unsb/final.pt
task3/artifacts/weights/paired/final.pt
task3/artifacts/exported/paired_ema.pt
task3/artifacts/inference_outputs/full/task3/
task3/artifacts/packages/task3.zip
```

这里的 `task3.zip` 是180文件的公开验证排行榜提交包，不是隐藏测试 Docker 输出。最终 Docker 位于独立代码库，必须读取 `/input/manifest.json` 并生成清单要求的120个完整 `(364,436,364)` NIfTI。

在线验证成功后，将 Submission ID 和 Submission Name 追加到审计日志：

```bash
cd /root/autodl-tmp/UNSB_real_final/task3/code && python scripts/record_validation_submission.py --config configs/paths.yaml --submission-id YOUR_ID --submission-name YOUR_NAME
```

最终还需按官方要求签署并单独提交 Data Usage Declaration。

## 切片格式

输出目录和文件名保持原版规则。每个不压缩 NPZ 只包含 `image_u16`、`z_index`、`foreground_ratio`，不包含 Canny 或 latent。全量缓存约222 GiB，完整流程建议至少保留260 GiB可用空间。
