#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-fresh}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "usage: $0 [fresh|resume]" >&2
  exit 2
fi

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PATHS < <(python "$CODE_ROOT/scripts/resolved_paths.py" --config "$CODE_ROOT/configs/paths.yaml" logs_root artifacts_root)
LOGS_ROOT="${PATHS[0]}"
ARTIFACTS_ROOT="${PATHS[1]}"
LOG_DIR="$LOGS_ROOT/workflow"
RUN_LOG="$LOG_DIR/run_${STAMP}.log"
mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$RUN_LOG")" "$LOG_DIR/latest.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "workflow_start_utc=$(date -u +%FT%TZ) mode=$MODE"
"$SCRIPT_DIR/01_preprocess.sh" "$MODE"
"$SCRIPT_DIR/02_train.sh" "$MODE"
"$SCRIPT_DIR/03_infer.sh" "$MODE"
"$SCRIPT_DIR/04_package.sh"
echo "workflow_complete_utc=$(date -u +%FT%TZ)"
echo "submission_zip=$ARTIFACTS_ROOT/packages/task3.zip"
