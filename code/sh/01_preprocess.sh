#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-fresh}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PATHS < <(python "$CODE_ROOT/scripts/resolved_paths.py" --config "$CODE_ROOT/configs/paths.yaml" task_root logs_root slices_root)
TASK_ROOT="${PATHS[0]}"
LOGS_ROOT="${PATHS[1]}"
SLICES_ROOT="${PATHS[2]}"
mkdir -p "$LOGS_ROOT/audit-logs" "$LOGS_ROOT/preprocessing"
cd "$LOGS_ROOT"

case "$MODE" in
  fresh) EXTRA=() ;;
  resume)
    if [[ -f "$TASK_ROOT/manifests/preprocessing_complete.json" && -f "$SLICES_ROOT/preprocessing_config.json" ]]; then
      echo "[preprocessing] completion marker found; skipping completed preprocessing"
      exit 0
    fi
    EXTRA=(--resume)
    ;;
  *) echo "usage: $0 [fresh|resume]" >&2; exit 2 ;;
esac

python -u "$CODE_ROOT/scripts/preprocess.py" \
  --config "$CODE_ROOT/configs/preprocess.yaml" \
  "${EXTRA[@]}"
