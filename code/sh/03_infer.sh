#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-fresh}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PATHS < <(python "$CODE_ROOT/scripts/resolved_paths.py" --config "$CODE_ROOT/configs/paths.yaml" logs_root artifacts_root)
LOGS_ROOT="${PATHS[0]}"
ARTIFACTS_ROOT="${PATHS[1]}"
mkdir -p "$LOGS_ROOT/inference" "$ARTIFACTS_ROOT/exported"
cd "$LOGS_ROOT"

case "$MODE" in
  fresh) EXTRA=() ;;
  resume) EXTRA=(--resume) ;;
  *) echo "usage: $0 [fresh|resume]" >&2; exit 2 ;;
esac

python -u "$CODE_ROOT/scripts/export_model.py" \
  --source "$ARTIFACTS_ROOT/weights/paired/final.pt" \
  --output "$ARTIFACTS_ROOT/exported/paired_ema.pt"

python -u "$CODE_ROOT/scripts/inference.py" \
  --config "$CODE_ROOT/configs/inference.yaml" \
  --device cuda "${EXTRA[@]}"

python -u "$CODE_ROOT/scripts/validate_predictions.py" \
  --config "$CODE_ROOT/configs/inference.yaml"
