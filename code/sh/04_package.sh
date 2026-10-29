#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PATHS < <(python "$CODE_ROOT/scripts/resolved_paths.py" --config "$CODE_ROOT/configs/paths.yaml" logs_root)
LOGS_ROOT="${PATHS[0]}"
mkdir -p "$LOGS_ROOT/packaging"
cd "$LOGS_ROOT"

python -u "$CODE_ROOT/scripts/package_predictions.py" \
  --config "$CODE_ROOT/configs/inference.yaml" --clean
