#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-fresh}"

export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PATHS < <(python "$CODE_ROOT/scripts/resolved_paths.py" --config "$CODE_ROOT/configs/paths.yaml" logs_root artifacts_root)
LOGS_ROOT="${PATHS[0]}"
ARTIFACTS_ROOT="${PATHS[1]}"
mkdir -p "$LOGS_ROOT/audit-logs" "$LOGS_ROOT/unsb" "$LOGS_ROOT/paired"
cd "$LOGS_ROOT"

case "$MODE" in
  fresh)
    if find "$ARTIFACTS_ROOT/weights" -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .; then
      echo "existing checkpoints found; use resume or move them out of artifacts/weights" >&2
      exit 1
    fi
    RESUME=none
    ;;
  resume) RESUME=auto ;;
  *) echo "usage: $0 [fresh|resume]" >&2; exit 2 ;;
esac

run_stage() {
  local label="$1"
  local checkpoint_dir="$2"
  local entry="$3"
  local config="$4"
  local stage_resume="$RESUME"
  if [[ "$MODE" == "resume" && -f "$checkpoint_dir/final.pt" ]]; then
    echo "[$label] final.pt exists; skipping completed stage"
    return
  fi
  if [[ "$MODE" == "resume" && ! -f "$checkpoint_dir/latest.pt" ]]; then
    stage_resume=none
  fi
  python -u "$entry" --config "$config" --device cuda --resume "$stage_resume"
  if [[ ! -f "$checkpoint_dir/final.pt" ]]; then
    echo "[$label] training ended without final.pt" >&2
    exit 1
  fi
}

run_stage unsb "$ARTIFACTS_ROOT/weights/unsb" \
  "$CODE_ROOT/scripts/train_unsb.py" "$CODE_ROOT/configs/train_unsb.yaml"
run_stage paired "$ARTIFACTS_ROOT/weights/paired" \
  "$CODE_ROOT/scripts/train_paired.py" "$CODE_ROOT/configs/train_paired.yaml"
