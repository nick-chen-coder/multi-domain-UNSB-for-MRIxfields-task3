#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

from mrixfields_unsb.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a successful online validation submission record")
    parser.add_argument("--config", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--submission-name", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    record = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "submission_id": args.submission_id.strip(),
        "submission_name": args.submission_name.strip(),
        "status": "SUCCESSFUL_ONLINE_VALIDATION",
    }
    if not record["submission_id"] or not record["submission_name"]:
        raise ValueError("submission ID and name must be non-empty")
    path = Path(cfg.paths.logs_root) / "validation_submission_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"appended validation submission record to {path}")


if __name__ == "__main__":
    main()
