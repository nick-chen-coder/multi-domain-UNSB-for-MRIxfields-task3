from __future__ import annotations

import argparse


def training_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="none", help="none, auto, or an explicit checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="temporary OmegaConf dotlist override; repeat for multiple values",
    )
    return parser
