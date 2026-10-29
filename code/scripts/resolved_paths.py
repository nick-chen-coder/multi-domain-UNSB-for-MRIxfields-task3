#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mrixfields_unsb.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print resolved values from paths.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("keys", nargs="+")
    args = parser.parse_args()
    paths = load_config(args.config).paths
    for key in args.keys:
        if key not in paths:
            raise KeyError(key)
        print(paths[key])


if __name__ == "__main__":
    main()
