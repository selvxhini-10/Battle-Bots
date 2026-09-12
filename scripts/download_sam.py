#!/usr/bin/env python3
"""Download the official SAM ViT-B checkpoint without adding it to Git."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/sam_vit_b_01ec64.pth"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    print(f"Downloading official SAM ViT-B checkpoint to {args.output}...")
    try:
        with urllib.request.urlopen(URL) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.replace(args.output)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
