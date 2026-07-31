#!/usr/bin/env python
"""Pre-download and cache the YAMNet TF-Hub module used by HydroSense-TL (README §7.3).

Run once before offline training so `hub.KerasLayer(...)` in
`src.models.hydrosense_tl.build_hydrosense_tl` resolves from the local
cache instead of requiring network access at training time.

Usage:
    python scripts/download_yamnet.py --cache_dir .cache/tfhub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hydrosense_tl import YAMNET_HUB_URL  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("download_yamnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=str, default=".cache/tfhub")
    parser.add_argument("--hub_url", type=str, default=YAMNET_HUB_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = args.cache_dir

    import tensorflow_hub as hub

    logger.info("Downloading %s into cache %s ...", args.hub_url, args.cache_dir)
    hub.resolve(args.hub_url)
    logger.info("YAMNet cached. Set TFHUB_CACHE_DIR=%s before training/inference.", args.cache_dir)


if __name__ == "__main__":
    main()
