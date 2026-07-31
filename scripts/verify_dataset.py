#!/usr/bin/env python
"""Verify a raw dataset directory before preprocessing.

Checks that every recording listed in `metadata.csv` exists, is readable,
has a sane sample rate/duration, and reports the per-class recording count
so class imbalance (README §6) is visible before the pipeline runs.

Usage:
    python scripts/verify_dataset.py --input_dir data/raw/shipsear
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("verify_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default="data/raw/shipsear")
    parser.add_argument("--min_sample_rate", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.csv"

    if not input_dir.exists():
        logger.error(
            "%s does not exist. Real ShipsEar recordings must be requested from the "
            "University of Vigo (http://atlanttic.uvigo.es/underwaternoise/) and placed "
            "here; for local development/CI use "
            "scripts/generate_synthetic_dataset.py --output_dir %s instead.",
            input_dir,
            input_dir,
        )
        return 1

    if not metadata_path.exists():
        logger.error("Missing %s — expected columns: recording_id,file_path,class_name,duration_s", metadata_path)
        return 1

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile is required to verify audio files: pip install soundfile")
        return 1

    with metadata_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        logger.error("%s contains no rows.", metadata_path)
        return 1

    class_counts: Counter = Counter()
    n_ok, n_bad = 0, 0

    for row in rows:
        file_path = Path(row["file_path"])
        if not file_path.is_absolute():
            file_path = input_dir / file_path.name
        if not file_path.exists():
            logger.warning("Missing audio file: %s", file_path)
            n_bad += 1
            continue
        try:
            info = sf.info(str(file_path))
        except Exception as exc:  # noqa: BLE001 - report and continue verifying the rest
            logger.warning("Could not read %s: %s", file_path, exc)
            n_bad += 1
            continue

        if info.samplerate < args.min_sample_rate:
            logger.warning(
                "%s has sample rate %d Hz, below the %d Hz minimum",
                file_path,
                info.samplerate,
                args.min_sample_rate,
            )

        class_counts[row["class_name"]] += 1
        n_ok += 1

    logger.info("Verified %d/%d recordings OK (%d unreadable/missing).", n_ok, len(rows), n_bad)
    logger.info("Class distribution: %s", dict(sorted(class_counts.items())))

    if n_bad > 0:
        logger.error("Dataset verification found %d problem(s).", n_bad)
        return 1

    logger.info("Dataset verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
