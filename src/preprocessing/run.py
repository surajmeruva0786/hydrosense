#!/usr/bin/env python
"""End-to-end preprocessing CLI: raw .wav -> conditioned -> segmented -> spectrogram manifest.

Usage (README §14):
    python -m src.preprocessing.run \\
        --input_dir data/raw/shipsear \\
        --output_dir data/processed \\
        --sr 16000 \\
        --segment_length 10.0 \\
        --overlap 0.5

Reads `<input_dir>/metadata.csv` (recording_id, file_path, class_name,
duration_s — see `scripts/verify_dataset.py` / `scripts/generate_synthetic_dataset.py`),
writes one `.npy` array per segment plus `<output_dir>/<representation>/manifest.csv`
consumed by `src.data.dataset.ManifestDataset`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.representations import compute_representation  # noqa: E402
from src.preprocessing.segmentation import segment_and_filter_silence  # noqa: E402
from src.preprocessing.signal_conditioning import condition_signal  # noqa: E402
from src.preprocessing.splits import make_recording_splits, save_splits  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("preprocessing.run")

DEFAULT_CLASS_NAMES = ["A", "B", "C", "D", "E"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_length", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--representation",
        type=str,
        default="mel",
        choices=["mel", "log_mel", "cqt", "mfcc", "waveform"],
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument(
        "--class_names",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated ordered class names; index = training label.",
    )
    return parser.parse_args()


def process_recording(
    file_path: Path,
    class_idx: int,
    class_name: str,
    recording_id: str,
    target_sr: int,
    segment_length: float,
    overlap: float,
    representation: str,
    out_dir: Path,
) -> list[dict]:
    import soundfile as sf

    waveform, orig_sr = sf.read(str(file_path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform)
    conditioned, sr = condition_signal(waveform, orig_sr, target_sr)

    keep_silence = class_name == "E"  # ambient/no-vessel: silence *is* the signal (README §6)
    segments = segment_and_filter_silence(
        conditioned,
        sr,
        segment_length_s=segment_length,
        overlap=overlap,
        keep_silence_for_class=keep_silence,
    )

    rows = []
    class_dir = out_dir / representation / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        rep = compute_representation(seg.waveform, sr, representation)
        segment_path = class_dir / f"{recording_id}_{i:04d}.npy"
        np.save(segment_path, rep)
        rows.append(
            {
                "segment_path": str(segment_path),
                "recording_id": recording_id,
                "label": class_idx,
                "class_name": class_name,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    class_names = [c.strip() for c in args.class_names.split(",")]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_path = input_dir / "metadata.csv"

    metadata = pd.read_csv(metadata_path)
    logger.info("Loaded %d recordings from %s", len(metadata), metadata_path)

    split_df = make_recording_splits(
        metadata, n_folds=args.n_folds, test_fraction=args.test_fraction, seed=args.seed
    )
    split_paths = save_splits(split_df, args.splits_dir)
    logger.info("Wrote split files: %s", split_paths)

    split_lookup = split_df.set_index("recording_id")[["split", "fold"]].to_dict(orient="index")

    all_rows: list[dict] = []
    for _, row in metadata.iterrows():
        recording_id = row["recording_id"]
        class_name = row["class_name"]
        if class_name not in class_to_idx:
            logger.warning("Skipping recording %s: unknown class '%s'", recording_id, class_name)
            continue

        file_path = Path(row["file_path"])
        if not file_path.is_absolute() and not file_path.exists():
            file_path = input_dir / file_path.name

        logger.info("Processing %s (%s)", recording_id, class_name)
        seg_rows = process_recording(
            file_path=file_path,
            class_idx=class_to_idx[class_name],
            class_name=class_name,
            recording_id=recording_id,
            target_sr=args.sr,
            segment_length=args.segment_length,
            overlap=args.overlap,
            representation=args.representation,
            out_dir=output_dir,
        )

        split_info = split_lookup.get(recording_id, {"split": "train", "fold": -1})
        for r in seg_rows:
            r.update(split_info)
        all_rows.extend(seg_rows)

    manifest_dir = output_dir / args.representation
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False)

    logger.info("Wrote %d segments to manifest: %s", len(all_rows), manifest_path)


if __name__ == "__main__":
    main()
