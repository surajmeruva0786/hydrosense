"""Shared pytest fixtures: small synthetic audio/manifest fixtures so the
test suite never depends on the (access-gated) real ShipsEar corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def synthetic_waveform(sample_rate):
    from src.data.synthetic import generate_synthetic_recording

    return generate_synthetic_recording("A", duration_s=6.0, sample_rate=sample_rate, seed=1)


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    import pandas as pd

    from src.data.synthetic import generate_synthetic_dataset

    out_dir = tmp_path / "shipsear_synthetic"
    records = generate_synthetic_dataset(
        out_dir,
        recordings_per_class={"A": 2, "B": 2, "C": 2, "D": 2, "E": 2},
        duration_range_s=(6.0, 8.0),
        sample_rate=16000,
        seed=7,
    )
    pd.DataFrame(records).to_csv(out_dir / "metadata.csv", index=False)
    return out_dir


@pytest.fixture
def processed_manifest(tmp_path, synthetic_dataset_dir):
    """Runs the real preprocessing CLI logic (in-process) to produce a small mel manifest."""
    import pandas as pd

    from src.preprocessing.representations import compute_representation
    from src.preprocessing.segmentation import segment_and_filter_silence
    from src.preprocessing.signal_conditioning import condition_signal
    from src.preprocessing.splits import make_recording_splits, save_splits

    metadata = pd.read_csv(synthetic_dataset_dir / "metadata.csv")
    splits_dir = tmp_path / "splits"
    split_df = make_recording_splits(metadata, n_folds=2, test_fraction=0.2, seed=7)
    save_splits(split_df, splits_dir)
    split_lookup = split_df.set_index("recording_id")[["split", "fold"]].to_dict(orient="index")

    class_names = ["A", "B", "C", "D", "E"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    output_dir = tmp_path / "processed"
    rows = []
    import soundfile as sf

    for _, row in metadata.iterrows():
        waveform, orig_sr = sf.read(row["file_path"], dtype="float32", always_2d=False)
        conditioned, sr = condition_signal(np.asarray(waveform), orig_sr, 16000)
        segments = segment_and_filter_silence(conditioned, sr, segment_length_s=2.0, overlap=0.5)

        class_dir = output_dir / "mel" / row["class_name"]
        class_dir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segments):
            rep = compute_representation(seg.waveform, sr, "mel")
            seg_path = class_dir / f"{row['recording_id']}_{i:03d}.npy"
            np.save(seg_path, rep)
            info = split_lookup[row["recording_id"]]
            rows.append(
                {
                    "segment_path": str(seg_path),
                    "recording_id": row["recording_id"],
                    "label": class_to_idx[row["class_name"]],
                    "class_name": row["class_name"],
                    "split": info["split"],
                    "fold": info["fold"],
                }
            )

    manifest_path = output_dir / "mel" / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
