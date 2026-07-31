"""PyTorch `Dataset` implementations over preprocessed HydroSense manifests.

`src.preprocessing.run` writes one row per segment to a manifest CSV with
columns: `segment_path, recording_id, label, class_name, split`. Everything
downstream (training, evaluation, XAI) reads through `ManifestDataset` so
there is a single source of truth for how a segment on disk becomes a
labelled tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"segment_path", "recording_id", "label", "class_name"}


class ManifestDataset(Dataset):
    """Loads `.npy` spectrogram segments referenced by a manifest CSV.

    Args:
        manifest_path: CSV with at least `segment_path` and `label` columns.
        transform: Callable applied to the loaded (freq, time) array.
        split: If given, filters the manifest to rows where `split == split`.
        base_dir: Root directory `segment_path` entries are relative to.
            Defaults to the manifest's own parent directory.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform=None,
        split: str | None = None,
        base_dir: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        df = pd.read_csv(self.manifest_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Manifest {self.manifest_path} is missing columns: {missing}")

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.df = df
        self.transform = transform
        # `segment_path` entries written by `src.preprocessing.run` are already
        # absolute-or-cwd-relative paths (not relative to the manifest file), so
        # only join with `base_dir` when the caller explicitly opts in.
        self.base_dir = Path(base_dir) if base_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, segment_path: str) -> Path:
        p = Path(segment_path)
        if p.is_absolute() or self.base_dir is None:
            return p
        return self.base_dir / p

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        array = np.load(self._resolve(row["segment_path"])).astype(np.float32)

        if self.transform is not None:
            array = self.transform(array)

        label = int(row["label"])
        return array, label

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def recording_ids(self) -> np.ndarray:
        return self.df["recording_id"].to_numpy()

    def class_weights(self, num_classes: int) -> np.ndarray:
        """Inverse-frequency class weights for `CrossEntropyLoss(weight=...)`.

        Used whenever `training.class_weighted_loss` is set, to counter the
        ShipsEar category imbalance described in README §6.
        """
        counts = np.bincount(self.labels, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        return weights.astype(np.float32)
