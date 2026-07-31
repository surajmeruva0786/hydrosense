from __future__ import annotations

import numpy as np

from src.data.dataset import ManifestDataset
from src.data.transforms import build_default_transform
from src.preprocessing.representations import expected_num_frames


def test_manifest_dataset_lengths_and_split_filter(processed_manifest):
    full = ManifestDataset(processed_manifest)
    train_only = ManifestDataset(processed_manifest, split="train")
    test_only = ManifestDataset(processed_manifest, split="test")

    assert len(full) == len(train_only) + len(test_only)
    assert len(full) > 0
    assert set(train_only.df["split"]) <= {"train"}
    assert set(test_only.df["split"]) <= {"test"}


def test_manifest_dataset_getitem_shape_and_transform(processed_manifest):
    transform = build_default_transform(target_frames=expected_num_frames(int(2.0 * 16000)))
    ds = ManifestDataset(processed_manifest, transform=transform)

    array, label = ds[0]
    assert array.ndim == 3  # (channels, freq, time)
    assert array.shape[0] == 1
    assert isinstance(label, int)
    assert 0 <= array.min() and array.max() <= 1.0 + 1e-5


def test_class_weights_sum_reasonable(processed_manifest):
    ds = ManifestDataset(processed_manifest, split="train")
    weights = ds.class_weights(num_classes=5)
    assert weights.shape == (5,)
    assert np.all(weights > 0)
