"""Recording-level train/val/test splitting (README §5.6, §10 validation strategy).

Splitting happens on **recordings**, never on segments — otherwise two
overlapping 10 s windows from the same recording could land in both train
and test, leaking near-identical signal across the split (README §5.6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_TEST_FRACTION = 0.20
DEFAULT_N_FOLDS = 5


def make_recording_splits(
    metadata: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every recording to the held-out test set or a CV fold.

    Args:
        metadata: DataFrame with at least `recording_id` and `class_name`.
        n_folds: Number of stratified CV folds over the non-test recordings.
        test_fraction: Fraction of recordings reserved as the held-out test
            set (README §10: "test set of 18 recordings (20%)" for the
            90-recording ShipsEar corpus — scales with corpus size here).
        seed: RNG seed for the deterministic split (README §17).

    Returns:
        `metadata` with two extra columns: `split` ("train" | "test") and
        `fold` (0..n_folds-1 for train rows, -1 for test rows).
    """
    df = metadata.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)

    df["split"] = "train"
    df["fold"] = -1

    test_indices = []
    for class_name, group in df.groupby("class_name"):
        n_test = max(1, round(len(group) * test_fraction)) if len(group) > 1 else 0
        chosen = rng.choice(group.index.to_numpy(), size=min(n_test, len(group) - 1), replace=False)
        test_indices.extend(chosen.tolist())

    df.loc[test_indices, "split"] = "test"

    train_df = df[df["split"] == "train"]
    class_counts = train_df["class_name"].value_counts()
    usable_folds = min(n_folds, int(class_counts.min())) if len(class_counts) else n_folds

    if usable_folds < 2:
        # Too few recordings per class for stratified CV (only expected on tiny
        # dev/test fixtures) -- everything becomes a single fold-0 validation set.
        df.loc[train_df.index, "fold"] = 0
        return df

    skf = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    train_idx_array = train_df.index.to_numpy()
    train_labels = train_df["class_name"].to_numpy()

    for fold_id, (_, val_idx) in enumerate(skf.split(train_idx_array, train_labels)):
        df.loc[train_idx_array[val_idx], "fold"] = fold_id

    return df


def save_splits(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write `test.csv` and `folds.csv` under `output_dir` (default `data/splits/`)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = output_dir / "test.csv"
    folds_path = output_dir / "folds.csv"

    df[df["split"] == "test"].to_csv(test_path, index=False)
    df[df["split"] == "train"].to_csv(folds_path, index=False)

    return {"test": test_path, "folds": folds_path}
