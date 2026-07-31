#!/usr/bin/env python
"""HydroSense training CLI — 5-fold recording-level CV (README §5.6, §8, §14).

Usage:
    python -m src.training.train \\
        --model hydrosense_se \\
        --representation mel \\
        --folds 5 \\
        --epochs 100 \\
        --batch_size 32 \\
        --use_synthetic \\
        --output_dir runs/hydrosense_se_mel

Dispatches to `_train_torch_cv` for HydroSense-Base/SE (PyTorch) or
`_train_tl` for HydroSense-TL (Keras/YAMNet, README §7.3) based on
`config["model"]`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ManifestDataset  # noqa: E402
from src.data.transforms import build_default_transform  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.preprocessing.representations import expected_num_frames  # noqa: E402
from src.training.augment import AugmentConfig, apply_training_augmentation  # noqa: E402
from src.training.checkpoint import EarlyStopping, save_checkpoint  # noqa: E402
from src.training.losses import build_loss, mixup_loss  # noqa: E402
from src.training.scheduler import build_warmup_cosine_scheduler  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.metrics import aggregate_fold_metrics, compute_metrics  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("training.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="hydrosense_base",
                         choices=["hydrosense_base", "hydrosense_se", "hydrosense_tl"])
    parser.add_argument("--representation", type=str, default="mel")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--synthetic_dir", type=str, default="data/synthetic")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Override config YAML path (defaults to configs/<model>.yaml)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_folds_to_run", type=int, default=None, help="Debug: cap the number of folds actually trained.")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict:
    config_path = args.config or f"configs/{args.model}.yaml"
    if Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = load_config("configs/hydrosense_base.yaml")
        config["model"] = args.model
        logger.warning("Config %s not found; using defaults with model=%s", config_path, args.model)

    config["representation"] = args.representation
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["use_synthetic"] = args.use_synthetic
    config["folds"] = args.folds
    return config


def _build_datasets(config: dict, args: argparse.Namespace, fold: int):
    manifest_path = Path(args.processed_dir) / args.representation / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `python -m src.preprocessing.run` first (README §14)."
        )

    num_frames = expected_num_frames(int(config["sample_rate"] * config["segment_length"]))
    transform = build_default_transform(target_frames=num_frames)

    full_train = ManifestDataset(manifest_path, transform=transform, split="train")
    fold_ids = full_train.df["fold"].to_numpy()

    train_indices = np.where(fold_ids != fold)[0].tolist()
    val_indices = np.where(fold_ids == fold)[0].tolist()

    train_ds = torch.utils.data.Subset(full_train, train_indices)
    val_ds = torch.utils.data.Subset(full_train, val_indices)

    if config["training"].get("use_synthetic"):
        synth_manifest = Path(args.synthetic_dir) / args.representation / "manifest.csv"
        if synth_manifest.exists():
            synth_ds = ManifestDataset(synth_manifest, transform=transform)
            train_ds = ConcatDataset([train_ds, synth_ds])
            logger.info("Fold %d: added %d synthetic (TimeGAN) training segments", fold, len(synth_ds))
        else:
            logger.warning("use_synthetic=True but %s not found; run src.augmentation.train_timegan first", synth_manifest)

    class_weights = full_train.class_weights(config["num_classes"]) if config["training"].get("class_weighted_loss", True) else None
    return train_ds, val_ds, class_weights


def _run_epoch(model, loader, criterion, optimizer, scheduler, augment_cfg, device, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        if train:
            x, y_a, y_b, lam = apply_training_augmentation(x, y, augment_cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                logits = model(x)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(y.cpu().numpy())

    if scheduler is not None and train:
        scheduler.step()

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / max(1, len(y_true))
    return avg_loss, y_true, y_pred, y_prob


def _train_torch_cv(config: dict, args: argparse.Namespace) -> dict:
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    augment_cfg = AugmentConfig.from_dict(config["augmentation"])
    n_folds = args.max_folds_to_run or config["folds"]
    fold_metrics = []
    best_macro_f1 = -1.0
    best_ckpt_path = None

    for fold in range(n_folds):
        set_seed(config["seed"] + fold)
        logger.info("=== Fold %d/%d ===", fold + 1, n_folds)

        train_ds, val_ds, class_weights = _build_datasets(config, args, fold)
        if len(val_ds) == 0:
            logger.warning("Fold %d has an empty validation split; skipping.", fold)
            continue

        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=args.num_workers)

        model = build_model(config).to(device)
        criterion = build_loss(class_weights, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer, config["training"]["epochs"], config["training"]["warmup_epochs"]
        )
        scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and config["training"].get("mixed_precision")) else None
        early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"], mode="max")

        fold_dir = output_dir / f"fold_{fold}"
        best_fold_ckpt = fold_dir / "best.ckpt"

        for epoch in range(config["training"]["epochs"]):
            t0 = time.time()
            train_loss, *_ = _run_epoch(model, train_loader, criterion, optimizer, scheduler, augment_cfg, device, scaler, train=True)
            val_loss, y_true, y_pred, y_prob = _run_epoch(model, val_loader, criterion, optimizer, None, augment_cfg, device, None, train=False)

            metrics = compute_metrics(y_true, y_pred, y_prob, config["class_names"])
            is_best = early_stopping.step(metrics["macro_f1"])
            if is_best:
                save_checkpoint(best_fold_ckpt, model, optimizer, epoch, metrics)

            logger.info(
                "fold=%d epoch=%d/%d train_loss=%.4f val_loss=%.4f macro_f1=%.4f acc=%.4f (%.1fs)%s",
                fold, epoch + 1, config["training"]["epochs"], train_loss, val_loss,
                metrics["macro_f1"], metrics["accuracy"], time.time() - t0,
                " *" if is_best else "",
            )

            if early_stopping.should_stop:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, early_stopping.patience)
                break

        fold_metrics.append({"fold": fold, "macro_f1": early_stopping.best_score or 0.0})
        if best_fold_ckpt.exists() and (early_stopping.best_score or -1) > best_macro_f1:
            best_macro_f1 = early_stopping.best_score or -1
            best_ckpt_path = best_fold_ckpt

    if best_ckpt_path is not None:
        import shutil

        shutil.copy(best_ckpt_path, output_dir / "best.ckpt")
        logger.info("Best checkpoint overall (macro_f1=%.4f): %s -> %s", best_macro_f1, best_ckpt_path, output_dir / "best.ckpt")

    summary = {"per_fold": fold_metrics, "best_macro_f1": best_macro_f1}
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _train_tl(config: dict, args: argparse.Namespace) -> dict:
    """Simplified single-split training loop for HydroSense-TL (Keras/YAMNet)."""
    import tensorflow as tf

    from src.models.hydrosense_tl import build_hydrosense_tl

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    manifest_path = Path(args.processed_dir) / "waveform" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run preprocessing with --representation waveform for HydroSense-TL."
        )

    import pandas as pd

    df = pd.read_csv(manifest_path)
    train_df = df[(df["split"] == "train") & (df["fold"] != 0)]
    val_df = df[(df["split"] == "train") & (df["fold"] == 0)]

    def _load_split(split_df):
        arrays = np.stack([np.load(p) for p in split_df["segment_path"]])
        labels = split_df["label"].to_numpy()
        return arrays.astype(np.float32), labels.astype(np.int64)

    x_train, y_train = _load_split(train_df)
    x_val, y_val = _load_split(val_df)

    model = build_hydrosense_tl(config)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config["training"]["lr"], weight_decay=config["training"]["weight_decay"]
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config["training"]["early_stopping_patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(str(output_dir / "best.weights.h5"), save_weights_only=True, save_best_only=True),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    y_prob = tf.nn.softmax(model.predict(x_val), axis=1).numpy()
    y_pred = y_prob.argmax(axis=1)
    metrics = compute_metrics(y_val, y_pred, y_prob, config["class_names"])

    summary = {"per_fold": [{"fold": 0, "macro_f1": metrics["macro_f1"]}], "best_macro_f1": metrics["macro_f1"]}
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    set_seed(config["seed"])

    logger.info("Training %s on representation=%s -> %s", config["model"], config["representation"], args.output_dir)

    if config["model"] == "hydrosense_tl":
        summary = _train_tl(config, args)
    else:
        summary = _train_torch_cv(config, args)

    logger.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
