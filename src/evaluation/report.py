"""Render evaluation metrics as the README §16-style tables/artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def confusion_matrix_to_markdown(confusion_matrix: list[list[int]], class_names: list[str], normalize: bool = True) -> str:
    """Render a confusion matrix as a Markdown table, normalised row-wise by default
    (matches the `True_X` / `Pred_X` table format in README §16)."""
    cm = np.array(confusion_matrix, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    header = "| True \\ Pred | " + " | ".join(f"Pred_{c}" for c in class_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(class_names)) + "|"
    rows = []
    for i, name in enumerate(class_names):
        cells = " | ".join(f"{cm[i, j]:.2f}" for j in range(len(class_names)))
        rows.append(f"| True_{name} | {cells} |")

    return "\n".join([header, sep, *rows])


def metrics_to_markdown_table(rows: list[dict]) -> str:
    """Render a list of {"model", "representation", "macro_f1", "accuracy", "top2_accuracy"}
    dicts as the README §16 results table."""
    header = "| Model | Representation | Macro-F1 | Accuracy | Top-2 Acc. |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['representation']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | {r['top2_accuracy']:.3f} |"
        )
    return "\n".join(lines)


def write_report(metrics: dict, output_dir: str | Path) -> dict[str, Path]:
    """Write `metrics.json` plus a human-readable `report.md` (confusion matrix + per-class table)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    class_names = metrics["class_names"]
    lines = [
        "# HydroSense Evaluation Report",
        "",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Macro-F1: **{metrics['macro_f1']:.4f}**",
        f"- Weighted-F1: **{metrics['weighted_f1']:.4f}**",
        f"- Top-2 Accuracy: **{metrics['top2_accuracy']:.4f}**",
        f"- Macro ROC-AUC: **{metrics['macro_roc_auc']:.4f}**",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name in class_names:
        pc = metrics["per_class"][name]
        lines.append(f"| {name} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |")

    lines += [
        "",
        "## Confusion Matrix (row-normalised)",
        "",
        confusion_matrix_to_markdown(metrics["confusion_matrix"], class_names),
        "",
    ]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {"metrics": metrics_path, "report": report_path}
