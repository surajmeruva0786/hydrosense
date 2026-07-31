"""Consistent console + file logging setup for CLI entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str, log_file: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Return a configured logger that writes to stdout, and optionally to a file.

    Safe to call repeatedly with the same `name` (e.g. across CV folds) —
    handlers are only attached once per logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger
