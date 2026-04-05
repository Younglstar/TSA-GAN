from __future__ import annotations

import logging
from pathlib import Path
import os
from typing import Optional


def setup_logging(level="INFO", log_file="logs/run.log"):
    Path("logs").mkdir(exist_ok=True)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger after `setup_logging` has been called."""
    return logging.getLogger(name)
