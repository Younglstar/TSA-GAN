from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any


def ensure_exists(path: str | os.PathLike, hint: str = "") -> Path:
    """Raise a clear error if the file does not exist."""
    p = Path(path)
    if not p.exists():
        msg = f"File not found: {p}"
        if hint:
            msg += f". {hint}"
        raise FileNotFoundError(msg)
    return p


def load_pickle(path: str | os.PathLike) -> Any:
    """Load a pickle file."""
    p = ensure_exists(path)
    with p.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | os.PathLike) -> Path:
    """Save an object to a pickle file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)
    return p
