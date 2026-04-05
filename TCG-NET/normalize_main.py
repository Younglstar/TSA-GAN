"""Normalization entrypoint.

Thin wrapper around `steps.normalize.normalize`.
"""

from __future__ import annotations

from steps.normalize import normalize
from utils.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    normalize()


if __name__ == "__main__":
    main()
