"""Preprocessing entrypoint.

This file is intentionally thin: the actual step logic lives in
`steps/preprocess.py` so it can be reused by `run.py` and imported in
notebooks/tests.
"""

from __future__ import annotations

from steps.preprocess import preprocess
from utils.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    preprocess()


if __name__ == "__main__":
    main()
