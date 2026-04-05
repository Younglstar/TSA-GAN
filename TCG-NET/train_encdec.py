"""CausalVAE entrypoint.

The heavy lifting is implemented in `steps/encdec.py`.
"""

from __future__ import annotations

from steps.encdec import train_and_encode
from utils.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    train_and_encode()


if __name__ == "__main__":
    main()
