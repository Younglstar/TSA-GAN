"""Categorical embeddings entrypoint.

Thin wrapper around `steps.embed.embed`.
"""

from __future__ import annotations

from steps.embed import embed
from utils.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    embed()


if __name__ == "__main__":
    main()
