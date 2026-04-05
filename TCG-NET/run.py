"""One-command pipeline runner.

Usage:
  python run.py                # preprocess -> normalize -> embed -> encdec
  python run.py --steps preprocess normalize

This runner calls step functions directly (no subprocess) to keep configs,
logging, and errors consistent.
"""

from __future__ import annotations
import sys
print("当前运行脚本的 Python 解释器路径是:", sys.executable)
import argparse
import os
import shutil

from steps.embed import embed
from steps.encdec import train_and_encode
from steps.normalize import normalize
from steps.preprocess import preprocess
from utils.logging_utils import setup_logging
from steps.postprocess import postprocess_synthetic

ALL_STEPS = ["preprocess", "normalize", "embed", "encdec"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the TSA/TCG pipeline")
    p.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help=f"Steps to run in order (default: {' '.join(ALL_STEPS)})",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG/INFO/WARNING/ERROR). Also supports LOG_LEVEL env.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 如果 log_level 为 None，设置默认值为 'INFO'
    log_level = args.log_level or 'INFO'  # 默认使用 'INFO' 级别
    setup_logging(log_level)  # 传递 log_level 给 logging 设置

    for step in args.steps:
        if step == "preprocess":
            preprocess()
        elif step == "normalize":
            normalize()
        elif step == "embed":
            embed()
        elif step == "encdec":
            train_and_encode()
        #elif step == "postprocess":
            #postprocess_synthetic()


if __name__ == "__main__":

    main()
    # 训练、编码全部结束后清理
    #if os.path.exists("./cache_encdec"):
     #   shutil.rmtree("./cache_encdec")