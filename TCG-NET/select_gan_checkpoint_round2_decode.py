#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
select_gan_checkpoint_round2_decode.py

第二轮 GAN checkpoint 精筛：
1. 读取候选 GAN checkpoint
2. 逐个生成 synthetic encoded
3. 调用你现有的 decode 脚本
4. 调用你现有的 synthetic_to_csv 脚本
5. 只比较关键列 fidelity，输出最终 round2 排名

说明：
- 这版脚本不强行绑定你工程里 decode / synthetic_to_csv 的具体 CLI。
- 你只需要通过 --decode_cmd_template 和 --tocsv_cmd_template 提供你当前实际命令模板即可。
- 适合作为“第一轮 encoded 排名之后”的第二轮精筛工具。
"""

import argparse
import json
import pickle
import random
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from gan_config import GANConfig
from gan_data import load_encoded_data, scale_data, inverse_scale_data
from gan_models import Generator, MappingNetwork


# =========================================================
# 基础工具
# =========================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def subsample_rows(x: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    if len(x) <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


# =========================================================
# GAN checkpoint 加载 / 采样
# =========================================================

def build_models_from_config(cfg: GANConfig, input_dim: int, device: str):
    generator = Generator(
        w_dim=cfg.W_DIM,
        output_dim=input_dim,
        hidden_dims=cfg.GENERATOR_HIDDEN_DIMS,
    ).to(device)

    mapping_network = MappingNetwork(
        z_dim=cfg.NOISE_DIM,
        w_dim=cfg.W_DIM,
        hidden_layers=cfg.MAPPING_HIDDEN_LAYERS,
        hidden_dim=cfg.MAPPING_HIDDEN_DIM,
    ).to(device)

    generator.eval()
    mapping_network.eval()
    return generator, mapping_network


def load_generator_mapping_from_checkpoint(
    ckpt_path: str,
    cfg: GANConfig,
    input_dim: int,
    device: str,
):
    payload = torch.load(ckpt_path, map_location=device)
    generator, mapping_network = build_models_from_config(cfg, input_dim, device)

    if "generator_state_dict" not in payload:
        raise KeyError(f"{ckpt_path} 缺少 generator_state_dict")
    if "mapping_network_state_dict" not in payload:
        raise KeyError(f"{ckpt_path} 缺少 mapping_network_state_dict")

    generator.load_state_dict(payload["generator_state_dict"])
    mapping_network.load_state_dict(payload["mapping_network_state_dict"])
    return payload, generator, mapping_network


@torch.no_grad()
def generate_synthetic_scaled(generator, mapping_network, cfg: GANConfig, num_samples: int, device: str) -> np.ndarray:
    out = []
    generated = 0
    while generated < num_samples:
        bs = min(cfg.BATCH_SIZE, num_samples - generated)
        z = torch.randn(bs, cfg.NOISE_DIM, device=device)
        w = mapping_network(z)
        fake = generator(w)
        out.append(fake.cpu().numpy())
        generated += bs
    return np.concatenate(out, axis=0)


# =========================================================
# fidelity
# =========================================================

def summarize_col(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce")
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "q90": float(np.nanquantile(x, 0.90)),
        "q99": float(np.nanquantile(x, 0.99)),
        "zero_rate": float(np.mean(x == 0)),
        "positive_rate": float(np.mean(x > 0)),
    }


def rel_err(a: float, b: float, eps: float = 1e-8) -> float:
    return abs(a - b) / (abs(a) + eps)


def compute_key_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame, cols: List[str]) -> Dict:
    rows = []
    col_scores = []

    for col in cols:
        if col not in real_df.columns or col not in synth_df.columns:
            rows.append({"col": col, "status": "missing"})
            continue

        xr = pd.to_numeric(real_df[col], errors="coerce").dropna().to_numpy()
        xs = pd.to_numeric(synth_df[col], errors="coerce").dropna().to_numpy()

        if len(xr) == 0 or len(xs) == 0:
            rows.append({"col": col, "status": "empty"})
            continue

        ks = ks_2samp(xr, xs).statistic
        sr = summarize_col(real_df[col])
        ss = summarize_col(synth_df[col])

        mean_err = rel_err(sr["mean"], ss["mean"])
        std_err = rel_err(sr["std"], ss["std"])
        q99_err = rel_err(sr["q99"], ss["q99"])
        zero_err = abs(sr["zero_rate"] - ss["zero_rate"])

        score = 1.0 - (
            0.35 * ks +
            0.20 * min(mean_err, 1.0) +
            0.15 * min(std_err, 1.0) +
            0.20 * min(q99_err, 1.0) +
            0.10 * min(zero_err, 1.0)
        )
        score = float(max(0.0, score))
        col_scores.append(score)

        rows.append({
            "col": col,
            "status": "ok",
            "ks": float(ks),
            "real_mean": sr["mean"],
            "synth_mean": ss["mean"],
            "real_std": sr["std"],
            "synth_std": ss["std"],
            "real_q99": sr["q99"],
            "synth_q99": ss["q99"],
            "real_zero_rate": sr["zero_rate"],
            "synth_zero_rate": ss["zero_rate"],
            "mean_rel_err": float(mean_err),
            "std_rel_err": float(std_err),
            "q99_rel_err": float(q99_err),
            "zero_rate_gap": float(zero_err),
            "col_score": score,
        })

    fidelity_score = float(np.mean(col_scores)) if len(col_scores) > 0 else 0.0
    return {
        "per_col": pd.DataFrame(rows),
        "fidelity_score": fidelity_score,
    }


# =========================================================
# subprocess
# =========================================================

def run_command_from_template(template: str, variables: Dict[str, str], cwd: str = None):
    cmd = template.format(**variables)
    print(f"[CMD] {cmd}")
    result = subprocess.run(
        shlex.split(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result


# =========================================================
# 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints", nargs="+", required=True, help="候选 GAN checkpoint 列表")
    parser.add_argument("--real_csv_path", type=str, required=True, help="真实源表 CSV 路径")
    parser.add_argument("--num_synth", type=int, default=200000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="./gan_round2_decode")

    parser.add_argument(
        "--key_cols",
        nargs="+",
        default=["leg1v", "leg2v", "air1", "grid", "solar"],
        help="第二轮关键列 fidelity 比较列表"
    )

    parser.add_argument(
        "--decode_cmd_template",
        type=str,
        required=True,
        help=(
            "decode 命令模板。可用占位符："
            "{synthetic_pkl}, {decode_out_dir}"
        ),
    )

    parser.add_argument(
        "--tocsv_cmd_template",
        type=str,
        required=True,
        help=(
            "synthetic_to_csv 命令模板。可用占位符："
            "{decoded_data_pkl}, {mask_logits_pkl}, {mask_probs_pkl}, {mask_binary_pkl}, {csv_out_dir}"
        ),
    )

    parser.add_argument(
        "--final_csv_template",
        type=str,
        default="{csv_out_dir}/synthetic_merged_long.csv",
        help="最终 long CSV 路径模板。默认 {csv_out_dir}/synthetic_merged_long.csv",
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="执行 decode / tocsv 子命令时的工作目录",
    )

    parser.add_argument(
        "--reuse_if_exists",
        action="store_true",
        help="如果候选目录下 final csv 已存在，则跳过生成流程直接比较",
    )

    args = parser.parse_args()

    cfg = GANConfig()
    set_seed(cfg.SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading encoded data from: {cfg.ENCODED_DATA_PATH}")
    encoded_data = load_encoded_data(cfg.ENCODED_DATA_PATH)
    if not isinstance(encoded_data, np.ndarray):
        encoded_data = np.asarray(encoded_data, dtype=np.float32)
    encoded_data = encoded_data.astype(np.float32, copy=False)

    train_data, _ = train_test_split(
        encoded_data,
        test_size=cfg.VALID_SPLIT,
        random_state=cfg.SEED,
        shuffle=True,
    )

    scaled_train_data, scaling_params = scale_data(train_data)
    input_dim = scaled_train_data.shape[1]

    real_df = pd.read_csv(args.real_csv_path)
    rows = []

    for idx, ckpt_path in enumerate(args.checkpoints, start=1):
        ckpt_path = str(ckpt_path)
        stem = Path(ckpt_path).stem

        candidate_dir = out_dir / stem
        candidate_dir.mkdir(parents=True, exist_ok=True)

        synthetic_pkl = candidate_dir / "synthetic_final_data.pkl"
        decode_out_dir = candidate_dir / "decode_outputs"
        csv_out_dir = candidate_dir / "csv_outputs"

        decoded_data_pkl = decode_out_dir / "decoded_data.pkl"
        mask_logits_pkl = decode_out_dir / "mask_logits.pkl"
        mask_probs_pkl = decode_out_dir / "mask_probs.pkl"
        mask_binary_pkl = decode_out_dir / "mask_binary.pkl"

        final_csv_path = Path(
            args.final_csv_template.format(
                csv_out_dir=str(csv_out_dir),
                decoded_data_pkl=str(decoded_data_pkl),
                mask_logits_pkl=str(mask_logits_pkl),
                mask_probs_pkl=str(mask_probs_pkl),
                mask_binary_pkl=str(mask_binary_pkl),
                synthetic_pkl=str(synthetic_pkl),
                decode_out_dir=str(decode_out_dir),
            )
        )

        print(f"\n===== [{idx}/{len(args.checkpoints)}] {ckpt_path} =====")

        try:
            if args.reuse_if_exists and final_csv_path.exists():
                print(f"[INFO] reuse existing final csv -> {final_csv_path}")
            else:
                _, generator, mapping_network = load_generator_mapping_from_checkpoint(
                    ckpt_path=ckpt_path,
                    cfg=cfg,
                    input_dim=input_dim,
                    device=args.device,
                )

                synthetic_scaled = generate_synthetic_scaled(
                    generator=generator,
                    mapping_network=mapping_network,
                    cfg=cfg,
                    num_samples=args.num_synth,
                    device=args.device,
                )
                synthetic_data = inverse_scale_data(synthetic_scaled, scaling_params)
                save_pickle(synthetic_data, str(synthetic_pkl))
                print(f"[INFO] synthetic encoded saved -> {synthetic_pkl}")

                decode_out_dir.mkdir(parents=True, exist_ok=True)
                csv_out_dir.mkdir(parents=True, exist_ok=True)

                variables = {
                    "synthetic_pkl": str(synthetic_pkl),
                    "decode_out_dir": str(decode_out_dir),
                    "decoded_data_pkl": str(decoded_data_pkl),
                    "mask_logits_pkl": str(mask_logits_pkl),
                    "mask_probs_pkl": str(mask_probs_pkl),
                    "mask_binary_pkl": str(mask_binary_pkl),
                    "csv_out_dir": str(csv_out_dir),
                }

                decode_ret = run_command_from_template(
                    template=args.decode_cmd_template,
                    variables=variables,
                    cwd=args.project_root,
                )
                (candidate_dir / "decode_stdout.txt").write_text(decode_ret.stdout, encoding="utf-8")
                (candidate_dir / "decode_stderr.txt").write_text(decode_ret.stderr, encoding="utf-8")
                if decode_ret.returncode != 0:
                    raise RuntimeError(f"decode failed, returncode={decode_ret.returncode}")

                tocsv_ret = run_command_from_template(
                    template=args.tocsv_cmd_template,
                    variables=variables,
                    cwd=args.project_root,
                )
                (candidate_dir / "tocsv_stdout.txt").write_text(tocsv_ret.stdout, encoding="utf-8")
                (candidate_dir / "tocsv_stderr.txt").write_text(tocsv_ret.stderr, encoding="utf-8")
                if tocsv_ret.returncode != 0:
                    raise RuntimeError(f"synthetic_to_csv failed, returncode={tocsv_ret.returncode}")

            if not final_csv_path.exists():
                raise FileNotFoundError(f"final csv not found: {final_csv_path}")

            synth_df = pd.read_csv(final_csv_path)
            fid = compute_key_fidelity(real_df=real_df, synth_df=synth_df, cols=args.key_cols)

            fid["per_col"].to_csv(candidate_dir / "key_fidelity.csv", index=False)

            row = {
                "checkpoint_path": ckpt_path,
                "final_csv_path": str(final_csv_path),
                "key_fidelity_score": fid["fidelity_score"],
            }

            for _, r in fid["per_col"].iterrows():
                if r.get("status") == "ok":
                    row[f"{r['col']}_score"] = float(r["col_score"])
                    row[f"{r['col']}_ks"] = float(r["ks"])
                    row[f"{r['col']}_mean_rel_err"] = float(r["mean_rel_err"])
                    row[f"{r['col']}_std_rel_err"] = float(r["std_rel_err"])
                    row[f"{r['col']}_q99_rel_err"] = float(r["q99_rel_err"])
                    row[f"{r['col']}_zero_rate_gap"] = float(r["zero_rate_gap"])

            rows.append(row)

            with open(candidate_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

            print(f"[INFO] key_fidelity_score = {fid['fidelity_score']:.6f}")

        except Exception as e:
            print(f"[ERROR] {ckpt_path} failed: {e}")
            rows.append({
                "checkpoint_path": ckpt_path,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "round2_raw.csv", index=False)

    if "key_fidelity_score" not in df.columns:
        print("[WARN] no successful candidates")
        return

    ok_df = df[df["key_fidelity_score"].notna()].copy()
    if len(ok_df) == 0:
        print("[WARN] no successful candidates")
        return

    ok_df = ok_df.sort_values("key_fidelity_score", ascending=False).reset_index(drop=True)
    ok_df.to_csv(out_dir / "round2_ranking.csv", index=False)

    best_row = ok_df.iloc[0].to_dict()
    with open(out_dir / "round2_best.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    print("\n===== ROUND2 FINAL RANKING =====")
    show_cols = ["checkpoint_path", "key_fidelity_score"] + [f"{c}_score" for c in args.key_cols if f"{c}_score" in ok_df.columns]
    show_cols = [c for c in show_cols if c in ok_df.columns]
    print(ok_df[show_cols].to_string(index=False))

    print("\n[ROUND2 SELECTED BEST CHECKPOINT]")
    print(best_row["checkpoint_path"])
    print(f"key_fidelity_score = {best_row['key_fidelity_score']:.6f}")


if __name__ == "__main__":
    main()
