#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from gan_config import GANConfig
from gan_data import load_encoded_data, scale_data, inverse_scale_data
from gan_models import Generator, MappingNetwork


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsample_rows(x: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    if len(x) <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = (x ** 2).sum(axis=1, keepdims=True)
    y2 = (y ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2 * x @ y.T, 0.0)


def median_heuristic_sigma(x: np.ndarray, max_points: int = 512) -> float:
    x = subsample_rows(x, max_points, seed=123)
    d2 = pairwise_sq_dists(x, x)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.sqrt(np.median(vals) + 1e-12))


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    d2 = pairwise_sq_dists(x, y)
    return np.exp(-d2 / (2.0 * sigma * sigma + 1e-12))


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray, max_samples: int = 2048) -> float:
    x = subsample_rows(x, max_samples, seed=101)
    y = subsample_rows(y, max_samples, seed=202)
    sigma = median_heuristic_sigma(np.concatenate([x, y], axis=0), max_points=512)

    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)

    n = len(x)
    m = len(y)

    term_x = (Kxx.sum() - np.trace(Kxx)) / max(n * (n - 1), 1)
    term_y = (Kyy.sum() - np.trace(Kyy)) / max(m * (m - 1), 1)
    term_xy = Kxy.mean()
    return float(term_x + term_y - 2.0 * term_xy)


def closeness_to_half_score(x: float) -> float:
    return max(0.0, 1.0 - 2.0 * abs(float(x) - 0.5))


def discover_checkpoints(cfg: GANConfig, checkpoint_dir: str, include_best_last: bool = True):
    paths = []

    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("*.pt")):
            paths.append(str(p))

    if include_best_last:
        for p in [cfg.MODEL_SAVE_PATH, cfg.LAST_MODEL_SAVE_PATH]:
            if p and os.path.exists(p):
                paths.append(str(Path(p)))

    seen = set()
    uniq = []
    for p in paths:
        ap = str(Path(p).resolve())
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


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


def load_generator_mapping_from_checkpoint(ckpt_path: str, cfg: GANConfig, input_dim: int, device: str):
    payload = torch.load(ckpt_path, map_location=device)
    generator, mapping_network = build_models_from_config(cfg, input_dim, device)

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


def evaluate_synthetic_vs_test(real_test: np.ndarray, synthetic: np.ndarray):
    real_test = np.asarray(real_test, dtype=np.float32)
    synthetic = np.asarray(synthetic, dtype=np.float32)

    n_eval = min(len(real_test), len(synthetic), 50000)
    real_eval = subsample_rows(real_test, n_eval, seed=11)
    synth_eval = subsample_rows(synthetic, n_eval, seed=22)

    mean_diff = np.abs(real_eval.mean(axis=0) - synth_eval.mean(axis=0))
    std_diff = np.abs(real_eval.std(axis=0) - synth_eval.std(axis=0))

    real_corr = np.corrcoef(real_eval, rowvar=False)
    synth_corr = np.corrcoef(synth_eval, rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_abs_diff = np.abs(real_corr - synth_corr)

    mmd = mmd_rbf_unbiased(real_eval, synth_eval, max_samples=2048)

    X = np.concatenate([real_eval, synth_eval], axis=0)
    y = np.concatenate([np.ones(len(real_eval)), np.zeros(len(synth_eval))], axis=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.float32)

    auc = float(roc_auc_score(y_val, y_prob))
    acc = float(accuracy_score(y_val, y_pred))

    return {
        "n_real_test": int(len(real_test)),
        "n_synthetic": int(len(synthetic)),
        "n_eval": int(n_eval),
        "mean_abs_diff_mean": float(mean_diff.mean()),
        "mean_abs_diff_max": float(mean_diff.max()),
        "std_abs_diff_mean": float(std_diff.mean()),
        "std_abs_diff_max": float(std_diff.max()),
        "corr_abs_diff_mean": float(corr_abs_diff.mean()),
        "corr_abs_diff_max": float(corr_abs_diff.max()),
        "mmd_rbf": float(mmd),
        "two_sample_auc": float(auc),
        "two_sample_acc": float(acc),
    }


def compute_selection_score(report):
    mmd = report["mmd_rbf"]
    mean_gap = report["mean_abs_diff_mean"]
    std_gap = report["std_abs_diff_mean"]
    corr_gap = report["corr_abs_diff_mean"]
    auc = report["two_sample_auc"]
    acc = report["two_sample_acc"]

    mmd_score = 1.0 / (1.0 + mmd)
    mean_score = 1.0 / (1.0 + mean_gap)
    std_score = 1.0 / (1.0 + std_gap)
    corr_score = 1.0 / (1.0 + corr_gap)
    auc_score = closeness_to_half_score(auc)
    acc_score = closeness_to_half_score(acc)

    total_score = (
        0.25 * mmd_score +
        0.15 * mean_score +
        0.15 * std_score +
        0.15 * corr_score +
        0.20 * auc_score +
        0.10 * acc_score
    )

    return {
        "score_mmd": float(mmd_score),
        "score_mean": float(mean_score),
        "score_std": float(std_score),
        "score_corr": float(corr_score),
        "score_auc_half": float(auc_score),
        "score_acc_half": float(acc_score),
        "selection_score": float(total_score),
    }


def save_pca_plot(real_test: np.ndarray, synthetic: np.ndarray, out_path: str):
    real_p = subsample_rows(real_test, 3000, seed=33)
    synth_p = subsample_rows(synthetic, 3000, seed=44)
    Xp = np.concatenate([real_p, synth_p], axis=0)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xp)

    n_real = len(real_p)
    Z_real = Z[:n_real]
    Z_synth = Z[n_real:]

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_real[:, 0], Z_real[:, 1], s=8, alpha=0.35, label="real_test")
    plt.scatter(Z_synth[:, 0], Z_synth[:, 1], s=8, alpha=0.35, label="synthetic")
    plt.title("PCA: real_test vs synthetic")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./cache_gan/checkpoints")
    parser.add_argument("--include_best_last", action="store_true")
    parser.add_argument("--num_synth", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="./gan_checkpoint_selection")
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

    train_data, test_data = train_test_split(
        encoded_data, test_size=cfg.VALID_SPLIT, random_state=cfg.SEED, shuffle=True
    )

    scaled_train_data, scaling_params = scale_data(train_data)
    input_dim = scaled_train_data.shape[1]

    ckpt_paths = discover_checkpoints(cfg, args.checkpoint_dir, include_best_last=args.include_best_last)
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"没有在 {args.checkpoint_dir} 找到任何 checkpoint")

    print(f"[INFO] discovered {len(ckpt_paths)} checkpoints")
    for p in ckpt_paths:
        print(f"  - {p}")

    rows = []

    for i, ckpt_path in enumerate(ckpt_paths, start=1):
        print(f"\n===== [{i}/{len(ckpt_paths)}] evaluating {ckpt_path} =====")
        try:
            payload, generator, mapping_network = load_generator_mapping_from_checkpoint(
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
            synthetic = inverse_scale_data(synthetic_scaled, scaling_params)

            report = evaluate_synthetic_vs_test(real_test=test_data, synthetic=synthetic)
            score_pack = compute_selection_score(report)

            stem = Path(ckpt_path).stem
            save_pca_plot(test_data, synthetic, str(out_dir / f"{stem}_pca.png"))

            row = {
                "checkpoint_path": ckpt_path,
                "payload_epoch": payload.get("epoch", None),
                "payload_best_metric_name": payload.get("best_metric_name", None),
                "payload_best_metric_value": payload.get("best_metric_value", None),
            }

            if "metrics" in payload and isinstance(payload["metrics"], dict):
                for k, v in payload["metrics"].items():
                    row[f"train_{k}"] = v

            row.update(report)
            row.update(score_pack)
            rows.append(row)

            with open(out_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)

            print(
                "[INFO] "
                f"score={row['selection_score']:.6f} | "
                f"mmd={row['mmd_rbf']:.6f} | "
                f"mean_gap={row['mean_abs_diff_mean']:.6f} | "
                f"std_gap={row['std_abs_diff_mean']:.6f} | "
                f"corr_gap={row['corr_abs_diff_mean']:.6f} | "
                f"auc={row['two_sample_auc']:.6f}"
            )
        except Exception as e:
            print(f"[ERROR] {ckpt_path} failed: {e}")
            rows.append({"checkpoint_path": ckpt_path, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "checkpoint_ranking_raw.csv", index=False)

    if "selection_score" not in df.columns:
        print("[WARN] 没有成功评估的 checkpoint")
        return

    ok_df = df[df["selection_score"].notna()].copy()
    if len(ok_df) == 0:
        print("[WARN] 没有成功评估的 checkpoint")
        return

    ok_df = ok_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    ok_df.to_csv(out_dir / "checkpoint_ranking_final.csv", index=False)

    best_row = ok_df.iloc[0].to_dict()
    with open(out_dir / "best_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    print("\n===== FINAL RANKING =====")
    show_cols = [
        "checkpoint_path", "payload_epoch", "selection_score", "mmd_rbf",
        "mean_abs_diff_mean", "std_abs_diff_mean", "corr_abs_diff_mean",
        "two_sample_auc", "two_sample_acc",
    ]
    show_cols = [c for c in show_cols if c in ok_df.columns]
    print(ok_df[show_cols].to_string(index=False))

    print("\n[SELECTED BEST CHECKPOINT]")
    print(best_row["checkpoint_path"])
    print(f"selection_score = {best_row['selection_score']:.6f}")
    print(f"ranking saved -> {out_dir / 'checkpoint_ranking_final.csv'}")


if __name__ == "__main__":
    main()
