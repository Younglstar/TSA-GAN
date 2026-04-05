
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import DataConfig
from encdec_config import EncoderDecoderConfig
from embedding_config import EmbeddingConfig
from encdec_model import CausalVAE
from encdec_processor import DataProcessor


class MemMapDataset(Dataset):
    def __init__(self, x_memmap: np.memmap, m_memmap: np.memmap):
        self.x = x_memmap
        self.m = m_memmap
        if len(self.x) != len(self.m):
            raise ValueError("X/M 样本数不一致")
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.x[idx], copy=True)).float()
        m = torch.from_numpy(np.array(self.m[idx], copy=True)).float()
        return x, m, idx


def open_memmaps(encdec_config: EncoderDecoderConfig, feature_dims: dict) -> tuple[np.memmap, np.memmap]:
    cache_dir = getattr(encdec_config, "CACHE_DIR", ".")
    x_path = os.path.join(cache_dir, "encdec_X.f32")
    m_path = os.path.join(cache_dir, "encdec_M.u8")

    if not (os.path.exists(x_path) and os.path.exists(m_path)):
        raise FileNotFoundError(
            f"未找到 memmap 文件:\n  X: {x_path}\n  M: {m_path}\n请先运行 processor.process_data()。"
        )

    t = int(feature_dims["sequence_length"])
    d = int(feature_dims["input_dim"])
    n = int(feature_dims["num_subjects"])

    X = np.memmap(x_path, dtype="float32", mode="r", shape=(n, t, d))
    M = np.memmap(m_path, dtype="uint8", mode="r", shape=(n, t, d))
    return X, M


def build_train_val_subsets(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if val_size <= 0 or train_size <= 0:
        raise ValueError(f"数据集过小: len={len(dataset)}, val_ratio={val_ratio}")
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=g)
    return train_subset, val_subset


def make_loader(ds: Dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def build_model_from_feature_dims(
    encdec_config: EncoderDecoderConfig,
    feature_dims: dict,
    device: torch.device,
) -> CausalVAE:
    input_dim = int(feature_dims["input_dim"])
    sequence_length = int(feature_dims["sequence_length"])
    num_total_features = int(feature_dims["num_total_features"])

    model = CausalVAE(
        input_dim=input_dim,
        output_dim=input_dim,
        sequence_length=sequence_length,
        tcn_channels=encdec_config.TCN_CHANNELS,
        latent_dim=encdec_config.LATENT_DIM,
        num_total_features=num_total_features,
        dropout=encdec_config.DROPOUT_RATE,
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    state = payload["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return payload


@torch.inference_mode()
def encode_q_stats(
    model: CausalVAE,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, np.ndarray]:
    mu_list, logvar_list, idx_list = [], [], []

    for batch, _mask, idx in loader:
        batch = batch.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                mu, logvar, _A_logits = model.encoder(batch)
        else:
            mu, logvar, _A_logits = model.encoder(batch)
        mu_list.append(mu.float().cpu().numpy())
        logvar_list.append(logvar.float().cpu().numpy())
        idx_list.append(idx.cpu().numpy())

    mu = np.concatenate(mu_list, axis=0)
    logvar = np.concatenate(logvar_list, axis=0)
    idx = np.concatenate(idx_list, axis=0)
    order = np.argsort(idx)
    return {
        "mu": mu[order],
        "logvar": logvar[order],
        "idx": idx[order],
    }


def fit_standardizer(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = (x ** 2).sum(axis=1, keepdims=True)
    y2 = (y ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2 * x @ y.T, 0.0)


def subsample_rows(x: np.ndarray, max_n: int = 1024, seed: int = 42) -> np.ndarray:
    if len(x) <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


def median_heuristic_sigma(x: np.ndarray, max_points: int = 512) -> float:
    x = subsample_rows(x, max_points, seed=42)
    d2 = pairwise_sq_dists(x, x)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.sqrt(np.median(vals) + 1e-12))


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    d2 = pairwise_sq_dists(x, y)
    return np.exp(-d2 / (2.0 * sigma * sigma + 1e-12))


def mmd_rbf_unbiased_subsampled(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 1024,
    sigma: float | None = None,
) -> float:
    x = subsample_rows(x, max_samples, seed=42)
    y = subsample_rows(y, max_samples, seed=43)
    if sigma is None:
        sigma = median_heuristic_sigma(np.concatenate([x, y], axis=0), max_points=min(512, len(x) + len(y)))
    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)
    n = len(x)
    m = len(y)
    term_x = (Kxx.sum() - np.trace(Kxx)) / max(n * (n - 1), 1)
    term_y = (Kyy.sum() - np.trace(Kyy)) / max(m * (m - 1), 1)
    term_xy = Kxy.mean()
    return float(term_x + term_y - 2.0 * term_xy)


def covariance_effective_rank(mu: np.ndarray) -> Dict[str, float | list]:
    mu0 = mu - mu.mean(axis=0, keepdims=True)
    cov = np.cov(mu0, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eps = 1e-12
    p = eigvals / (eigvals.sum() + eps)
    effective_rank = np.exp(-(p * np.log(p + eps)).sum())
    return {
        "eigvals": eigvals.tolist(),
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / mu.shape[1]),
    }


def mean_abs_offdiag_corr(mu: np.ndarray) -> float:
    corr = np.corrcoef(mu, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.mean(np.abs(offdiag)))


def dead_dim_stats(mu: np.ndarray, dead_std_threshold: float = 1e-3) -> Dict:
    mean = mu.mean(axis=0)
    std = mu.std(axis=0)
    dead = std < dead_std_threshold
    return {
        "latent_dim": int(mu.shape[1]),
        "dead_dims": int(dead.sum()),
        "dead_ratio": float(dead.mean()),
        "mean_abs_mean": float(np.mean(np.abs(mean))),
        "std_min": float(std.min()),
        "std_median": float(np.median(std)),
        "std_max": float(std.max()),
        "std_to_one_gap_mean": float(np.mean(np.abs(std - 1.0))),
        "per_dim_std": std.tolist(),
        "per_dim_mean": mean.tolist(),
    }


def train_val_shift_stats(mu_train: np.ndarray, mu_val: np.ndarray) -> Dict:
    mean_tr = mu_train.mean(axis=0)
    mean_va = mu_val.mean(axis=0)
    std_tr = mu_train.std(axis=0)
    std_va = mu_val.std(axis=0)

    mean_gap = np.abs(mean_tr - mean_va)
    std_gap = np.abs(std_tr - std_va)
    denom = np.maximum(std_tr, 1e-6)

    return {
        "mean_gap_median": float(np.median(mean_gap)),
        "mean_gap_max": float(mean_gap.max()),
        "std_gap_median": float(np.median(std_gap)),
        "std_gap_max": float(std_gap.max()),
        "mean_gap_norm_median": float(np.median(mean_gap / denom)),
        "mean_gap_norm_max": float(np.max(mean_gap / denom)),
        "std_gap_norm_median": float(np.median(std_gap / denom)),
        "std_gap_norm_max": float(np.max(std_gap / denom)),
    }


def gaussian_aggregated_prior_stats(mu: np.ndarray, logvar: np.ndarray) -> Dict:
    mean_mu = mu.mean(axis=0)
    std_mu = mu.std(axis=0)
    mean_var = np.exp(logvar).mean(axis=0)
    return {
        "agg_mu_abs_mean": float(np.mean(np.abs(mean_mu))),
        "agg_mu_abs_max": float(np.max(np.abs(mean_mu))),
        "agg_mu_std_mean": float(np.mean(std_mu)),
        "agg_mu_std_to_one_gap_mean": float(np.mean(np.abs(std_mu - 1.0))),
        "agg_exp_logvar_mean": float(np.mean(mean_var)),
        "agg_exp_logvar_to_one_gap_mean": float(np.mean(np.abs(mean_var - 1.0))),
    }


@torch.inference_mode()
def interpolation_smoothness(
    model: CausalVAE,
    mu_pool: np.ndarray,
    device: torch.device,
    num_pairs: int = 8,
    num_steps: int = 6,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    mu_pool = subsample_rows(mu_pool, max_n=min(len(mu_pool), 2048), seed=seed)
    n = len(mu_pool)
    if n < 2:
        return {
            "interp_num_pairs": 0,
            "interp_num_steps": int(num_steps),
            "interp_avg_first_diff": float("nan"),
            "interp_avg_second_diff": float("nan"),
            "interp_curvature_ratio": float("nan"),
        }

    pairs = []
    for _ in range(min(num_pairs, n // 2 if n >= 2 else 0)):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((i, j))

    first_diffs = []
    second_diffs = []
    for i, j in pairs:
        outs = []
        for alpha in np.linspace(0.0, 1.0, num_steps):
            mu = (1.0 - alpha) * mu_pool[i] + alpha * mu_pool[j]
            mu_t = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
            recon_data, _mask_logits = model.decoder(mu_t)
            outs.append(recon_data.flatten().cpu().numpy())

        outs = np.stack(outs, axis=0)
        d1 = outs[1:] - outs[:-1]
        d2 = outs[2:] - 2.0 * outs[1:-1] + outs[:-2]
        first_diffs.append(np.linalg.norm(d1, axis=1).mean())
        second_diffs.append(np.linalg.norm(d2, axis=1).mean() if len(d2) > 0 else 0.0)

    first_mean = float(np.mean(first_diffs)) if first_diffs else float("nan")
    second_mean = float(np.mean(second_diffs)) if second_diffs else float("nan")
    return {
        "interp_num_pairs": int(len(pairs)),
        "interp_num_steps": int(num_steps),
        "interp_avg_first_diff": first_mean,
        "interp_avg_second_diff": second_mean,
        "interp_curvature_ratio": float(second_mean / max(first_mean, 1e-8)) if np.isfinite(first_mean) else float("nan"),
    }


def heuristic_score(metrics: Dict) -> Dict:
    dead_ratio = metrics["dead"]["dead_ratio"]
    eff_rank_ratio = metrics["cov"]["effective_rank_ratio"]
    shift_mean = metrics["shift"]["mean_gap_norm_median"]
    shift_std = metrics["shift"]["std_gap_norm_median"]
    mmd = metrics["mmd_train_val"]
    corr = metrics["corr_mean_abs_offdiag"]
    curvature = metrics["interp"]["interp_curvature_ratio"]
    mu_gap = metrics["gaussian"]["agg_mu_abs_mean"]
    std_gap = metrics["dead"]["std_to_one_gap_mean"]

    score = 0.0
    score += 0.20 * max(0.0, 1.0 - min(dead_ratio / 0.25, 1.0))
    score += 0.20 * min(eff_rank_ratio / 0.70, 1.0)
    score += 0.15 * max(0.0, 1.0 - min(shift_mean / 0.20, 1.0))
    score += 0.10 * max(0.0, 1.0 - min(shift_std / 0.20, 1.0))
    score += 0.10 * max(0.0, 1.0 - min(mmd / 0.20, 1.0))
    score += 0.10 * max(0.0, 1.0 - min(corr / 0.50, 1.0))
    score += 0.10 * max(0.0, 1.0 - min(curvature / 0.50, 1.0))
    score += 0.05 * max(0.0, 1.0 - min(mu_gap / 0.30, 1.0))
    score += 0.05 * max(0.0, 1.0 - min(std_gap / 0.50, 1.0))
    return {"gan_readiness_score": float(score)}


def save_pca_plot(mu_train: np.ndarray, mu_val: np.ndarray, out_path: str, title: str) -> None:
    mu_train_p = subsample_rows(mu_train, 3000, seed=1)
    mu_val_p = subsample_rows(mu_val, 3000, seed=2)
    pca = PCA(n_components=2)
    X = np.concatenate([mu_train_p, mu_val_p], axis=0)
    Z = pca.fit_transform(X)
    n_train = len(mu_train_p)
    z_train = Z[:n_train]
    z_val = Z[n_train:]

    plt.figure(figsize=(6, 5))
    plt.scatter(z_train[:, 0], z_train[:, 1], s=6, alpha=0.35, label="train")
    plt.scatter(z_val[:, 0], z_val[:, 1], s=6, alpha=0.35, label="val")
    plt.legend()
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_std_bar_plot(mu_train: np.ndarray, out_path: str, title: str) -> None:
    std = mu_train.std(axis=0)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(std)), std)
    plt.title(title)
    plt.xlabel("latent dim")
    plt.ylabel("std")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_eigs_plot(eigvals: List[float], out_path: str, title: str) -> None:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")
    plt.title(title)
    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def grade_dead_ratio(x: float) -> str:
    if x < 0.10:
        return "A"
    if x < 0.25:
        return "B"
    return "C"


def grade_eff_rank_ratio(x: float) -> str:
    if x >= 0.60:
        return "A"
    if x >= 0.40:
        return "B"
    return "C"


def grade_shift(x: float) -> str:
    if x < 0.10:
        return "A"
    if x < 0.25:
        return "B"
    return "C"


def grade_mmd(x: float) -> str:
    if x < 0.05:
        return "A"
    if x < 0.15:
        return "B"
    return "C"


def grade_corr(x: float) -> str:
    if x < 0.20:
        return "A"
    if x < 0.35:
        return "B"
    return "C"


def grade_curvature(x: float) -> str:
    if x < 0.10:
        return "A"
    if x < 0.30:
        return "B"
    return "C"


def build_grade_sheet(metrics: Dict) -> Dict:
    return {
        "dead_ratio_grade": grade_dead_ratio(metrics["dead"]["dead_ratio"]),
        "eff_rank_ratio_grade": grade_eff_rank_ratio(metrics["cov"]["effective_rank_ratio"]),
        "train_val_mean_shift_grade": grade_shift(metrics["shift"]["mean_gap_norm_median"]),
        "train_val_std_shift_grade": grade_shift(metrics["shift"]["std_gap_norm_median"]),
        "mmd_grade": grade_mmd(metrics["mmd_train_val"]),
        "corr_grade": grade_corr(metrics["corr_mean_abs_offdiag"]),
        "interp_curvature_grade": grade_curvature(metrics["interp"]["interp_curvature_ratio"]),
    }


def run_one_checkpoint(
    ckpt_path: str,
    out_dir: Path,
    model_builder_cfg: EncoderDecoderConfig,
    feature_dims: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_arrays: bool = True,
    mmd_max_samples: int = 1024,
    interp_pairs: int = 8,
    interp_steps: int = 6,
    use_amp: bool = True,
) -> Dict:
    ckpt_name = Path(ckpt_path).stem
    ckpt_out = out_dir / ckpt_name
    ckpt_out.mkdir(parents=True, exist_ok=True)

    stage_t0 = time.perf_counter()
    model = build_model_from_feature_dims(model_builder_cfg, feature_dims, device)
    payload = load_checkpoint_to_model(model, ckpt_path, device)
    t_model = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    train_stats = encode_q_stats(model, train_loader, device, use_amp=use_amp)
    val_stats = encode_q_stats(model, val_loader, device, use_amp=use_amp)
    t_encode = time.perf_counter() - stage_t0

    mu_train = train_stats["mu"]
    mu_val = val_stats["mu"]
    logvar_train = train_stats["logvar"]
    logvar_val = val_stats["logvar"]

    if save_arrays:
        np.save(ckpt_out / "mu_train.npy", mu_train)
        np.save(ckpt_out / "mu_val.npy", mu_val)
        np.save(ckpt_out / "logvar_train.npy", logvar_train)
        np.save(ckpt_out / "logvar_val.npy", logvar_val)

    stage_t0 = time.perf_counter()
    mean_tr, std_tr = fit_standardizer(mu_train)
    mu_train_std = transform_standardize(mu_train, mean_tr, std_tr)
    mu_val_std = transform_standardize(mu_val, mean_tr, std_tr)
    if save_arrays:
        np.save(ckpt_out / "mu_train_std.npy", mu_train_std)
        np.save(ckpt_out / "mu_val_std.npy", mu_val_std)

    dead = dead_dim_stats(mu_train)
    shift = train_val_shift_stats(mu_train_std, mu_val_std)
    cov = covariance_effective_rank(mu_train)
    corr = mean_abs_offdiag_corr(mu_train)
    mmd = mmd_rbf_unbiased_subsampled(mu_train_std, mu_val_std, max_samples=mmd_max_samples)
    gaussian = gaussian_aggregated_prior_stats(mu_train, logvar_train)
    interp = interpolation_smoothness(model, mu_val, device=device, num_pairs=interp_pairs, num_steps=interp_steps)
    t_metrics = time.perf_counter() - stage_t0

    metrics = {
        "checkpoint_path": ckpt_path,
        "payload_epoch": int(payload.get("epoch", -1)),
        "payload_best_metric": float(payload.get("best_metric", np.nan)),
        "dead": dead,
        "shift": shift,
        "cov": cov,
        "corr_mean_abs_offdiag": corr,
        "mmd_train_val": mmd,
        "gaussian": gaussian,
        "interp": interp,
        "timing_sec": {
            "build_and_load_model": t_model,
            "encode_train_val": t_encode,
            "metrics": t_metrics,
            "total": t_model + t_encode + t_metrics,
        },
    }
    metrics["score"] = heuristic_score(metrics)
    metrics["grade"] = build_grade_sheet(metrics)

    stage_t0 = time.perf_counter()
    save_pca_plot(mu_train_std, mu_val_std, str(ckpt_out / "pca_train_val_std.png"), f"{ckpt_name} PCA(train/val standardized mu)")
    save_std_bar_plot(mu_train, str(ckpt_out / "latent_std_bar.png"), f"{ckpt_name} latent std per dim")
    save_eigs_plot(metrics["cov"]["eigvals"], str(ckpt_out / "eigvals.png"), f"{ckpt_name} covariance eigvals")
    metrics["timing_sec"]["plots"] = time.perf_counter() - stage_t0
    metrics["timing_sec"]["total_with_plots"] = metrics["timing_sec"]["total"] + metrics["timing_sec"]["plots"]

    with open(ckpt_out / "report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def print_summary(report: Dict) -> None:
    print(f"\n===== {report['checkpoint_path']} =====")
    print(f"epoch={report['payload_epoch'] + 1} | best_metric={report['payload_best_metric']:.6f}")
    print(f"dead_ratio={report['dead']['dead_ratio']:.4f} ({report['grade']['dead_ratio_grade']})")
    print(f"eff_rank_ratio={report['cov']['effective_rank_ratio']:.4f} ({report['grade']['eff_rank_ratio_grade']})")
    print(f"mean_shift_norm_median={report['shift']['mean_gap_norm_median']:.4f} ({report['grade']['train_val_mean_shift_grade']})")
    print(f"std_shift_norm_median={report['shift']['std_gap_norm_median']:.4f} ({report['grade']['train_val_std_shift_grade']})")
    print(f"mmd_train_val={report['mmd_train_val']:.4f} ({report['grade']['mmd_grade']})")
    print(f"mean_abs_offdiag_corr={report['corr_mean_abs_offdiag']:.4f} ({report['grade']['corr_grade']})")
    print(f"interp_curvature_ratio={report['interp']['interp_curvature_ratio']:.4f} ({report['grade']['interp_curvature_grade']})")
    print(f"gan_readiness_score={report['score']['gan_readiness_score']:.4f}")
    print(
        "timing_sec="
        f"load:{report['timing_sec']['build_and_load_model']:.1f}, "
        f"encode:{report['timing_sec']['encode_train_val']:.1f}, "
        f"metrics:{report['timing_sec']['metrics']:.1f}, "
        f"plots:{report['timing_sec']['plots']:.1f}, "
        f"total:{report['timing_sec']['total_with_plots']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fast compare checkpoints and diagnose latent space for downstream GAN.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint paths to compare")
    parser.add_argument("--out_dir", type=str, default="./latent_ckpt_compare_fast", help="output directory")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--diag_batch_size", type=int, default=256, help="batch size for encoding diagnostics")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mmd_max_samples", type=int, default=1024)
    parser.add_argument("--interp_pairs", type=int, default=8)
    parser.add_argument("--interp_steps", type=int, default=6)
    parser.add_argument("--skip_process_data", action="store_true", help="skip processor.process_data(), use existing cache")
    parser.add_argument("--no_save_arrays", action="store_true", help="do not save mu/logvar arrays")
    parser.add_argument("--no_amp", action="store_true", help="disable AMP for encoding")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    data_cfg = DataConfig()
    encdec_cfg = EncoderDecoderConfig()
    embed_cfg = EmbeddingConfig()

    global_t0 = time.perf_counter()

    processor = DataProcessor(encdec_cfg, data_cfg, embed_cfg)
    if not args.skip_process_data:
        t0 = time.perf_counter()
        _ = processor.process_data()
        print(f"[GLOBAL] process_data() done in {time.perf_counter() - t0:.1f}s")
    else:
        print("[GLOBAL] skip process_data(), use existing cache")

    feature_dims = processor.load_feature_dims()
    X_mm, M_mm = open_memmaps(encdec_cfg, feature_dims)
    dataset = MemMapDataset(X_mm, M_mm)
    train_ds, val_ds = build_train_val_subsets(
        dataset,
        val_ratio=float(encdec_cfg.VAL_RATIO),
        seed=int(encdec_cfg.SEED),
    )
    train_loader = make_loader(train_ds, batch_size=args.diag_batch_size, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=args.diag_batch_size, num_workers=args.num_workers)

    print(
        f"[GLOBAL] dataset ready | train={len(train_ds)} val={len(val_ds)} | "
        f"diag_batch_size={args.diag_batch_size} | num_workers={args.num_workers}"
    )

    reports = []
    for ckpt in args.checkpoints:
        ckpt_t0 = time.perf_counter()
        report = run_one_checkpoint(
            ckpt_path=ckpt,
            out_dir=out_dir,
            model_builder_cfg=encdec_cfg,
            feature_dims=feature_dims,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_arrays=(not args.no_save_arrays),
            mmd_max_samples=args.mmd_max_samples,
            interp_pairs=args.interp_pairs,
            interp_steps=args.interp_steps,
            use_amp=(not args.no_amp),
        )
        reports.append(report)
        print_summary(report)
        print(f"[GLOBAL] finished {ckpt} in {time.perf_counter() - ckpt_t0:.1f}s")

    ranking = sorted(
        [
            {
                "checkpoint_path": r["checkpoint_path"],
                "payload_epoch": r["payload_epoch"],
                "gan_readiness_score": r["score"]["gan_readiness_score"],
                "dead_ratio": r["dead"]["dead_ratio"],
                "effective_rank_ratio": r["cov"]["effective_rank_ratio"],
                "mmd_train_val": r["mmd_train_val"],
                "interp_curvature_ratio": r["interp"]["interp_curvature_ratio"],
                "timing_total_with_plots": r["timing_sec"]["total_with_plots"],
            }
            for r in reports
        ],
        key=lambda x: x["gan_readiness_score"],
        reverse=True,
    )
    with open(out_dir / "ranking.json", "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)

    print("\n===== RANKING =====")
    for i, item in enumerate(ranking, start=1):
        print(
            f"{i}. {item['checkpoint_path']} | "
            f"score={item['gan_readiness_score']:.4f} | "
            f"dead_ratio={item['dead_ratio']:.4f} | "
            f"eff_rank_ratio={item['effective_rank_ratio']:.4f} | "
            f"mmd={item['mmd_train_val']:.4f} | "
            f"interp_curvature={item['interp_curvature_ratio']:.4f} | "
            f"time={item['timing_total_with_plots']:.1f}s"
        )

    print(f"\n[GLOBAL] all done in {time.perf_counter() - global_t0:.1f}s")


if __name__ == "__main__":
    main()
