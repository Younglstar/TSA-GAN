import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from gan_config import GANConfig
from gan_data import EncodedDataset, inverse_scale_data, load_encoded_data, scale_data
from gan_models import Discriminator, Generator, MappingNetwork
from gan_trainer import EarlyStopping, GANTrainer


def plot_gan_loss_curves(history: list, save_path: str):
    history_df = pd.DataFrame(history).dropna()
    if history_df.empty or len(history_df) < 2:
        print("[WARN] 历史记录过少，跳过损失曲线绘制")
        return

    plt.figure(figsize=(12, 8))
    plt.plot(history_df["d_loss"], label="Discriminator Loss")
    plt.plot(history_df["g_loss"], label="Generator Loss")
    plt.plot(history_df["wasserstein_dist"], label="Wasserstein Distance", linestyle="--")
    if "gradient_penalty" in history_df.columns:
        plt.plot(history_df["gradient_penalty"], label="Gradient Penalty", linestyle=":")
    plt.title("GAN Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] loss curves saved -> {save_path}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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


def evaluate_synthetic_vs_test(real_test: np.ndarray, synthetic: np.ndarray, out_json_path: str, out_txt_path: str, out_pca_path: str):
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

    report = {
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

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt = []
    txt.append("=== GAN Synthetic vs Real Test Encoded Evaluation ===")
    txt.append("")
    txt.append(f"n_real_test = {report['n_real_test']}")
    txt.append(f"n_synthetic = {report['n_synthetic']}")
    txt.append(f"n_eval = {report['n_eval']}")
    txt.append("")
    txt.append(f"mean_abs_diff_mean = {report['mean_abs_diff_mean']:.6f}")
    txt.append(f"mean_abs_diff_max  = {report['mean_abs_diff_max']:.6f}")
    txt.append(f"std_abs_diff_mean  = {report['std_abs_diff_mean']:.6f}")
    txt.append(f"std_abs_diff_max   = {report['std_abs_diff_max']:.6f}")
    txt.append(f"corr_abs_diff_mean = {report['corr_abs_diff_mean']:.6f}")
    txt.append(f"corr_abs_diff_max  = {report['corr_abs_diff_max']:.6f}")
    txt.append(f"mmd_rbf            = {report['mmd_rbf']:.6f}")
    txt.append(f"two_sample_auc     = {report['two_sample_auc']:.6f}")
    txt.append(f"two_sample_acc     = {report['two_sample_acc']:.6f}")
    txt.append("")
    txt.append("Interpretation:")
    txt.append("- mean/std/corr 差异越小越好")
    txt.append("- mmd_rbf 越小越好")
    txt.append("- two_sample_auc 越接近 0.5 越好（说明 synthetic 和 real test 越难区分）")

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))

    real_p = subsample_rows(real_eval, 3000, seed=33)
    synth_p = subsample_rows(synth_eval, 3000, seed=44)
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
    plt.savefig(out_pca_path, dpi=150)
    plt.close()

    print(f"[INFO] eval json saved -> {out_json_path}")
    print(f"[INFO] eval txt saved  -> {out_txt_path}")
    print(f"[INFO] eval pca saved  -> {out_pca_path}")
    return report


def main():
    config = GANConfig()
    set_seed(config.SEED)

    os.makedirs(getattr(config, "CHECKPOINT_DIR", "./cache_gan/checkpoints"), exist_ok=True)

    eval_json_path = getattr(config, "EVAL_REPORT_JSON", "gan_eval_report.json")
    eval_txt_path = getattr(config, "EVAL_REPORT_TXT", "gan_eval_report.txt")
    eval_pca_path = getattr(config, "EVAL_PCA_PATH", "gan_eval_pca.png")

    print(f"[INFO] loading encoded data from: {config.ENCODED_DATA_PATH}")
    encoded_data = load_encoded_data(config.ENCODED_DATA_PATH)
    if not isinstance(encoded_data, np.ndarray):
        encoded_data = np.asarray(encoded_data, dtype=np.float32)
    encoded_data = encoded_data.astype(np.float32, copy=False)
    print(f"[INFO] encoded data shape = {encoded_data.shape}")

    train_data, test_data = train_test_split(
        encoded_data,
        test_size=config.VALID_SPLIT,
        random_state=config.SEED,
        shuffle=True,
    )
    print(f"[INFO] train size = {len(train_data)} | test size = {len(test_data)}")

    save_pickle(test_data, config.TEST_DATA_PATH)
    print(f"[INFO] test encoded data saved -> {config.TEST_DATA_PATH}")

    scaled_train_data, scaling_params = scale_data(train_data)
    save_pickle(scaling_params, config.SCALING_PARAMS_PATH)
    print(f"[INFO] scaling params saved -> {config.SCALING_PARAMS_PATH}")

    dataset = EncodedDataset(scaled_train_data)
    input_dim = scaled_train_data.shape[1]
    print(f"[INFO] GAN input_dim = {input_dim}")

    generator = Generator(
        w_dim=config.W_DIM,
        output_dim=input_dim,
        hidden_dims=config.GENERATOR_HIDDEN_DIMS,
    )
    discriminator = Discriminator(
        input_dim=input_dim,
        hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        dropout_rate=config.DROPOUT_RATE,
    )
    mapping_network = MappingNetwork(
        z_dim=config.NOISE_DIM,
        w_dim=config.W_DIM,
        hidden_layers=config.MAPPING_HIDDEN_LAYERS,
        hidden_dim=config.MAPPING_HIDDEN_DIM,
    )

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        mapping_network=mapping_network,
        config=config,
        dataset=dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.maybe_resume()

    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        monitor=config.EARLY_STOPPING_MONITOR,
        mode=config.EARLY_STOPPING_MODE,
    )

    history = []

    print("[INFO] start GAN training...")
    for epoch in range(trainer.start_epoch, config.NUM_EPOCHS):
        losses = trainer.train_epoch(epoch)
        history.append(losses)

        print(
            f"Epoch {epoch + 1:03d}/{config.NUM_EPOCHS} | "
            f"D={losses['d_loss']:.4f} | "
            f"G={losses['g_loss']:.4f} | "
            f"W={losses['wasserstein_dist']:.4f} | "
            f"GP={losses['gradient_penalty']:.4f} | "
            f"lr_g={losses['lr_g']:.2e} | "
            f"lr_d={losses['lr_d']:.2e}"
        )

        trainer.maybe_save_training_checkpoints(epoch=epoch, metrics=losses)

        if config.EARLY_STOPPING_ENABLE and early_stopper.step(losses):
            print(f"[INFO] early stop at epoch {epoch + 1}")
            break

    final_epoch = trainer.start_epoch + len(history) - 1
    trainer._save_checkpoint(
        path=config.LAST_MODEL_SAVE_PATH,
        epoch=final_epoch,
        metrics=history[-1] if history else None,
        extra={"kind": "final"},
    )
    print(f"[INFO] final save -> {config.LAST_MODEL_SAVE_PATH}")

    save_pickle(history, config.HISTORY_SAVE_PATH)
    print(f"[INFO] history saved -> {config.HISTORY_SAVE_PATH}")

    plot_gan_loss_curves(history, config.LOSS_CURVE_PATH)

    if os.path.exists(config.MODEL_SAVE_PATH):
        trainer.load_model(config.MODEL_SAVE_PATH)
        print(f"[INFO] reload best GAN -> {config.MODEL_SAVE_PATH}")
    else:
        print("[WARN] best model not found, fallback to current in-memory model")

    print("[INFO] generating synthetic samples from BEST GAN...")
    synthetic_scaled = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)

    with open(config.SCALING_PARAMS_PATH, "rb") as f:
        scaling_params = pickle.load(f)

    synthetic_data = inverse_scale_data(synthetic_scaled, scaling_params)
    save_pickle(synthetic_data, config.SYNTHETIC_DATA_PATH)
    print(f"[INFO] synthetic data saved -> {config.SYNTHETIC_DATA_PATH}")

    report = evaluate_synthetic_vs_test(
        real_test=test_data,
        synthetic=synthetic_data,
        out_json_path=eval_json_path,
        out_txt_path=eval_txt_path,
        out_pca_path=eval_pca_path,
    )

    print("[INFO] evaluation summary:")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.6f}")
        else:
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
