from __future__ import annotations

from typing import Tuple
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from causal_target_builder import CausalTargetBuilderConfig, build_global_causal_target

from config import DataConfig
from encdec_config import EncoderDecoderConfig
from embedding_config import EmbeddingConfig
from encdec_model import CausalVAE
from encdec_trainer import EncoderDecoderTrainer
from encdec_processor import DataProcessor
from utils.io import save_pickle
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not _dist_is_ready()) or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if _dist_is_ready() else 1


def barrier() -> None:
    if _dist_is_ready():
        dist.barrier()


def init_distributed(cfg: EncoderDecoderConfig) -> tuple[bool, int, int, int, torch.device]:
    use_ddp = bool(getattr(cfg, "USE_DDP", False)) and torch.cuda.device_count() > 1
    if use_ddp and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = str(getattr(cfg, "DDP_BACKEND", "nccl"))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def cleanup_distributed() -> None:
    if _dist_is_ready():
        dist.barrier()
        dist.destroy_process_group()


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


def plot_loss_curves(train_history: list, val_history: list, save_path: str) -> None:
    if not train_history or not val_history:
        logger.warning("Loss history is empty; skip plotting.")
        return

    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history)

    epochs = np.arange(1, len(train_history) + 1)
    loss_names = list(train_df.columns)

    fig, axes = plt.subplots(len(loss_names), 1, figsize=(10, 4.5 * len(loss_names)), sharex=True)
    if len(loss_names) == 1:
        axes = [axes]

    for i, loss_name in enumerate(loss_names):
        axes[i].plot(epochs, train_df[loss_name], label=f"Train {loss_name}")
        axes[i].plot(epochs, val_df[loss_name], label=f"Val {loss_name}", linestyle="--")
        axes[i].set_ylabel("Value")
        axes[i].set_title(loss_name.replace("_", " ").title())
        axes[i].legend()
        axes[i].grid(True)
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xlim(left=1, right=len(epochs))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved loss curves -> %s", save_path)


def _save_aux_checkpoints(
    trainer: EncoderDecoderTrainer,
    encdec_config: EncoderDecoderConfig,
    *,
    epoch: int,
    best_val_score: float,
) -> None:
    if not is_main_process():
        return

    ckpt_dir = getattr(encdec_config, "CHECKPOINT_DIR", "./cache_encdec/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if bool(getattr(encdec_config, "SAVE_LAST_EVERY_EPOCH", True)):
        last_path = os.path.join(ckpt_dir, "cvae_last.pkl")
        trainer.save_checkpoint(last_path, epoch=epoch, best_metric=best_val_score)

    save_every_epoch = bool(getattr(encdec_config, "SAVE_EVERY_EPOCH", False))
    save_every_n = int(getattr(encdec_config, "SAVE_EVERY_N_EPOCHS", 0))

    need_periodic_save = False
    if save_every_epoch:
        need_periodic_save = True
    elif save_every_n > 0 and ((epoch + 1) % save_every_n == 0):
        need_periodic_save = True

    if need_periodic_save:
        periodic_path = os.path.join(ckpt_dir, f"cvae_epoch_{epoch + 1:03d}.pkl")
        trainer.save_checkpoint(periodic_path, epoch=epoch, best_metric=best_val_score)
        logger.info("Saved periodic checkpoint -> %s", periodic_path)


def _open_memmaps(encdec_config: EncoderDecoderConfig, feature_dims: dict) -> tuple[np.memmap, np.memmap]:
    cache_dir = getattr(encdec_config, "CACHE_DIR", ".")
    x_path = os.path.join(cache_dir, "encdec_X.f32")
    m_path = os.path.join(cache_dir, "encdec_M.u8")

    if not (os.path.exists(x_path) and os.path.exists(m_path)):
        raise FileNotFoundError(
            f"未找到 memmap 文件：\n  X: {x_path}\n  M: {m_path}\n请先运行 processor.process_data() 生成它们。"
        )

    t = int(feature_dims["sequence_length"])
    d = int(feature_dims["input_dim"])

    n = feature_dims.get("num_subjects", None)
    if n is None:
        x_bytes = os.path.getsize(x_path)
        bytes_per_sample = t * d * 4
        if x_bytes % bytes_per_sample != 0:
            raise ValueError("无法从文件大小推断 num_subjects，请在 feature_dims 中保存它")
        n = x_bytes // bytes_per_sample
        feature_dims["num_subjects"] = int(n)

    n = int(n)
    x = np.memmap(x_path, dtype="float32", mode="r", shape=(n, t, d))
    m = np.memmap(m_path, dtype="uint8", mode="r", shape=(n, t, d))
    return x, m


def _make_dataloaders(dataset: Dataset, encdec_config: EncoderDecoderConfig, distributed: bool) -> tuple[DataLoader, DataLoader]:
    val_ratio = float(getattr(encdec_config, "VAL_RATIO", 0.2))
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if val_size <= 0 or train_size <= 0:
        raise ValueError(f"Dataset 太小或 VAL_RATIO 不合理：len={len(dataset)}, val_ratio={val_ratio}")

    seed = int(getattr(encdec_config, "SEED", 42))
    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    num_workers = int(getattr(encdec_config, "NUM_WORKERS", 0))
    pin_memory = bool(getattr(encdec_config, "PIN_MEMORY", torch.cuda.is_available())) and torch.cuda.is_available()
    prefetch_factor = int(getattr(encdec_config, "PREFETCH_FACTOR", 2)) if num_workers > 0 else None

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=encdec_config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=encdec_config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )
    return train_loader, val_loader


'''def _get_val_score(val_losses: dict, cfg) -> float:
    recon = float(val_losses.get("recon_loss", np.nan))
    causal = float(val_losses.get("causal_loss", np.nan))
    kld = float(val_losses.get("kld_loss", np.nan))
    beta = float(val_losses.get("current_beta", 0.0))

    score = recon + float(cfg.GAMMA_CAUSAL) * causal + beta * kld
    return score if np.isfinite(score) else 1e30'''
    
def _get_val_score(val_losses: dict, cfg) -> float:
    recon = float(val_losses.get("recon_loss", 1e30))
    causal = float(val_losses.get("causal_loss", 0.0))
    kld = float(val_losses.get("kld_free", 0.0))
    beta = float(val_losses.get("current_beta", 0.0))
    latent_corr = float(val_losses.get("latent_corr_loss", 0.0))
    latent_var = float(val_losses.get("latent_var_loss", 0.0))

    return (
        recon
        + float(cfg.GAMMA_CAUSAL) * causal
        + beta * kld
        + float(getattr(cfg, "LAMBDA_LATENT_CORR", 0.0)) * latent_corr
        + float(getattr(cfg, "LAMBDA_LATENT_VAR", 0.0)) * latent_var
    )

def _expected_graph_size(feature_dims: dict, encdec_config: EncoderDecoderConfig) -> int:
    use_temporal_only = bool(getattr(encdec_config, "CAUSAL_USE_ONLY_TEMPORAL_FEATURES", False))
    if use_temporal_only:
        if "num_temporal_graph_features" in feature_dims:
            return int(feature_dims["num_temporal_graph_features"])
        temporal_slices = feature_dims.get("temporal_feature_slices", {})
        if isinstance(temporal_slices, dict) and temporal_slices:
            return len(temporal_slices)
    return int(feature_dims["num_total_features"])


def _build_or_reuse_causal_target(encdec_config: EncoderDecoderConfig, feature_dims: dict) -> None:
    path = encdec_config.CAUSAL_TARGET_PATH
    expected_n = _expected_graph_size(feature_dims, encdec_config)

    def _validate_existing_target() -> bool:
        if not os.path.exists(path):
            return False
        try:
            a = np.load(path)
        except Exception:
            return False
        ok = a.shape == (expected_n, expected_n)
        if is_main_process():
            logger.info("Found pre-generated A_target at %s | shape=%s | expected=%s", path, a.shape, (expected_n, expected_n))
        return ok

    if _validate_existing_target():
        return

    if is_main_process() and os.path.exists(path):
        logger.warning("Existing A_target shape mismatch or unreadable, regenerating: %s", path)
        try:
            os.remove(path)
        except OSError:
            pass

    if not is_main_process():
        return

    logger.info("Generating A_target ...")
    builder_cfg = CausalTargetBuilderConfig(
        cache_dir=encdec_config.CACHE_DIR,
        feature_dims_file=encdec_config.FEATURE_DIMS_FILE,
        output_path=encdec_config.CAUSAL_TARGET_PATH,
        max_lag=int(getattr(encdec_config, "CAUSAL_MAX_LAG", 1)),
        min_valid_points=int(getattr(encdec_config, "CAUSAL_MIN_VALID_POINTS", 10)),
        ridge_alpha=float(getattr(encdec_config, "CAUSAL_RIDGE_ALPHA", 1e-4)),
        zero_diagonal=bool(getattr(encdec_config, "CAUSAL_ZERO_DIAGONAL", True)),
        standardize_per_subject=bool(getattr(encdec_config, "CAUSAL_STANDARDIZE_PER_SUBJECT", False)),
        use_only_temporal_features=bool(getattr(encdec_config, "CAUSAL_USE_ONLY_TEMPORAL_FEATURES", False)),
        threshold=float(getattr(encdec_config, "CAUSAL_THRESHOLD", 0.0)),
        keep_topk_per_target=int(getattr(encdec_config, "CAUSAL_KEEP_TOPK_PER_TARGET", 0)),
        symmetrize=bool(getattr(encdec_config, "CAUSAL_SYMMETRIZE", False)),
        save_improve_matrix=bool(getattr(encdec_config, "CAUSAL_SAVE_IMPROVE_MATRIX", True)),
        save_valid_count_matrix=bool(getattr(encdec_config, "CAUSAL_SAVE_VALID_COUNT_MATRIX", True)),
        save_metadata=bool(getattr(encdec_config, "CAUSAL_SAVE_METADATA", True)),
        verbose=bool(getattr(encdec_config, "CAUSAL_VERBOSE", True)),
    )
    a = build_global_causal_target(builder_cfg)
    if a.shape != (expected_n, expected_n):
        raise ValueError(
            f"A_target shape mismatch: got {a.shape}, expected {(expected_n, expected_n)}. "
            f"请检查 feature_dims 中的 active_graph_feature_slices / temporal_feature_slices 是否与 "
            f"CAUSAL_USE_ONLY_TEMPORAL_FEATURES 设置一致。"
        )


def train_vae(
    encdec_config: EncoderDecoderConfig,
    data_config: DataConfig,
    embed_config: EmbeddingConfig,
) -> Tuple[EncoderDecoderTrainer, DataProcessor]:
    if torch.cuda.is_available() and bool(getattr(encdec_config, "TF32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    distributed, rank, world_size, local_rank, device = init_distributed(encdec_config)

    try:
        if is_main_process():
            logger.info("Step: encdec | training CausalVAE")
            logger.info("Device: %s | distributed=%s | world_size=%d", device, distributed, world_size)

        processor = DataProcessor(encdec_config, data_config, embed_config)

        if is_main_process():
            _ = processor.process_data()
        barrier()

        feature_dims = processor.load_feature_dims()
        input_dim = int(feature_dims["input_dim"])
        sequence_length = int(feature_dims["sequence_length"])
        num_total_features = int(feature_dims["num_total_features"])

        if is_main_process():
            logger.info(
                "Dims | input_dim=%s seq_len=%s num_total_features=%s",
                input_dim, sequence_length, num_total_features,
            )

        _build_or_reuse_causal_target(encdec_config, feature_dims)
        barrier()

        x_mm, m_mm = _open_memmaps(encdec_config, feature_dims)
        dataset = MemMapDataset(x_mm, m_mm)
        train_loader, val_loader = _make_dataloaders(dataset, encdec_config, distributed=distributed)

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

        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=bool(getattr(encdec_config, "DDP_FIND_UNUSED_PARAMETERS", False)),
                broadcast_buffers=bool(getattr(encdec_config, "DDP_BROADCAST_BUFFERS", False)),
            )

        trainer = EncoderDecoderTrainer(model, encdec_config, device)

        ckpt_path = getattr(encdec_config, "RESUME_PATH", encdec_config.MODEL_SAVE_PATH)
        resume = bool(getattr(encdec_config, "RESUME_TRAINING", False))
        reset_opt = bool(getattr(encdec_config, "RESET_OPTIMIZER_ON_RESUME", False))
        reset_best_on_resume = bool(getattr(encdec_config, "RESET_BEST_ON_RESUME", True))

        start_epoch, best_val_score = trainer.maybe_resume(
            ckpt_path,
            resume=resume,
            reset_optimizer=reset_opt,
        )

        if resume and os.path.exists(ckpt_path):
            if is_main_process():
                logger.info(
                    "Resume enabled | ckpt=%s | start_epoch=%d | best_val_score=%0.6f | reset_optimizer=%s",
                    ckpt_path, start_epoch, best_val_score, reset_opt
                )
            if reset_best_on_resume:
                if is_main_process():
                    logger.info("RESET_BEST_ON_RESUME=True -> reset best_val_score to inf for current scoring rule")
                best_val_score = float("inf")
        else:
            if is_main_process():
                logger.info("Start from scratch | resume=%s | ckpt_exists=%s", resume, os.path.exists(ckpt_path))
            best_val_score = float("inf")

        best_epoch_metrics = {}
        best_epoch_idx = 0
        epochs_no_improve = 0
        train_history, val_history = [], []

        scheduler_metric = getattr(encdec_config, "SCHEDULER_METRIC", "recon_loss")
        best_start_epoch = int(getattr(encdec_config, "BEST_SCORE_START_EPOCH", 1))

        for epoch in range(start_epoch, encdec_config.NUM_EPOCHS):
            if is_main_process():
                logger.info("Epoch %d/%d", epoch + 1, encdec_config.NUM_EPOCHS)

            if distributed:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                if hasattr(val_loader.sampler, "set_epoch"):
                    val_loader.sampler.set_epoch(epoch)

            train_losses = trainer.train_epoch(train_loader, epoch)
            val_losses = trainer.validate(val_loader, epoch, scheduler_metric=scheduler_metric)

            train_history.append(train_losses)
            val_history.append(val_losses)

            current_score = _get_val_score(val_losses, encdec_config)

            if (epoch + 1) < best_start_epoch:
                if is_main_process():
                    logger.info("Skip best selection before epoch %d", best_start_epoch)
                _save_aux_checkpoints(
                    trainer,
                    encdec_config,
                    epoch=epoch,
                    best_val_score=best_val_score,
                )
                continue

            if np.isfinite(current_score) and current_score < best_val_score:
                best_val_score = current_score
                best_epoch_metrics = val_losses
                best_epoch_idx = epoch + 1
                epochs_no_improve = 0

                trainer.save_checkpoint(ckpt_path, epoch=epoch, best_metric=best_val_score)
                if is_main_process():
                    logger.info("New best val_score=%0.6f | saved ckpt -> %s", best_val_score, ckpt_path)
            else:
                epochs_no_improve += 1
                if is_main_process():
                    logger.info("No improve (%d/%d)", epochs_no_improve, encdec_config.EARLY_STOPPING_PATIENCE)

            _save_aux_checkpoints(
                trainer,
                encdec_config,
                epoch=epoch,
                best_val_score=best_val_score,
            )

            if epochs_no_improve >= encdec_config.EARLY_STOPPING_PATIENCE:
                if is_main_process():
                    logger.info("Early stopping triggered")
                break

        if is_main_process():
            logger.info("=" * 50)
            logger.info("TRAINING FINISHED")
            logger.info("Best Epoch: %s", best_epoch_idx)
            logger.info("Best Validation Metrics:")
            for k, v in best_epoch_metrics.items():
                if k in {"struct_loss", "dag_loss", "causal_loss"}:
                    logger.info("  - %s: %.8e", k, float(v))
                else:
                    logger.info("  - %s: %.6f", k, float(v))
            logger.info(
                "Best Validation Score (recon + gamma*causal + beta*kld): %.6f",
                float(best_val_score)
            )
            logger.info("=" * 50)
            plot_loss_curves(train_history, val_history, save_path="cvae_loss_curves.png")

        barrier()
        if os.path.exists(ckpt_path):
            trainer.load_checkpoint(ckpt_path, reset_optimizer=True)
        barrier()

        return trainer, processor
    except Exception:
        cleanup_distributed()
        raise


def encode_all(
    trainer: EncoderDecoderTrainer,
    processor: DataProcessor,
    encdec_config: EncoderDecoderConfig,
) -> np.ndarray:
    feature_dims = processor.load_feature_dims()
    x_mm, m_mm = _open_memmaps(encdec_config, feature_dims)

    full_dataset = MemMapDataset(x_mm, m_mm)
    num_workers = int(getattr(encdec_config, "NUM_WORKERS", 0))
    pin_memory = bool(getattr(encdec_config, "PIN_MEMORY", torch.cuda.is_available())) and torch.cuda.is_available()
    prefetch_factor = int(getattr(encdec_config, "PREFETCH_FACTOR", 2)) if num_workers > 0 else None

    distributed = _dist_is_ready()
    full_sampler = DistributedSampler(full_dataset, shuffle=False, drop_last=False) if distributed else None

    full_loader = DataLoader(
        full_dataset,
        batch_size=encdec_config.BATCH_SIZE,
        shuffle=False,
        sampler=full_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        drop_last=False,
    )

    encoded_mu_list = []
    encoded_idx_list = []
    model_to_encode = trainer.model.module if isinstance(trainer.model, DDP) else (trainer.model.module if hasattr(trainer.model, 'module') else trainer.model)
    model_to_encode.eval()

    with torch.inference_mode():
        pbar = tqdm(full_loader, desc="Encoding", disable=not is_main_process())
        for batch, _mask, idx in pbar:
            batch = batch.to(trainer.device, non_blocking=True)
            mu, _, _ = model_to_encode.encoder(batch)
            encoded_mu_list.append(mu.cpu().numpy())
            encoded_idx_list.append(idx.cpu().numpy())

    local_mu = np.concatenate(encoded_mu_list, axis=0) if encoded_mu_list else np.empty((0, encdec_config.LATENT_DIM), dtype=np.float32)
    local_idx = np.concatenate(encoded_idx_list, axis=0) if encoded_idx_list else np.empty((0,), dtype=np.int64)

    if distributed:
        gathered = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered, {"idx": local_idx, "mu": local_mu})
        if is_main_process():
            idx_all = np.concatenate([g["idx"] for g in gathered], axis=0)
            mu_all = np.concatenate([g["mu"] for g in gathered], axis=0)
            order = np.argsort(idx_all)
            idx_all = idx_all[order]
            mu_all = mu_all[order]
            _, first_pos = np.unique(idx_all, return_index=True)
            encoded_data = mu_all[first_pos]
            save_pickle(encoded_data, encdec_config.ENCODED_DATA_PATH)
            logger.info("Saved encoded data (shape=%s) -> %s", encoded_data.shape, encdec_config.ENCODED_DATA_PATH)
        else:
            encoded_data = np.empty((0, encdec_config.LATENT_DIM), dtype=np.float32)
        barrier()
        return encoded_data

    encoded_data = local_mu
    save_pickle(encoded_data, encdec_config.ENCODED_DATA_PATH)
    logger.info("Saved encoded data (shape=%s) -> %s", encoded_data.shape, encdec_config.ENCODED_DATA_PATH)
    return encoded_data


def train_and_encode(
    data_config: DataConfig | None = None,
    encdec_config: EncoderDecoderConfig | None = None,
    embed_config: EmbeddingConfig | None = None,
) -> np.ndarray:
    dcfg = data_config or DataConfig()
    ecfg = encdec_config or EncoderDecoderConfig()
    emcfg = embed_config or EmbeddingConfig()

    trainer, processor = train_vae(ecfg, dcfg, emcfg)
    encoded = encode_all(trainer, processor, ecfg)
    cleanup_distributed()
    return encoded


def main():
    train_and_encode()


if __name__ == "__main__":
    main()
