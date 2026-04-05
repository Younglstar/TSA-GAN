from __future__ import annotations

import os
import pickle
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


EXPECTED_METRIC_KEYS = [
    "total_loss",
    "recon_loss",
    "recon_data_loss",
    "recon_mask_loss",
    "kld_loss",
    "kld_free",
    "struct_loss",
    "dag_loss",
    "causal_loss",
    "latent_corr_loss",
    "latent_var_loss",
    "current_beta",
]


class _RunningAverages:
    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.sums: Dict[str, float] = {k: 0.0 for k in self.keys}
        self.counts: Dict[str, float] = {k: 0.0 for k in self.keys}

    def update(self, values: Dict[str, float], weight: float = 1.0) -> None:
        for k in self.keys:
            v = values.get(k, float("nan"))
            if np.isfinite(v):
                self.sums[k] += float(v) * weight
                self.counts[k] += float(weight)

    def means(self) -> Dict[str, float]:
        out = {}
        for k in self.keys:
            out[k] = self.sums[k] / self.counts[k] if self.counts[k] > 0 else float("nan")
        return out


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _dist_is_ready()) or dist.get_rank() == 0


class EncoderDecoderTrainer:
    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.is_distributed = _dist_is_ready()
        self.is_main = _is_main_process()

        self.lr = float(getattr(config, "LEARNING_RATE", 1e-3))
        self.scheduler_factor = float(getattr(config, "SCHEDULER_FACTOR", 0.1))
        self.scheduler_patience = int(getattr(config, "SCHEDULER_PATIENCE", 10))

        self.input_noise_std = float(getattr(config, "INPUT_NOISE_STD", 0.0))
        self.mask_loss_only_on_observed = bool(getattr(config, "MASK_LOSS_ONLY_ON_OBSERVED", False))
        self.recon_mask_weight = float(getattr(config, "RECON_MASK_WEIGHT", 1.0))
        self.free_bits_threshold = float(getattr(config, "FREE_BITS_THRESHOLD", 0.0))
        self.gamma_causal = float(getattr(config, "GAMMA_CAUSAL", 1.0))
        self.dag_loss_weight = float(getattr(config, "DAG_LOSS_WEIGHT", 1.0))
        self.huge_loss_threshold = float(getattr(config, "HUGE_LOSS_THRESHOLD", 100.0))
        self.grad_clip_norm = float(getattr(config, "GRAD_CLIP_NORM", 1.0))
        self.causal_ignore_diagonal = bool(getattr(config, "CAUSAL_IGNORE_DIAGONAL", True))
        self.causal_sup_loss = str(getattr(config, "CAUSAL_SUP_LOSS", "mse")).lower()
        self.causal_target_value = float(getattr(config, "CAUSAL_TARGET_VALUE", 0.0))
        self.max_skipped_batches = int(getattr(config, "MAX_SKIPPED_BATCHES_PER_EPOCH", 1000000))

        self.lambda_latent_corr = float(getattr(config, "LAMBDA_LATENT_CORR", 0.0))
        self.lambda_latent_var = float(getattr(config, "LAMBDA_LATENT_VAR", 0.0))
        self.latent_var_target = float(getattr(config, "LATENT_VAR_TARGET", 0.8))
        self.latent_reg_start_epoch = int(getattr(config, "LATENT_REG_START_EPOCH", 0))

        self.dag_warmup_epochs = int(getattr(config, "DAG_WARMUP_EPOCHS", 20))
        self.dag_loss_every_steps = max(1, int(getattr(config, "DAG_LOSS_EVERY_STEPS", 4)))
        self.dag_use_batch_mean = bool(getattr(config, "DAG_USE_BATCH_MEAN", True))
        self.dag_disable_in_val = bool(getattr(config, "DAG_DISABLE_IN_VAL", True))

        self.logvar_min = float(getattr(config, "LOGVAR_MIN", -10.0))
        self.logvar_max = float(getattr(config, "LOGVAR_MAX", 10.0))
        self.a_prob_clamp_min = float(getattr(config, "A_PROB_CLAMP_MIN", 0.0))
        self.a_prob_clamp_max = float(getattr(config, "A_PROB_CLAMP_MAX", 0.60))
        self.use_safe_dag = bool(getattr(config, "USE_SAFE_DAG_LOSS", True))
        self.max_dag_value = float(getattr(config, "MAX_DAG_VALUE", 1e4))

        self.debug_causal_print = bool(getattr(config, "DEBUG_CAUSAL_PRINT", False))
        self.debug_causal_print_every = int(getattr(config, "DEBUG_CAUSAL_PRINT_EVERY", 200))
        self.debug_causal_print_on_val_only = bool(getattr(config, "DEBUG_CAUSAL_PRINT_ON_VAL_ONLY", True))
        self.debug_causal_print_stats = bool(getattr(config, "DEBUG_CAUSAL_PRINT_STATS", False))

        self.trace_enable = bool(getattr(config, "TRACE_ENABLE", False))
        self.trace_stall_epoch = int(getattr(config, "TRACE_STALL_EPOCH", 21))
        self.trace_first_batch_only = bool(getattr(config, "TRACE_FIRST_BATCH_ONLY", True))
        self.trace_force_cuda_sync = bool(getattr(config, "TRACE_FORCE_CUDA_SYNC", False))

        self.use_amp = bool(getattr(config, "USE_AMP", torch.cuda.is_available() and device.type == "cuda"))
        amp_dtype_name = str(getattr(config, "AMP_DTYPE", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        adam_kwargs = {"lr": self.lr}
        use_fused = bool(getattr(config, "USE_FUSED_ADAM", True)) and self.device.type == "cuda"
        if use_fused:
            try:
                self.optimizer = optim.Adam(self.model.parameters(), fused=True, **adam_kwargs)
            except TypeError:
                self.optimizer = optim.Adam(self.model.parameters(), **adam_kwargs)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), **adam_kwargs)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=self.is_main,
        )

        self.best_metric = float("inf")
        self.causal_target_tensor_cpu: Optional[torch.Tensor] = None
        self._cached_target_dev: Optional[torch.Tensor] = None
        self._cached_eye: Dict[tuple[str, int, str], torch.Tensor] = {}
        self._cached_offdiag: Dict[tuple[str, int], torch.Tensor] = {}
        self._load_causal_target_if_needed()

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = _RunningAverages(EXPECTED_METRIC_KEYS)
        skipped_batches = 0

        if self.is_main:
            print(f"[TRAIN-ENTER] epoch={epoch + 1}", flush=True)

        it = iter(dataloader)
        step = 0
        while True:
            t_fetch = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            step += 1

            trace = self._should_trace(epoch=epoch, step=step)
            if trace:
                self._trace(
                    f"epoch={epoch + 1} step={step} batch fetched | "
                    f"fetch_time={time.perf_counter() - t_fetch:.4f}s"
                )

            if len(batch) == 3:
                x, mask, idx = batch
            else:
                x, mask = batch
                idx = None

            loss_tensors = self._run_batch(
                batch=x,
                mask=mask,
                batch_idx=idx,
                is_training=True,
                current_epoch=epoch,
                step=step,
                trace=trace,
            )
            loss_dict = self._to_float_dict(loss_tensors)
            running.update(loss_dict)

            if not np.isfinite(loss_dict.get("total_loss", np.nan)):
                skipped_batches += 1
                if skipped_batches >= self.max_skipped_batches:
                    if self.is_main:
                        print(f"[WARN] skipped_batches >= {self.max_skipped_batches}, stop epoch early", flush=True)
                    break

        avg_losses = self._sync_epoch_means(running.means())
        if self.is_main:
            self._print_epoch_summary("TRAIN", epoch, avg_losses, skipped_batches=skipped_batches)
        return avg_losses

    def validate(self, dataloader, epoch: int, scheduler_metric: str = "recon_loss") -> Dict[str, float]:
        self.model.eval()
        running = _RunningAverages(EXPECTED_METRIC_KEYS)

        if self.is_main:
            print(f"[VAL-ENTER] epoch={epoch + 1}", flush=True)

        with torch.inference_mode():
            it = iter(dataloader)
            step = 0
            while True:
                t_fetch = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                step += 1

                trace = self._should_trace(epoch=epoch, step=step)
                if trace:
                    self._trace(
                        f"[VAL] epoch={epoch + 1} step={step} batch fetched | "
                        f"fetch_time={time.perf_counter() - t_fetch:.4f}s"
                    )

                if len(batch) == 3:
                    x, mask, idx = batch
                else:
                    x, mask = batch
                    idx = None

                loss_tensors = self._run_batch(
                    batch=x,
                    mask=mask,
                    batch_idx=idx,
                    is_training=False,
                    current_epoch=epoch,
                    step=step,
                    trace=trace,
                )
                loss_dict = self._to_float_dict(loss_tensors)
                running.update(loss_dict)

        avg_losses = self._sync_epoch_means(running.means())
        metric = float(avg_losses.get(scheduler_metric, avg_losses.get("total_loss", np.nan)))
        if np.isfinite(metric):
            self.scheduler.step(metric)
        elif self.is_main:
            print(f"[WARN] scheduler metric invalid at epoch={epoch + 1}: {metric}", flush=True)

        if self.is_main:
            self._print_epoch_summary("VAL", epoch, avg_losses)
        return avg_losses

    def save_checkpoint(self, path: str, epoch: int, best_metric: float) -> None:
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        payload = {
            "epoch": int(epoch),
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": float(best_metric),
            "config": getattr(self.config, "__dict__", {}),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_checkpoint(self, path: str, reset_optimizer: bool = True) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        state = payload["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)

        if not reset_optimizer:
            if "optimizer_state_dict" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "scheduler_state_dict" in payload:
                self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        self.best_metric = float(payload.get("best_metric", float("inf")))

    def maybe_resume(self, ckpt_path: str, resume: bool, reset_optimizer: bool = False):
        if resume and os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                payload = pickle.load(f)
            state = payload["model_state_dict"]
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)

            if not reset_optimizer:
                if "optimizer_state_dict" in payload:
                    self.optimizer.load_state_dict(payload["optimizer_state_dict"])
                if "scheduler_state_dict" in payload:
                    self.scheduler.load_state_dict(payload["scheduler_state_dict"])

            start_epoch = int(payload.get("epoch", -1)) + 1
            best_metric = float(payload.get("best_metric", float("inf")))
            self.best_metric = best_metric
            return start_epoch, best_metric
        return 0, float("inf")

    def _run_batch(
        self,
        batch,
        mask,
        batch_idx,
        is_training: bool,
        current_epoch: int,
        step: int,
        trace: bool = False,
    ):
        if trace:
            self._trace("_run_batch start")

        t = time.perf_counter()
        inputs_clean = batch.to(self.device, non_blocking=True).float()
        mask = mask.to(self.device, non_blocking=True).float()
        self._sync_cuda()
        if trace:
            self._trace(f"after to(device) | dt={time.perf_counter() - t:.4f}s")

        if is_training and self.input_noise_std > 0:
            noise = torch.randn_like(inputs_clean) * self.input_noise_std
            inputs_noisy = inputs_clean + noise * mask
        else:
            inputs_noisy = inputs_clean

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)
            if self.use_amp and self.device.type == "cuda"
            else nullcontext()
        )

        t = time.perf_counter()
        with autocast_ctx:
            recon_data, mask_logits, mu, logvar, A_logits = self.model(inputs_noisy)
            logvar = logvar.clamp(self.logvar_min, self.logvar_max)
            loss_tensors = self._compute_loss_tensors(
                inputs=inputs_clean,
                mask=mask,
                recon_data=recon_data,
                mask_logits=mask_logits,
                mu=mu,
                logvar=logvar,
                A_logits=A_logits,
                current_epoch=current_epoch,
                step=step,
                is_training=is_training,
                trace=trace,
            )
        self._sync_cuda()
        if trace:
            self._trace(f"after forward+loss | dt={time.perf_counter() - t:.4f}s")

        total_loss = loss_tensors["total_loss"]

        if not torch.isfinite(total_loss):
            if self.is_main:
                print(
                    f"[WARN] non-finite loss at epoch={current_epoch + 1}, "
                    f"step={step}, idx={self._idx_to_print(batch_idx)}",
                    flush=True,
                )
            out = {k: torch.tensor(float("nan"), device=self.device) for k in EXPECTED_METRIC_KEYS}
            out["current_beta"] = loss_tensors["current_beta"]
            return out

        if is_training:
            t = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.device.type == "cuda":
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
            self._sync_cuda()
            if trace:
                self._trace(f"after backward+step | dt={time.perf_counter() - t:.4f}s")

        if trace:
            self._trace("_run_batch end")
        return loss_tensors

    def _compute_loss_tensors(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        recon_data: torch.Tensor,
        mask_logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        A_logits: torch.Tensor,
        current_epoch: int,
        step: int,
        is_training: bool,
        trace: bool = False,
    ):
        if self.mask_loss_only_on_observed:
            denom = mask.sum().clamp_min(1.0)
            recon_data_loss = (((recon_data - inputs) ** 2) * mask).sum() / denom
        else:
            recon_data_loss = F.mse_loss(recon_data, inputs)

        recon_mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask)
        recon_loss = recon_data_loss + self.recon_mask_weight * recon_mask_loss

        kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        reduce_dims = tuple(range(kld_per_dim.ndim - 1))
        kld_dim_mean = kld_per_dim.mean(dim=reduce_dims)
        kld_dim_free = torch.clamp(kld_dim_mean, min=self.free_bits_threshold) if self.free_bits_threshold > 0 else kld_dim_mean
        kld_loss = kld_dim_mean.sum()
        kld_free = kld_dim_free.sum()
        current_beta = torch.tensor(self._get_current_beta(current_epoch), device=self.device, dtype=recon_loss.dtype)

        apply_latent_reg = (
            (current_epoch + 1) >= self.latent_reg_start_epoch
            and (self.lambda_latent_corr > 0.0 or self.lambda_latent_var > 0.0)
        )
        if apply_latent_reg:
            latent_corr_loss, latent_var_loss = self._compute_latent_reg_loss(mu)
        else:
            latent_corr_loss = torch.zeros((), device=mu.device, dtype=mu.dtype)
            latent_var_loss = torch.zeros((), device=mu.device, dtype=mu.dtype)

        A_prob = torch.sigmoid(A_logits)
        A_prob = self._zero_diag(A_prob)
        A_prob = torch.clamp(A_prob, min=self.a_prob_clamp_min, max=self.a_prob_clamp_max)

        struct_loss = self._compute_struct_loss(A_prob)

        dag_allowed = (not self.dag_disable_in_val) or is_training
        compute_dag = (
            dag_allowed
            and (current_epoch + 1) > self.dag_warmup_epochs
            and (step % self.dag_loss_every_steps == 0)
            and self.dag_loss_weight > 0
        )
        if trace:
            self._trace(f"compute_dag={compute_dag}")

        if compute_dag:
            dag_input = A_prob.mean(dim=0, keepdim=True) if self.dag_use_batch_mean else A_prob
            t = time.perf_counter()
            dag_loss = self._compute_dag_loss(dag_input)
            self._sync_cuda()
            if trace:
                self._trace(f"after dag_loss | dt={time.perf_counter() - t:.4f}s")
        else:
            dag_loss = torch.zeros((), device=A_prob.device, dtype=A_prob.dtype)

        if not torch.isfinite(struct_loss):
            struct_loss = torch.zeros((), device=A_prob.device, dtype=A_prob.dtype)
        if not torch.isfinite(dag_loss):
            dag_loss = torch.zeros((), device=A_prob.device, dtype=A_prob.dtype)

        causal_loss = struct_loss + self.dag_loss_weight * dag_loss
        total_loss = (
            recon_loss
            + current_beta * kld_free
            + self.gamma_causal * causal_loss
            + self.lambda_latent_corr * latent_corr_loss
            + self.lambda_latent_var * latent_var_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "recon_data_loss": recon_data_loss,
            "recon_mask_loss": recon_mask_loss,
            "kld_loss": kld_loss,
            "kld_free": kld_free,
            "struct_loss": struct_loss,
            "dag_loss": dag_loss,
            "causal_loss": causal_loss,
            "latent_corr_loss": latent_corr_loss,
            "latent_var_loss": latent_var_loss,
            "current_beta": current_beta,
        }

    def _compute_latent_reg_loss(self, mu: torch.Tensor, eps: float = 1e-6):
        B, D = mu.shape
        if B < 2:
            zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
            return zero, zero

        mu_centered = mu - mu.mean(dim=0, keepdim=True)
        var = mu_centered.pow(2).mean(dim=0)
        std = torch.sqrt(var + eps)

        mu_norm = mu_centered / std.unsqueeze(0)
        corr = (mu_norm.T @ mu_norm) / float(B)

        offdiag_mask = ~torch.eye(D, dtype=torch.bool, device=mu.device)
        corr_offdiag = corr[offdiag_mask]
        latent_corr_loss = (corr_offdiag ** 2).mean()

        latent_var_loss = torch.relu(self.latent_var_target - std).pow(2).mean()

        return latent_corr_loss, latent_var_loss

    def _compute_struct_loss(self, A_prob: torch.Tensor) -> torch.Tensor:
        target = self._get_causal_target_tensor(A_prob)
        vec_pred = self._offdiag_view(A_prob) if self.causal_ignore_diagonal else A_prob
        vec_target = self._offdiag_view(target) if self.causal_ignore_diagonal else target
        return F.l1_loss(vec_pred, vec_target) if self.causal_sup_loss == "l1" else F.mse_loss(vec_pred, vec_target)

    def _compute_dag_loss(self, A_prob: torch.Tensor) -> torch.Tensor:
        d = A_prob.shape[-1]
        A_sq = (A_prob * A_prob).float() if self.use_safe_dag else (A_prob * A_prob)
        h_tensor = torch.diagonal(torch.matrix_exp(A_sq), dim1=-2, dim2=-1).sum(dim=-1) - float(d)
        dag_loss = torch.mean(h_tensor ** 2)
        if self.use_safe_dag:
            dag_loss = torch.clamp(dag_loss, max=self.max_dag_value).to(A_prob.dtype)
        return dag_loss

    def _get_causal_target_tensor(self, A_prob: torch.Tensor) -> torch.Tensor:
        if self.causal_target_tensor_cpu is None:
            return self._zero_diag(torch.full_like(A_prob, self.causal_target_value))

        if (
            self._cached_target_dev is None
            or self._cached_target_dev.device != A_prob.device
            or self._cached_target_dev.dtype != A_prob.dtype
        ):
            self._cached_target_dev = self.causal_target_tensor_cpu.to(A_prob.device, dtype=A_prob.dtype)

        target = self._cached_target_dev
        if target.ndim != 2:
            raise ValueError(f"A_target must be 2D, got shape={tuple(target.shape)}")
        if target.shape != A_prob.shape[-2:]:
            raise ValueError(
                f"A_target shape mismatch: target={tuple(target.shape)} vs model={tuple(A_prob.shape[-2:])}. "
                f"请确保 causal_target_builder 输出的是 feature-level 图，且节点数等于 num_total_features。"
            )
        return self._zero_diag(target.unsqueeze(0).expand(A_prob.shape[0], -1, -1))

    def _load_causal_target_if_needed(self) -> None:
        path = getattr(self.config, "CAUSAL_TARGET_PATH", None)
        if not path or not os.path.exists(path):
            self.causal_target_tensor_cpu = None
            return

        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = getattr(self.config, "CAUSAL_TARGET_KEY", "A_target")
            if key not in arr:
                raise KeyError(f"CAUSAL_TARGET_KEY={key} not found in {path}")
            arr = arr[key]

        arr = np.asarray(arr, dtype=np.float32)
        if bool(getattr(self.config, "CAUSAL_TARGET_CLAMP_01", True)):
            arr = np.clip(arr, 0.0, 1.0)
        self.causal_target_tensor_cpu = torch.from_numpy(arr)

    def _get_current_beta(self, current_epoch: int) -> float:
        beta_final = float(getattr(self.config, "BETA_KL_FINAL", 0.0))
        warmup = int(getattr(self.config, "KL_ANNEALING_WARMUP_EPOCHS", 0))
        ramp = max(1, int(getattr(self.config, "KL_ANNEALING_RAMP_EPOCHS", 1)))
        use_cyclical = bool(getattr(self.config, "USE_CYCLICAL_ANNEALING", False))
        cycle_epochs = max(1, int(getattr(self.config, "KL_ANNEALING_CYCLE_EPOCHS", ramp)))
        epoch_1based = current_epoch + 1
        if epoch_1based <= warmup:
            return 0.0
        if use_cyclical:
            pos = (epoch_1based - warmup - 1) % cycle_epochs + 1
            return beta_final * min(1.0, pos / ramp)
        progress = epoch_1based - warmup
        return beta_final * min(1.0, progress / ramp)

    def _sync_epoch_means(self, means: Dict[str, float]) -> Dict[str, float]:
        if not self.is_distributed:
            return means

        keys = list(EXPECTED_METRIC_KEYS)
        sum_tensor = torch.tensor([means.get(k, float("nan")) for k in keys], device=self.device, dtype=torch.float64)
        cnt_tensor = torch.tensor(
            [1.0 if np.isfinite(means.get(k, np.nan)) else 0.0 for k in keys],
            device=self.device,
            dtype=torch.float64,
        )
        valid_mask = torch.isfinite(sum_tensor)
        sum_tensor = torch.where(valid_mask, sum_tensor, torch.zeros_like(sum_tensor))
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)

        out = {}
        for i, k in enumerate(keys):
            out[k] = float((sum_tensor[i] / cnt_tensor[i]).item()) if cnt_tensor[i].item() > 0 else float("nan")
        return out

    def _to_float_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out = {}
        for k in EXPECTED_METRIC_KEYS:
            v = d.get(k, None)
            out[k] = float("nan") if v is None else (float(v.detach().float().cpu().item()) if torch.is_tensor(v) else float(v))
        return out

    def _offdiag_view(self, A: torch.Tensor) -> torch.Tensor:
        _, d, _ = A.shape
        key = (str(A.device), d)
        if key not in self._cached_offdiag:
            self._cached_offdiag[key] = ~torch.eye(d, dtype=torch.bool, device=A.device)
        return A[:, self._cached_offdiag[key]]

    def _zero_diag(self, A: torch.Tensor) -> torch.Tensor:
        _, d, _ = A.shape
        key = (str(A.device), d, str(A.dtype))
        if key not in self._cached_eye:
            self._cached_eye[key] = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0)
        return A * (1.0 - self._cached_eye[key])

    def _should_trace(self, epoch: int, step: int) -> bool:
        if not self.trace_enable:
            return False
        if (epoch + 1) != self.trace_stall_epoch:
            return False
        if self.trace_first_batch_only and step != 1:
            return False
        return True

    def _sync_cuda(self) -> None:
        if self.trace_force_cuda_sync and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _trace(self, msg: str) -> None:
        if self.is_main:
            print(f"[TRACE] {msg}", flush=True)

    def _print_epoch_summary(self, phase: str, epoch: int, avg_losses: Dict[str, float], skipped_batches: int | None = None) -> None:
        msg = (
            f"[{phase}] Epoch {epoch + 1} | "
            f"total={self._fmt(avg_losses.get('total_loss', np.nan), '.6f')} | "
            f"recon={self._fmt(avg_losses.get('recon_loss', np.nan), '.6f')} | "
            f"recon_data={self._fmt(avg_losses.get('recon_data_loss', np.nan), '.6f')} | "
            f"recon_mask={self._fmt(avg_losses.get('recon_mask_loss', np.nan), '.6f')} | "
            f"kld={self._fmt(avg_losses.get('kld_loss', np.nan), '.6f')} | "
            f"kld_free={self._fmt(avg_losses.get('kld_free', np.nan), '.6f')} | "
            f"struct={self._fmt(avg_losses.get('struct_loss', np.nan), '.8e')} | "
            f"dag={self._fmt(avg_losses.get('dag_loss', np.nan), '.8e')} | "
            f"causal={self._fmt(avg_losses.get('causal_loss', np.nan), '.8e')} | "
            f"latent_corr={self._fmt(avg_losses.get('latent_corr_loss', np.nan), '.8e')} | "
            f"latent_var={self._fmt(avg_losses.get('latent_var_loss', np.nan), '.8e')} | "
            f"beta={self._fmt(avg_losses.get('current_beta', np.nan), '.6f')}"
        )
        if skipped_batches is not None:
            msg += f" | skipped={skipped_batches}"
        print(msg, flush=True)

    @staticmethod
    def _idx_to_print(idx) -> Any:
        if idx is None:
            return None
        return idx.detach().cpu().tolist() if torch.is_tensor(idx) else idx

    @staticmethod
    def _fmt(value: float, fmt: str) -> str:
        return format(value, fmt) if np.isfinite(value) else "nan"