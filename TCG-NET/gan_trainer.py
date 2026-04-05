import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from opacus import PrivacyEngine
except Exception:
    PrivacyEngine = None

from gan_models import Generator, Discriminator, MappingNetwork
from gan_config import GANConfig
from gan_data import create_dataloader


class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4, monitor="g_loss", mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def _is_improved(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta

    def step(self, metrics: Dict[str, float]) -> bool:
        current_value = metrics.get(self.monitor, None)
        if current_value is None or not np.isfinite(current_value):
            return False

        if self._is_improved(current_value):
            self.best_value = float(current_value)
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            print(
                f"[EarlyStopping] monitor={self.monitor} "
                f"mode={self.mode} patience={self.patience} reached. stop."
            )
            return True
        return False


class GANTrainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        mapping_network: MappingNetwork,
        config: GANConfig,
        dataset: torch.utils.data.Dataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if dataset is None:
            raise ValueError("GANTrainer 需要一个 Dataset 对象")

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.mapping_network = mapping_network.to(device)
        self.config = config
        self.device = device
        self.epoch = 0
        self.dp_enabled = False
        self._cached_real: Optional[torch.Tensor] = None

        self.checkpoint_dir = Path(getattr(self.config, "CHECKPOINT_DIR", "./cache_gan/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        self.best_metric_value = -float("inf") if getattr(self.config, "BEST_MODE", "min") == "max" else float("inf")

        self.g_optimizer = optim.Adam(
            list(self.generator.parameters()) + list(self.mapping_network.parameters()),
            lr=config.LEARNING_RATE_G,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        )

        if getattr(config, "USE_SCHEDULER", True):
            self.g_scheduler = CosineAnnealingLR(
                self.g_optimizer, T_max=max(1, config.NUM_EPOCHS), eta_min=config.LR_MIN
            )
            self.d_scheduler = CosineAnnealingLR(
                self.d_optimizer, T_max=max(1, config.NUM_EPOCHS), eta_min=config.LR_MIN
            )
        else:
            self.g_scheduler = None
            self.d_scheduler = None

        dataloader = create_dataloader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        if getattr(config, "DP_ENABLE", False):
            if PrivacyEngine is None:
                print("[DP] opacus 不可用，回退到非 DP 模式")
                self.dataloader = dataloader
            else:
                try:
                    self.privacy_engine = PrivacyEngine()
                    self.discriminator, self.d_optimizer, self.dataloader = self.privacy_engine.make_private(
                        module=self.discriminator,
                        optimizer=self.d_optimizer,
                        data_loader=dataloader,
                        noise_multiplier=config.DP_NOISE_MULTIPLIER,
                        max_grad_norm=config.DP_MAX_GRAD_NORM,
                    )
                    self.dp_enabled = True
                    print("[DP] 已启用差分隐私")
                except Exception as e:
                    print(f"[DP] 启用失败：{e}，回退到非 DP 模式")
                    self.dataloader = dataloader
        else:
            self.dataloader = dataloader

    def get_privacy_spent(self):
        if self.dp_enabled:
            return self.privacy_engine.get_epsilon(self.config.DP_TARGET_DELTA)
        return None

    def _set_requires_grad(self, module: torch.nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    def _instance_noise_std(self) -> float:
        s0 = self.config.INSTANCE_NOISE_STD_INIT
        s1 = self.config.INSTANCE_NOISE_STD_FINAL
        t = min(1.0, self.epoch / max(1, self.config.NUM_EPOCHS - 1))
        return s0 * (1.0 - t) + s1 * t

    def _apply_instance_noise(self, x: torch.Tensor) -> torch.Tensor:
        std = self._instance_noise_std()
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std

    def _maybe_mixup(self, real: torch.Tensor, fake: torch.Tensor):
        if np.random.rand() < self.config.MIXUP_PROB:
            lam = np.random.beta(self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA)
            mixed_real = lam * real + (1 - lam) * fake
            mixed_fake = lam * fake + (1 - lam) * real
            return mixed_real, mixed_fake
        return real, fake

    def _batch_moments(self, x: torch.Tensor):
        mean = x.mean(dim=0)
        xc = x - mean
        var = (xc ** 2).mean(dim=0) + 1e-8
        std = torch.sqrt(var)
        skew = (xc ** 3).mean(dim=0) / (std ** 3)
        kurt = (xc ** 4).mean(dim=0) / (var ** 2)
        return mean, var, skew, kurt

    def _moment_matching_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        orders = set(getattr(self.config, "MOMENT_ORDERS", [1, 2, 3, 4]))
        rm, rv, rs, rk = self._batch_moments(real)
        fm, fv, fs, fk = self._batch_moments(fake)

        eps = 1e-6
        loss = torch.tensor(0.0, device=real.device)

        if 1 in orders:
            loss = loss + ((rm - fm) ** 2 / (rv + eps)).mean()
        if 2 in orders:
            loss = loss + ((rv - fv) ** 2 / (rv ** 2 + eps)).mean()
        if 3 in orders:
            loss = loss + ((rs - fs) ** 2).mean()
        if 4 in orders:
            loss = loss + ((rk - fk) ** 2).mean()
        return loss

    def _gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        critic = self.discriminator._module if self.dp_enabled else self.discriminator
        d_interpolates = critic(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    def _disc_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.discriminator, "extract_features"):
            return self.discriminator.extract_features(x)
        return self.discriminator(x)

    def _build_checkpoint_payload(self, epoch: int, metrics: Optional[Dict] = None, extra: Optional[Dict] = None):
        payload = {
            "epoch": int(epoch),
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "mapping_network_state_dict": self.mapping_network.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "config": self.config.__dict__,
            "best_metric_name": getattr(self.config, "BEST_METRIC", "g_loss"),
            "best_metric_mode": getattr(self.config, "BEST_MODE", "min"),
            "best_metric_value": self.best_metric_value,
        }
        if self.g_scheduler is not None:
            payload["g_scheduler_state_dict"] = self.g_scheduler.state_dict()
        if self.d_scheduler is not None:
            payload["d_scheduler_state_dict"] = self.d_scheduler.state_dict()
        if metrics is not None:
            payload["metrics"] = metrics
        if extra is not None:
            payload["extra"] = extra
        return payload

    def _save_checkpoint(self, path: str, epoch: int, metrics: Optional[Dict] = None, extra: Optional[Dict] = None):
        payload = self._build_checkpoint_payload(epoch=epoch, metrics=metrics, extra=extra)
        torch.save(payload, path)
        print(f"[GAN-CKPT] saved -> {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.mapping_network.load_state_dict(checkpoint["mapping_network_state_dict"])
        if not getattr(self.config, "RESET_OPTIMIZER_ON_RESUME", False):
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        if self.g_scheduler is not None and "g_scheduler_state_dict" in checkpoint:
            self.g_scheduler.load_state_dict(checkpoint["g_scheduler_state_dict"])
        if self.d_scheduler is not None and "d_scheduler_state_dict" in checkpoint:
            self.d_scheduler.load_state_dict(checkpoint["d_scheduler_state_dict"])

        self.best_metric_value = checkpoint.get("best_metric_value", self.best_metric_value)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"[GAN-CKPT] resumed from {path}, next epoch={self.start_epoch}")

    def maybe_resume(self):
        if getattr(self.config, "RESUME_TRAINING", False) and getattr(self.config, "RESUME_PATH", ""):
            self.load_model(self.config.RESUME_PATH)

    def is_better_metric(self, metric_value: float) -> bool:
        return metric_value > self.best_metric_value if getattr(self.config, "BEST_MODE", "min") == "max" else metric_value < self.best_metric_value

    def maybe_save_training_checkpoints(self, epoch: int, metrics: Dict[str, float]):
        if getattr(self.config, "SAVE_LAST_EVERY_EPOCH", True):
            self._save_checkpoint(
                path=getattr(self.config, "LAST_MODEL_SAVE_PATH", str(self.checkpoint_dir / "gan_last_model.pt")),
                epoch=epoch,
                metrics=metrics,
                extra={"kind": "last"},
            )

        save_every_n = int(getattr(self.config, "SAVE_EVERY_N_EPOCHS", 0))
        if save_every_n > 0 and (epoch + 1) % save_every_n == 0:
            periodic_path = str(self.checkpoint_dir / f"gan_epoch_{epoch + 1:03d}.pt")
            self._save_checkpoint(periodic_path, epoch, metrics, extra={"kind": "periodic"})

        if getattr(self.config, "SAVE_BEST", True):
            metric_name = getattr(self.config, "BEST_METRIC", "g_loss")
            metric_value = metrics.get(metric_name, None)
            if metric_value is not None and np.isfinite(metric_value) and self.is_better_metric(float(metric_value)):
                self.best_metric_value = float(metric_value)
                self._save_checkpoint(
                    path=getattr(self.config, "MODEL_SAVE_PATH", str(self.checkpoint_dir / "gan_best_model.pt")),
                    epoch=epoch,
                    metrics=metrics,
                    extra={"kind": "best", "monitor": metric_name, "value": self.best_metric_value},
                )
                print(f"[GAN-BEST] epoch={epoch + 1} | {metric_name}={self.best_metric_value:.6f}")

    def train_discriminator(self, real_data: torch.Tensor) -> Dict[str, float]:
        self._set_requires_grad(self.discriminator, True)
        self._set_requires_grad(self.generator, False)
        self._set_requires_grad(self.mapping_network, False)

        self.d_optimizer.zero_grad(set_to_none=True)

        batch_size = real_data.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, self.config.NOISE_DIM, device=self.device)
            w = self.mapping_network(noise)
            fake_data = self.generator(w)

        real_aug = self._apply_instance_noise(real_data)
        fake_aug = self._apply_instance_noise(fake_data)
        real_aug, fake_aug = self._maybe_mixup(real_aug, fake_aug)

        d_real = self.discriminator(real_aug)
        d_fake = self.discriminator(fake_aug.detach())

        wasserstein = torch.mean(d_real) - torch.mean(d_fake)
        d_loss = -wasserstein

        if self.dp_enabled:
            gp = torch.tensor(0.0, device=self.device)
            d_total = d_loss
        else:
            gp = self._gradient_penalty(real_data, fake_data)
            d_total = d_loss + self.config.GRAD_PENALTY_WEIGHT * gp

        d_total.backward()
        self.d_optimizer.step()

        self._cached_real = real_data.detach()
        return {
            "d_loss": float(d_total.item()),
            "wasserstein_dist": float(wasserstein.item()),
            "gradient_penalty": float(gp.item()),
        }

    def train_generator(self) -> Dict[str, float]:
        if self._cached_real is None:
            return {
                "g_loss": float("nan"),
                "g_loss_adv": float("nan"),
                "g_loss_div": float("nan"),
                "g_loss_fm": float("nan"),
                "g_loss_mm": float("nan"),
            }

        self._set_requires_grad(self.discriminator, False)
        self._set_requires_grad(self.generator, True)
        self._set_requires_grad(self.mapping_network, True)
        self.g_optimizer.zero_grad(set_to_none=True)

        real_ref = self._cached_real
        B = real_ref.size(0)

        noise = torch.randn(B, self.config.NOISE_DIM, device=self.device)
        w = self.mapping_network(noise)
        fake = self.generator(w)

        g_loss_adv = -torch.mean(self.discriminator(fake))

        with torch.no_grad():
            noise2 = torch.randn(B, self.config.NOISE_DIM, device=self.device)
            w2 = self.mapping_network(noise2)
            fake2 = self.generator(w2)

        dist_z = torch.pdist(noise.view(B, -1), p=1)
        dist_g = torch.pdist(fake.view(B, -1), p=1)
        loss_div = -torch.mean(dist_g) / (torch.mean(dist_z) + 1e-8)

        if real_ref.size(0) >= B:
            real_ref_aligned = real_ref[:B]
        else:
            rep = math.ceil(B / real_ref.size(0))
            real_ref_aligned = real_ref.repeat(rep, 1)[:B]

        real_feat = self._disc_features(real_ref_aligned).mean(dim=0)
        fake_feat = self._disc_features(fake).mean(dim=0)
        fm_loss = (real_feat - fake_feat).pow(2).mean()
        mm_loss = self._moment_matching_loss(real_ref_aligned, fake)

        g_total = (
            g_loss_adv
            + self.config.DIVERSITY_LAMBDA * loss_div
            + self.config.FM_LOSS_WEIGHT * fm_loss
            + self.config.MOMENT_MATCHING_WEIGHT * mm_loss
        )

        g_total.backward()
        self.g_optimizer.step()

        return {
            "g_loss": float(g_total.item()),
            "g_loss_adv": float(g_loss_adv.item()),
            "g_loss_div": float(loss_div.item()),
            "g_loss_fm": float(fm_loss.item()),
            "g_loss_mm": float(mm_loss.item()),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.epoch = epoch
        self.generator.train()
        self.discriminator.train()
        self.mapping_network.train()

        d_losses, g_losses, w_dists, gp_vals = [], [], [], []
        for i, batch in enumerate(self.dataloader):
            real_data = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            d_out = self.train_discriminator(real_data)
            d_losses.append(d_out["d_loss"])
            w_dists.append(d_out["wasserstein_dist"])
            gp_vals.append(d_out["gradient_penalty"])

            if (i + 1) % self.config.N_CRITIC == 0:
                g_out = self.train_generator()
                g_losses.append(g_out["g_loss"])

        if self.g_scheduler is not None:
            self.g_scheduler.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()

        return {
            "d_loss": float(np.mean(d_losses)) if d_losses else float("nan"),
            "g_loss": float(np.mean(g_losses)) if g_losses else float("nan"),
            "wasserstein_dist": float(np.mean(w_dists)) if w_dists else float("nan"),
            "gradient_penalty": float(np.mean(gp_vals)) if gp_vals else float("nan"),
            "lr_g": float(self.g_optimizer.param_groups[0]["lr"]),
            "lr_d": float(self.d_optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> np.ndarray:
        self.generator.eval()
        self.mapping_network.eval()
        synthetic_samples = []
        generated_count = 0

        while generated_count < num_samples:
            batch_size = min(self.config.BATCH_SIZE, num_samples - generated_count)
            noise = torch.randn(batch_size, self.config.NOISE_DIM, device=self.device)
            w = self.mapping_network(noise)
            fake = self.generator(w)
            synthetic_samples.append(fake.cpu().numpy())
            generated_count += batch_size
        return np.concatenate(synthetic_samples, axis=0)
