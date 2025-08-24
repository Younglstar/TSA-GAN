# file: gan_trainer.py (最终完整版)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from opacus import PrivacyEngine
from tqdm import tqdm
from gan_models import Generator, Discriminator, MappingNetwork
from gan_config import GANConfig

class GANTrainer:
    def __init__(
            self,
            generator: Generator,
            discriminator: Discriminator,
            mapping_network: MappingNetwork,
            config: GANConfig,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.mapping_network = mapping_network.to(device)
        self.config = config
        self.device = device

        # --- 核心修改：使用新的beta参数 ---
        self.g_optimizer = optim.Adam(
            generator.parameters(), lr=config.LEARNING_RATE_G, betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )
        self.d_optimizer = optim.Adam(
            discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )
        self.m_optimizer = optim.Adam(
            mapping_network.parameters(), lr=config.LEARNING_RATE_G, betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )
        # 在 __init__ 末尾，添加如下（保留你原有的优化器）：
        self.epoch = 0  # 用于实例噪声的线性衰减

        # (可选) 差分隐私
        self.dp_enabled = False
        if getattr(config, "DP_ENABLE", False):
            try:

                self.privacy_engine = PrivacyEngine()
                self.discriminator, self.d_optimizer, self.train_dataloader = \
                    self.privacy_engine.make_private_with_epsilon(
                        module=self.discriminator,
                        optimizer=self.d_optimizer,
                        data_loader=None,  # 这里为None时按sigma配置，不走dataloader会退化到 make_private
                        noise_multiplier=config.DP_NOISE_MULTIPLIER,
                        max_grad_norm=config.DP_MAX_GRAD_NORM,
                    )
                self.dp_enabled = True
                print("[DP] Opacus 已启用差分隐私（判别器）")
            except Exception as e:
                print(f"[DP] 未启用差分隐私（{e}），将以非DP模式训练")

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
        # 有概率把 real 和 fake 做一次 convex mix，WGAN里相当于“软标签”
        if np.random.rand() < self.config.MIXUP_PROB:
            lam = np.random.beta(self.config.MIXUP_ALPHA, self.config.MIXUP_ALPHA)
            # 返回扰动后的 (real, fake)
            r = lam * real + (1 - lam) * fake
            f = lam * fake + (1 - lam) * real
            return r, f
        return real, fake

    def _batch_moments(self, x: torch.Tensor):
        # x: (B, F)
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

        loss = 0.0
        if 1 in orders: loss = loss + (rm - fm).pow(2).mean()
        if 2 in orders: loss = loss + (rv - fv).pow(2).mean()
        if 3 in orders: loss = loss + (rs - fs).pow(2).mean()
        if 4 in orders: loss = loss + (rk - fk).pow(2).mean()
        return loss

    def _gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    '''def train_discriminator(self, real_data: torch.Tensor) -> Dict[str, float]:
        self.d_optimizer.zero_grad()
        batch_size = real_data.size(0)
        noise = torch.randn(batch_size, self.config.NOISE_DIM, device=self.device)
        with torch.no_grad():
            w = self.mapping_network(noise)
            fake_data = self.generator(w)

        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data.detach())
        d_loss = torch.mean(d_fake) - torch.mean(d_real)
        gradient_penalty = self._gradient_penalty(real_data, fake_data)
        d_total_loss = d_loss + self.config.GRAD_PENALTY_WEIGHT * gradient_penalty
        d_total_loss.backward()
        self.d_optimizer.step()

        return {'d_loss': d_total_loss.item(), 'wasserstein_dist': -d_loss.item()}'''

    def _augment_timeseries(self, data: torch.Tensor) -> torch.Tensor:
        """
        对一批时序数据应用增强。(已修正，可处理2D和3D数据)
        :param data: 输入张量，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, num_features)
        :return: 增强后的张量，与输入形状相同
        """
        # --- 核心修改：检查数据维度并处理2D数据 ---
        was_2d = data.ndim == 2
        if was_2d:
            # 如果是2D数据 (batch_size, seq_len)，临时增加一个特征维度变为 (batch_size, seq_len, 1)
            data_3d = data.unsqueeze(-1)
        else:
            data_3d = data

        policies = self.config.AUGMENTATION_POLICY.split(',')

        augmented_data = data_3d.clone()

        if 'noise' in policies:
            noise_std = self.config.AUGMENT_NOISE_STD
            noise = torch.randn_like(augmented_data) * noise_std
            augmented_data += noise

        if 'cutout' in policies:
            # 现在 augmented_data 保证是3D的，这行代码不会再报错
            batch_size, seq_len, _ = augmented_data.shape
            cutout_ratio = self.config.AUGMENT_CUTOUT_RATIO

            for i in range(batch_size):
                cutout_len = int(seq_len * cutout_ratio)
                if cutout_len > 0:
                    start_idx = torch.randint(0, seq_len - cutout_len, (1,)).item()
                    augmented_data[i, start_idx: start_idx + cutout_len, :] = 0

        # --- 核心修改：如果原始数据是2D，则恢复其形状 ---
        if was_2d:
            # 将 (batch_size, seq_len, 1) 挤压回 (batch_size, seq_len)
            return augmented_data.squeeze(-1)
        else:
            return augmented_data

    def train_discriminator(self, real_data: torch.Tensor) -> Dict[str, float]:
        self.d_optimizer.zero_grad()
        batch_size = real_data.size(0)

        with torch.no_grad():
            noise = torch.randn(batch_size, self.config.NOISE_DIM, device=self.device)
            w = self.mapping_network(noise)
            fake_data = self.generator(w)

        # 实例噪声
        real_aug = self._apply_instance_noise(real_data)
        fake_aug = self._apply_instance_noise(fake_data)

        # 轻量 mixup（可抑制过拟合，弱化属性推断）
        real_aug, fake_aug = self._maybe_mixup(real_aug, fake_aug)


        # WGAN critic
        d_real = self.discriminator(real_aug)
        d_fake = self.discriminator(fake_aug.detach())
        d_loss = torch.mean(d_fake) - torch.mean(d_real)

        # 梯度惩罚仍在原始数据上计算
        gradient_penalty = self._gradient_penalty(real_data, fake_data)
        d_total_loss = d_loss + self.config.GRAD_PENALTY_WEIGHT * gradient_penalty
        d_total_loss.backward()
        self.d_optimizer.step()

        # 缓存一个最近的real batch，供生成器feature matching使用
        self._cached_real = real_data.detach()

        return {'d_loss': d_total_loss.item(), 'wasserstein_dist': -d_loss.item()}

    def _disc_features(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容不同写法：优先用 extract_features；没有则直接返回判别器输出（退化为0影响很小）
        if hasattr(self.discriminator, "extract_features"):
            return self.discriminator.extract_features(x)
        # 退化：当没有特征时，只能用输出；这样FM loss近似退化，不会报错
        return self.discriminator(x)

    def train_generator(self) -> Dict[str, float]:
        self.g_optimizer.zero_grad()
        self.m_optimizer.zero_grad()

        B = self.config.BATCH_SIZE
        noise_g = torch.randn(B, self.config.NOISE_DIM, device=self.device)
        w_g = self.mapping_network(noise_g)
        fake_g = self.generator(w_g)

        # 对抗损失
        g_loss_adv = -torch.mean(self.discriminator(fake_g))

        # 多样性
        with torch.no_grad():
            noise_g2 = torch.randn(B, self.config.NOISE_DIM, device=self.device)
            w_g2 = self.mapping_network(noise_g2)
            fake_g2 = self.generator(w_g2)
        dist_z = torch.pdist(noise_g.view(B, -1), p=1)
        dist_g = torch.pdist(fake_g.view(B, -1), p=1)
        loss_div = -torch.mean(dist_g) / (torch.mean(dist_z) + 1e-8)

        # Feature Matching（与最近一次real batch对齐）
        fm_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, "_cached_real") and self._cached_real is not None:
            real_ref = self._cached_real
            if real_ref.size(0) >= B:
                real_ref = real_ref[:B]
            else:
                # 不足则重复补齐
                rep = (B + real_ref.size(0) - 1) // real_ref.size(0)
                real_ref = real_ref.repeat(rep, 1)[:B]

            real_feat = self._disc_features(real_ref).mean(dim=0)
            fake_feat = self._disc_features(fake_g).mean(dim=0)
            fm_loss = (real_feat - fake_feat).pow(2).mean()

        # Moment Matching（显式对齐 1–4 阶矩）
        mm_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, "_cached_real") and self._cached_real is not None:
            real_ref_for_mm = real_ref  # 复用上面整理过的 real_ref
            mm_loss = self._moment_matching_loss(real_ref_for_mm, fake_g)

        g_total_loss = (
                g_loss_adv
                + self.config.DIVERSITY_LAMBDA * loss_div
                + self.config.FM_LOSS_WEIGHT * fm_loss
                + self.config.MOMENT_MATCHING_WEIGHT * mm_loss
        )

        g_total_loss.backward()
        self.g_optimizer.step()
        self.m_optimizer.step()

        return {
            'g_loss': g_total_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_div': loss_div.item(),
        }

    # --- 关键：这里定义了 train_epoch 方法 ---
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.epoch = epoch
        self.generator.train()
        self.discriminator.train()
        self.mapping_network.train()
        d_losses, g_losses, w_dists = [], [], []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
        for i, batch in pbar:
            real_data = batch.to(self.device)
            d_loss_dict = self.train_discriminator(real_data)
            d_losses.append(d_loss_dict['d_loss'])
            w_dists.append(d_loss_dict['wasserstein_dist'])
            if (i + 1) % self.config.N_CRITIC == 0:
                g_loss_dict = self.train_generator()
                g_losses.append(g_loss_dict['g_loss'])
                pbar.set_postfix({
                    'D_loss': f"{np.mean(d_losses[-self.config.N_CRITIC:]):.4f}",
                    'G_loss': f"{g_losses[-1]:.4f}",
                    'W_dist': f"{np.mean(w_dists[-self.config.N_CRITIC:]):.4f}"
                })
        return {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses) if g_losses else float('nan'),
            'wasserstein_dist': np.mean(w_dists)
        }
    # --- train_epoch 方法定义结束 ---

    def generate_samples(self, num_samples: int) -> np.ndarray:
        self.generator.eval()
        self.mapping_network.eval()
        synthetic_samples = []
        with torch.no_grad():
            generated_count = 0
            while generated_count < num_samples:
                batch_size = min(self.config.BATCH_SIZE, num_samples - generated_count)
                if batch_size <= 0: break
                noise = torch.randn(batch_size, self.config.NOISE_DIM, device=self.device)
                w = self.mapping_network(noise)
                fake_data = self.generator(w)
                synthetic_samples.append(fake_data.cpu().numpy())
                generated_count += batch_size
        return np.concatenate(synthetic_samples, axis=0)

    def save_model(self, path: str):
        print(f"\n💾 正在将模型保存到 {path}...")
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'mapping_network_state_dict': self.mapping_network.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'm_optimizer_state_dict': self.m_optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        print(f"\n💾 正在从 {path} 加载模型...")
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.mapping_network.load_state_dict(checkpoint['mapping_network_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.m_optimizer.load_state_dict(checkpoint['m_optimizer_state_dict'])