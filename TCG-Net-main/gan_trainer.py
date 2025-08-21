# gan_trainer.py (修改后的版本)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from tqdm import tqdm

from gan_models import Generator, Discriminator, MappingNetwork
from gan_config import GANConfig

class GANTrainer:
    # ... (__init__ 和 _gradient_penalty 方法保持不变) ...
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


    def train_discriminator(self, real_data: torch.Tensor) -> Dict[str, float]:
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

        # --- 核心修改：在优化器更新前进行梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # --- 修改结束 ---

        self.d_optimizer.step()

        return {'d_loss': d_total_loss.item(), 'wasserstein_dist': -d_loss.item()}

    def train_generator(self) -> Dict[str, float]:
        self.g_optimizer.zero_grad()
        self.m_optimizer.zero_grad()
        noise_g = torch.randn(self.config.BATCH_SIZE, self.config.NOISE_DIM, device=self.device)
        w_g = self.mapping_network(noise_g)
        fake_data_g = self.generator(w_g)
        g_loss_adv = -torch.mean(self.discriminator(fake_data_g))
        with torch.no_grad():
            noise_g2 = torch.randn(self.config.BATCH_SIZE, self.config.NOISE_DIM, device=self.device)
            w_g2 = self.mapping_network(noise_g2)
            fake_data_g2 = self.generator(w_g2)
        dist_z = torch.pdist(noise_g.view(noise_g.size(0), -1), p=1)
        dist_g = torch.pdist(fake_data_g.view(fake_data_g.size(0), -1), p=1)
        loss_diversity = -torch.mean(dist_g) / (torch.mean(dist_z) + 1e-8)
        g_total_loss = g_loss_adv + self.config.DIVERSITY_LAMBDA * loss_diversity
        g_total_loss.backward()

        # --- 核心修改：在优化器更新前进行梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)
        # --- 修改结束 ---

        self.g_optimizer.step()
        self.m_optimizer.step()
        return {'g_loss': g_total_loss.item(), 'g_loss_adv': g_loss_adv.item(), 'g_loss_div': loss_diversity.item()}

    # ... (train_epoch 和其他方法保持不变) ...
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        self.mapping_network.train()
        d_losses, g_losses, w_dists = [], [], []
        last_g_loss = float('nan')

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
        for i, batch in pbar:
            real_data = batch.to(self.device)
            d_loss_dict = self.train_discriminator(real_data)
            d_losses.append(d_loss_dict['d_loss'])
            w_dists.append(d_loss_dict['wasserstein_dist'])
            if (i + 1) % self.config.N_CRITIC == 0:
                g_loss_dict = self.train_generator()
                last_g_loss = g_loss_dict['g_loss']

            if not np.isnan(last_g_loss):
                g_losses.append(last_g_loss)
            pbar.set_postfix({
                    'D_loss': f"{np.mean(d_losses[-self.config.N_CRITIC:]):.4f}",
                    'G_loss': f"{last_g_loss:.4f}",
                    'W_dist': f"{np.mean(w_dists[-self.config.N_CRITIC:]):.4f}"
                })
        return {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses) if g_losses else float('nan'),
            'wasserstein_dist': np.mean(w_dists)
        }
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