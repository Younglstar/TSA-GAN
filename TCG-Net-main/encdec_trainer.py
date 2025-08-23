# encdec_trainer.py (修改后)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from encdec_config import EncoderDecoderConfig
from encdec_model import CausalVAE


class EncoderDecoderTrainer:
    def __init__(self, model: CausalVAE, config: EncoderDecoderConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR
        )
        self.recon_data_criterion = nn.MSELoss(reduction='mean')
        self.recon_mask_criterion = nn.BCELoss(reduction='mean')
        self.causal_criterion = nn.BCELoss(reduction='mean')

    def _compute_loss(self, inputs, mask, recon_data, recon_mask, mu, logvar, A_pred, current_epoch: int):
        """
        (已修改) 计算CausalVAE的复合损失函数。
        实现了带预热的周期性KL退火和Free Bits技术。
        """
        # 1. 重建损失
        recon_loss = self.recon_data_criterion(recon_data * mask, inputs * mask) + self.recon_mask_criterion(
            recon_mask, mask)

        # 2. KL 散度损失 (原始计算)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. 因果损失
        causal_target = torch.full_like(A_pred, self.config.CAUSAL_TARGET_VALUE)  # <--- 修改：使用配置文件中的值
        causal_loss = self.causal_criterion(A_pred, causal_target)

        # 4. 计算KL权重 (beta)，采用带预热的平滑周期退火策略
        # <--- 以下是全新的、更温和的beta计算逻辑 ---
        warmup_epochs = self.config.KL_ANNEALING_WARMUP_EPOCHS
        cycle_length = self.config.KL_ANNEALING_CYCLE_EPOCHS

        if current_epoch < warmup_epochs:
            beta = 0.0
        else:
            epoch_after_warmup = current_epoch - warmup_epochs
            epoch_in_cycle = epoch_after_warmup % cycle_length
            # 在整个周期内，beta从0平滑地线性增长到BETA_KL_FINAL
            beta_ratio = epoch_in_cycle / cycle_length
            beta = beta_ratio * self.config.BETA_KL_FINAL

        # 5. 应用 "Free Bits" 技术
        # <--- 新增：只有当KL损失超过阈值时，才对其施加惩罚 ---
        free_bits_threshold = self.config.FREE_BITS_THRESHOLD
        modified_kld_loss = torch.clamp(kld_loss, min=free_bits_threshold)

        # 6. 加权总损失 (使用修正后的KL损失)
        total_loss = recon_loss + beta * modified_kld_loss + self.config.GAMMA_CAUSAL * causal_loss

        # 7. 记录各项损失 (注意：记录原始的kld_loss以方便监控其真实变化)
        loss_dict = {'total_loss': total_loss.item(),
                     'recon_loss': recon_loss.item(),
                     'kld_loss': kld_loss.item(),  # 记录原始值
                     'causal_loss': causal_loss.item(),
                     'current_beta': beta}

        return total_loss, loss_dict

    def _run_batch(self, batch: torch.Tensor, is_training: bool, current_epoch: int):
        inputs_clean = batch.to(self.device)  # 原始干净数据
        # --- 新增：只在训练时加入噪声 ---
        if is_training:
            noise = torch.randn_like(inputs_clean) * 0.1  # 0.1是噪声标准差，可调
            inputs_noisy = inputs_clean + noise
        else:
            inputs_noisy = inputs_clean  # 验证时不加噪声

        mask = (inputs_clean != 0).float()
        # 将带噪声的数据送入模型
        recon_data, recon_mask, mu, logvar, A_pred = self.model(inputs_noisy)
        # 用干净的数据计算损失
        loss, loss_dict = self._compute_loss(inputs_clean, mask, recon_data, recon_mask, mu, logvar, A_pred,current_epoch)
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_dict
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.train()
        epoch_losses = []
        for (batch,) in tqdm(dataloader, desc=f"训练中 (Epoch {epoch})"):
            epoch_losses.append(self._run_batch(batch, is_training=True, current_epoch=epoch))
        return {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}

    def validate(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.eval()
        validation_losses = []
        with torch.no_grad():
            for (batch,) in tqdm(dataloader, desc=f"验证中 (Epoch {epoch})"):
                validation_losses.append(self._run_batch(batch, is_training=False, current_epoch=epoch))
        avg_losses = {k: np.mean([d[k] for d in validation_losses]) for k in validation_losses[0]}
        self.scheduler.step(avg_losses['total_loss'])
        return avg_losses

    def save_model(self, path: str):
        model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                   nn.DataParallel) else self.model.state_dict()
        torch.save({'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])