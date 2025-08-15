# file: encdec_trainer.py (修改后以适配CausalVAE)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from encdec_config import EncoderDecoderConfig
# --- 核心修改：导入新的CausalVAE模型 ---
from encdec_model import CausalVAE


class EncoderDecoderTrainer:
    # file: encdec_trainer.py (只修改这两个方法)

    def __init__(
            self,
            model: CausalVAE,
            config: EncoderDecoderConfig,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )

        # --- 核心修正 1：将损失计算方式改为 'mean' ---
        self.recon_data_criterion = nn.MSELoss(reduction='mean')
        self.recon_mask_criterion = nn.BCELoss(reduction='mean')
        self.causal_criterion = nn.BCELoss(reduction='mean')  # <-- 新增，用于因果损失

    def _compute_loss(
            self,
            inputs: torch.Tensor,
            mask: torch.Tensor,
            recon_data: torch.Tensor,
            recon_mask: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            A_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        (已修改) 计算CausalVAE的复合损失函数。
        """
        # 1. 重构损失 (现在是均值，数值会小很多)
        recon_loss_data = self.recon_data_criterion(recon_data * mask, inputs * mask)
        recon_loss_mask = self.recon_mask_criterion(recon_mask, mask)
        recon_loss = recon_loss_data + recon_loss_mask

        # 2. KL散度损失 (现在也是均值)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 3. 因果损失 (使用BCE Loss改进)
        # 我们创建一个非常稀疏的目标矩阵 (例如，假设只有5%的连接)
        # 这会鼓励模型去预测一些非零的连接，而不是直接输出全零
        causal_target = torch.full_like(A_pred, 0.05)
        causal_loss = self.causal_criterion(A_pred, causal_target)

        # 4. 加权总损失
        total_loss = (
                recon_loss +
                self.config.BETA_KL * kld_loss +
                self.config.GAMMA_CAUSAL * causal_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item(),
            'causal_loss': causal_loss.item()
        }

        return total_loss, loss_dict

    def _run_batch(self, batch: torch.Tensor, is_training: bool) -> Dict[str, float]:
        """
        一个统一的函数来处理一个批次的训练或验证。
        """
        inputs = batch.to(self.device)

        # 假设掩码是数据中非零的部分 (这是一个简化，可以根据需要修改)
        mask = (inputs != 0).float()

        # 前向传播，获取所有输出
        recon_data, recon_mask, mu, logvar, A_pred = self.model(inputs)

        # 计算复合损失
        loss, loss_dict = self._compute_loss(
            inputs, mask, recon_data, recon_mask, mu, logvar, A_pred
        )

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_dict

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """训练一个轮次。"""
        self.model.train()
        epoch_losses = []

        for (batch,) in tqdm(dataloader, desc="训练中"):
            loss_dict = self._run_batch(batch, is_training=True)
            epoch_losses.append(loss_dict)

        # 计算整个轮次的平均损失
        avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}
        return avg_losses

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """验证模型。"""
        self.model.eval()
        validation_losses = []

        with torch.no_grad():
            for (batch,) in tqdm(dataloader, desc="验证中"):
                loss_dict = self._run_batch(batch, is_training=False)
                validation_losses.append(loss_dict)

        avg_losses = {k: np.mean([d[k] for d in validation_losses]) for k in validation_losses[0]}
        return avg_losses

    def save_model(self, path: str):
        """保存模型和优化器状态。"""
        model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                   nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        """加载模型和优化器状态。"""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])