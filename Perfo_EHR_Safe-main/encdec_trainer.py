# file: encdec_trainer.py (修改后以适配TCN)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import numpy as np
from tqdm import tqdm
from encdec_config import EncoderDecoderConfig
# --- 核心修改：导入新的TCNAutoencoder模型 ---
from encdec_model import TCNAutoencoder


class EncoderDecoderTrainer:
    def __init__(
            self,
            model: TCNAutoencoder,  # <-- 模型类型变为 TCNAutoencoder
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

        # --- 核心修改：损失函数简化 ---
        # 我们现在只有一个统一的重构目标，因此只需要一个MSE损失
        self.reconstruction_criterion = nn.MSELoss()

    def _run_batch(self, batch: torch.Tensor, is_training: bool) -> Dict[str, float]:
        """
        一个统一的函数来处理一个批次的训练或验证。
        """
        # 数据现在是一个单一的张量
        inputs = batch.to(self.device)

        # 前向传播
        reconstructed, latent = self.model(inputs)

        # --- 核心修改：损失计算简化 ---
        loss = self.reconstruction_criterion(reconstructed, inputs)

        if is_training:
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'total_loss': loss.item()}

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """训练一个轮次。"""
        self.model.train()
        epoch_losses = []

        for (batch,) in tqdm(dataloader, desc="训练中"):  # <-- 注意 (batch,) 的解包方式
            loss_dict = self._run_batch(batch, is_training=True)
            epoch_losses.append(loss_dict)

        # 计算整个轮次的平均损失
        avg_loss = np.mean([d['total_loss'] for d in epoch_losses])
        return {'total_loss': avg_loss}

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """验证模型。"""
        self.model.eval()
        validation_losses = []

        with torch.no_grad():
            for (batch,) in tqdm(dataloader, desc="验证中"):  # <-- 注意 (batch,) 的解包方式
                loss_dict = self._run_batch(batch, is_training=False)
                validation_losses.append(loss_dict)

        # 计算整个验证集的平均损失
        avg_loss = np.mean([d['total_loss'] for d in validation_losses])
        return {'total_loss': avg_loss}

    def save_model(self, path: str):
        """保存模型和优化器状态。"""
        # 注意：如果使用了DataParallel，需要保存 .module
        model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                   nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        """加载模型和优化器状态。"""
        checkpoint = torch.load(path, map_location=self.device)
        # 同样需要处理DataParallel的情况
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])