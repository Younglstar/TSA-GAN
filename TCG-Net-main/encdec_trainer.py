# file: encdec_trainer.py (实现周期性KL退火)

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

        # file: encdec_trainer.py (只修改 _compute_loss 方法)

        # file: encdec_trainer.py (只修改 _compute_loss 方法)

    def _compute_loss(self, inputs, mask, recon_data, recon_mask, mu, logvar, A_pred, current_epoch: int):
            """
            (已修改) 计算CausalVAE的复合损失函数，并实现周期性KL退火。
            """
            # ... (重构、KL、因果损失的计算保持不变) ...
            recon_loss = self.recon_data_criterion(recon_data * mask, inputs * mask) + self.recon_mask_criterion(
                recon_mask, mask)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            causal_target = torch.full_like(A_pred, 0.05);
            causal_loss = self.causal_criterion(A_pred, causal_target)

            # --- 核心修改：使用新的周期长度参数来计算beta ---
            cycle_length = self.config.KL_ANNEALING_CYCLE_EPOCHS
            epoch_in_cycle = current_epoch % cycle_length
            # 在每个周期的前半段，beta从0线性增长到1 (这里的逻辑也可以调整得更缓和)
            beta = min(1.0, (epoch_in_cycle / (cycle_length / 2))) * self.config.BETA_KL_FINAL

            # 4. 加权总损失
            total_loss = recon_loss + beta * kld_loss + self.config.GAMMA_CAUSAL * causal_loss

            loss_dict = {'total_loss': total_loss.item(), 'recon_loss': recon_loss.item(),
                         'kld_loss': kld_loss.item(), 'causal_loss': causal_loss.item(), 'current_beta': beta}

            return total_loss, loss_dict



    # ... (其余方法 _run_batch, train_epoch, validate, save, load 都保持不变) ...
    # 为了简洁，此处省略，它们已经能够正确传递epoch参数
    def _run_batch(self, batch: torch.Tensor, is_training: bool, current_epoch: int):
        inputs = batch.to(self.device);
        mask = (inputs != 0).float()
        recon_data, recon_mask, mu, logvar, A_pred = self.model(inputs)
        loss, loss_dict = self._compute_loss(inputs, mask, recon_data, recon_mask, mu, logvar, A_pred, current_epoch)
        if is_training: self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss_dict

    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.train();
        epoch_losses = []
        for (batch,) in tqdm(dataloader, desc="训练中"):
            epoch_losses.append(self._run_batch(batch, is_training=True, current_epoch=epoch))
        return {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}

    def validate(self, dataloader: torch.utils.data.DataLoader, epoch: int):
        self.model.eval();
        validation_losses = []
        with torch.no_grad():
            for (batch,) in tqdm(dataloader, desc="验证中"):
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