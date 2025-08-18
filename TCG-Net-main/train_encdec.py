# file: train_encdec.py (增加优化策略)

import torch
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from encdec_config import EncoderDecoderConfig
from encdec_model import CausalVAE
from encdec_trainer import EncoderDecoderTrainer
from encdec_processor import DataProcessor

def plot_loss_curves(train_history: list, val_history: list, save_path: str):
    """
    一个新增的函数，用于绘制并保存损失曲线图。
    """
    print("\n[附加步骤] 正在生成损失曲线图...")

    # --- 核心修正：增加防御性检查 ---
    if not train_history or not val_history:
        print("\n[警告] 损失历史记录为空，无法生成损失曲线图。")
        print("  -> 这通常意味着训练循环因为早停或其他原因而提前终止。")
        return # 直接退出函数，避免报错
    # --- 修正结束 ---

    # 将损失历史转换为DataFrame，方便处理
    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history)

    # 获取所有损失的名称
    loss_names = train_df.columns

    # 为每种损失创建一个子图
    fig, axes = plt.subplots(len(loss_names), 1, figsize=(10, 6 * len(loss_names)), sharex=True)
    if len(loss_names) == 1:
        axes = [axes]

    for i, loss_name in enumerate(loss_names):
        axes[i].plot(train_df.index, train_df[loss_name], label=f'Train {loss_name}', color='blue', marker='o',
                     linestyle='-')
        axes[i].plot(val_df.index, val_df[loss_name], label=f'Validation {loss_name}', color='red', marker='x',
                     linestyle='--')
        axes[i].set_ylabel("Loss")
        axes[i].set_title(f"Training and Validation {loss_name.replace('_', ' ').title()}")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 损失曲线图已保存至: {save_path}")

def train(config: EncoderDecoderConfig):
    """主训练函数"""
    print("🚀 开始CausalVAE训练流程 (带优化策略)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据处理和模型初始化部分 (保持不变) ---
    processor = DataProcessor(config);
    combined_data = processor.process_data();
    processor.save_feature_dims()
    input_dim = processor.feature_dims['input_dim'];
    sequence_length = processor.feature_dims['sequence_length']
    config.NUM_TOTAL_FEATURES = input_dim
    data_tensor = torch.tensor(combined_data, dtype=torch.float32);
    dataset = TensorDataset(data_tensor)
    val_size = int(len(dataset) * 0.2);
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    model = CausalVAE(
        input_dim=input_dim, output_dim=input_dim, sequence_length=sequence_length,
        tcn_channels=config.TCN_CHANNELS, latent_dim=config.LATENT_DIM,
        num_total_features=config.NUM_TOTAL_FEATURES, dropout=config.DROPOUT_RATE
    )
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model.to(device)

    # --- 初始化我们升级后的训练器 ---
    trainer = EncoderDecoderTrainer(model, config, device)

    # --- 核心修改：实现早停法逻辑 ---
    print("\n[步骤 4/4] 🔥 开始训练 (带学习率调度和早停)...")
    best_val_loss = float('inf')
    epochs_no_improve = 0  # 记录验证损失没有提升的轮次数
    train_history, val_history = [], []

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_losses = trainer.train_epoch(train_loader, epoch)
        val_losses = trainer.validate(val_loader, epoch)

        # --- 核心修正：在这里补上被遗漏的存储步骤 ---
        train_history.append(train_losses)
        val_history.append(val_losses)
        # --- 修正结束 ---

        train_log = " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        print(f"训练损失: {train_log}")

        val_log = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
        print(f"验证损失: {val_log}")

        current_val_loss = val_losses['total_loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            trainer.save_model(config.MODEL_SAVE_PATH)
            print(f"🎉 新的最佳验证损失: {best_val_loss:.6f}. 模型已保存。")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> 验证损失未提升，已连续 {epochs_no_improve} 轮。")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n✋ 早停触发！验证损失已连续 {config.EARLY_STOPPING_PATIENCE} 轮未提升。")
            print(f"训练在第 {epoch + 1} 轮结束。")
            break

    print("\n训练完成!")
    plot_loss_curves(train_history, val_history, save_path='causal_vae_loss_curves.png')

    # 加载回最佳模型进行后续编码
    print(f"\n正在加载性能最佳的模型 ({config.MODEL_SAVE_PATH}) 用于数据编码...")
    trainer.load_model(config.MODEL_SAVE_PATH)
    return trainer


# ... (encode_data 和 main 函数保持不变) ...
def encode_data(trainer: EncoderDecoderTrainer, config: EncoderDecoderConfig):
    # ... (此函数无需修改)
    print("\n[附加步骤] 正在使用最佳VAE模型编码所有数据...")
    processor = DataProcessor(config);
    combined_data = processor.process_data()
    data_tensor = torch.tensor(combined_data, dtype=torch.float32);
    full_dataset = TensorDataset(data_tensor)
    full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    encoded_mu_list = [];
    trainer.model.eval()
    model_to_encode = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
    with torch.no_grad():
        for (batch,) in tqdm(full_loader, desc="编码中"):
            inputs = batch.to(trainer.device)
            mu, _, _ = model_to_encode.encoder(inputs)
            encoded_mu_list.append(mu.cpu().numpy())
    encoded_data = np.concatenate(encoded_mu_list, axis=0)
    with open(config.ENCODED_DATA_PATH, 'wb') as f: pickle.dump(encoded_data, f)
    print(f"\n编码后的数据已保存至 {config.ENCODED_DATA_PATH}")
    return encoded_data


def main():
    config = EncoderDecoderConfig()
    trainer = train(config)
    encode_data(trainer, config)
    print("\n✅ CausalVAE流程全部完成!")


if __name__ == "__main__":
    main()