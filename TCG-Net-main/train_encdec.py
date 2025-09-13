# train_encdec.py (修改后的版本)
import pickle

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm

# ... (顶部的 imports) ...
from config import DataConfig
from encdec_config import EncoderDecoderConfig
from embedding_config import EmbeddingConfig  # <-- 1. 导入 EmbeddingConfig
from encdec_model import CausalVAE
from encdec_trainer import EncoderDecoderTrainer
from encdec_processor import DataProcessor


# ... (plot_loss_curves 函数保持不变) ...
def plot_loss_curves(train_history: list, val_history: list, save_path: str):
    """
    (已优化) 绘制并保存损失曲线图，增加健壮性检查。
    """
    print("\n[附加步骤] 正在生成损失曲线图...")
    if not train_history or not val_history:
        print("  - 警告: 损失历史记录为空，无法生成损失曲线图。")
        print("  - 这通常发生在训练因早停在第一个epoch就结束的情况下。")
        return

    train_df = pd.DataFrame(train_history);
    val_df = pd.DataFrame(val_history)
    loss_names = train_df.columns
    fig, axes = plt.subplots(len(loss_names), 1, figsize=(10, 5 * len(loss_names)), sharex=True)
    if len(loss_names) == 1: axes = [axes]

    for i, loss_name in enumerate(loss_names):
        axes[i].plot(train_df.index, train_df[loss_name], label=f'Train {loss_name}', color='royalblue')
        axes[i].plot(val_df.index, val_df[loss_name], label=f'Validation {loss_name}', color='tomato', linestyle='--')
        axes[i].set_ylabel("Loss");
        axes[i].set_title(f"Training & Validation: {loss_name.replace('_', ' ').title()}");
        axes[i].legend();
        axes[i].grid(True)

    axes[-1].set_xlabel("Epoch");
    plt.tight_layout();
    plt.savefig(save_path);
    plt.close()
    print(f"  - 损失曲线图已保存至: {save_path}")


def train(encdec_config: EncoderDecoderConfig, data_config: DataConfig,
          embed_config: EmbeddingConfig):  # <-- 2. 增加 embed_config 参数
    """主训练函数"""
    print("--- 开始CausalVAE训练流程 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 1. 初始化数据处理器并处理数据
    print("\n[步骤 1/4] 正在处理数据并转换为3D张量...")
    # <-- 3. 将 embed_config 传入 DataProcessor
    processor = DataProcessor(encdec_config, data_config, embed_config)
    combined_data = processor.process_data()  # 此步骤会生成 feature_dims.pkl

    # ... (函数剩余部分保持不变) ...
    print("\n[步骤 2/4] 正在动态加载数据维度...")
    feature_dims = processor.load_feature_dims()
    input_dim = feature_dims['input_dim']
    sequence_length = feature_dims['sequence_length']
    num_total_features = feature_dims['num_total_features']
    print(f"  - 输入维度: {input_dim}, 序列长度: {sequence_length}, 总特征数: {num_total_features}")

    # 3. 创建数据集和数据加载器
    data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    val_size = int(len(dataset) * 0.2);
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=encdec_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=encdec_config.BATCH_SIZE, shuffle=False)

    # 4. 动态实例化模型
    print("\n[步骤 3/4] 正在根据动态维度构建CausalVAE模型...")
    model = CausalVAE(
        input_dim=input_dim, output_dim=input_dim, sequence_length=sequence_length,
        tcn_channels=encdec_config.TCN_CHANNELS, latent_dim=encdec_config.LATENT_DIM,
        num_total_features=num_total_features, dropout=encdec_config.DROPOUT_RATE
    )
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model.to(device)

    # 5. 训练模型
    trainer = EncoderDecoderTrainer(model, encdec_config, device)
    print("\n[步骤 4/4] 开始训练 (带学习率调度和早停)...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_history, val_history = [], []
    # --- 新增的诊断代码 ---
    print("\n[诊断信息] 检查数据集大小:")
    print(f"  - 总样本数 (len(dataset)): {len(dataset)}")
    print(f"  - 训练集大小 (train_size): {train_size}")
    print(f"  - 验证集大小 (val_size): {val_size}")
    print(f"  - 训练加载器批次数 (len(train_loader)): {len(train_loader)}")
    for epoch in range(encdec_config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{encdec_config.NUM_EPOCHS} ---")
        train_losses = trainer.train_epoch(train_loader, epoch)
        val_losses = trainer.validate(val_loader, epoch)
        train_history.append(train_losses);
        val_history.append(val_losses)

        train_log = " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        print(f"  - 训练损失: {train_log}")
        val_log = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
        print(f"  - 验证损失: {val_log}")

        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            trainer.save_model(encdec_config.MODEL_SAVE_PATH)
            print(f"  - 🎉 新的最佳验证损失: {best_val_loss:.6f}。模型已保存。")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  - 验证损失未提升，已连续 {epochs_no_improve} 轮。")

        if epochs_no_improve >= encdec_config.EARLY_STOPPING_PATIENCE:
            print(f"\n✋ 早停触发！验证损失已连续 {encdec_config.EARLY_STOPPING_PATIENCE} 轮未提升。")
            break

    print("\n--- 训练完成！ ---")
    plot_loss_curves(train_history, val_history, save_path='cvae_loss_curves.png')

    print(f"\n正在加载性能最佳的模型 ({encdec_config.MODEL_SAVE_PATH}) 用于数据编码...")
    trainer.load_model(encdec_config.MODEL_SAVE_PATH)
    return trainer


def encode_data(trainer: EncoderDecoderTrainer, encdec_config: EncoderDecoderConfig, data_config: DataConfig,
                embed_config: EmbeddingConfig):  # <-- 4. 增加 embed_config 参数
    """使用训练好的最佳模型编码所有数据。"""
    print("\n--- 开始使用最佳VAE模型编码全部数据 ---")
    # <-- 5. 将 embed_config 传入 DataProcessor
    processor = DataProcessor(encdec_config, data_config, embed_config)
    combined_data = processor.process_data()
    data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    full_loader = DataLoader(TensorDataset(data_tensor), batch_size=encdec_config.BATCH_SIZE, shuffle=False)

    encoded_mu_list = []
    model_to_encode = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
    model_to_encode.eval()
    with torch.no_grad():
        for (batch,) in tqdm(full_loader, desc="编码中"):
            mu, _, _ = model_to_encode.encoder(batch.to(trainer.device))
            encoded_mu_list.append(mu.cpu().numpy())

    encoded_data = np.concatenate(encoded_mu_list, axis=0)
    with open(encdec_config.ENCODED_DATA_PATH, 'wb') as f: pickle.dump(encoded_data, f)
    print(f"  - 编码后的数据 (形状: {encoded_data.shape}) 已保存至 {encdec_config.ENCODED_DATA_PATH}")
    return encoded_data


def main():
    """主执行函数"""
    data_config = DataConfig()
    encdec_config = EncoderDecoderConfig()
    embed_config = EmbeddingConfig()  # <-- 6. 创建 embed_config 实例

    # <-- 7. 将所有需要的 config 传入
    trainer = train(encdec_config, data_config, embed_config)
    encode_data(trainer, encdec_config, data_config, embed_config)

    print("\n--- ✅ CausalVAE 流程全部完成！ ---")


if __name__ == "__main__":
    main()