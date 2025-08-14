# file: train_encdec.py (修改后以适配TCN)

import torch
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- 核心修改：导入新的模块 ---
from encdec_config import EncoderDecoderConfig
from encdec_model import TCNAutoencoder  # <-- 导入新模型
from encdec_trainer import EncoderDecoderTrainer  # <-- 导入修改后的训练器
from encdec_processor import DataProcessor


def train(config: EncoderDecoderConfig):
    """主训练函数"""
    print("🚀 开始TCN自编码器训练流程...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 处理数据 (现在返回一个统一的数组)
    print("\n[步骤 1/4] 正在处理数据...")
    processor = DataProcessor(config)
    combined_data = processor.process_data()
    processor.save_feature_dims()

    # 2. 创建Dataloaders (逻辑简化)
    print("\n[步骤 2/4] 正在创建数据加载器...")
    data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化模型 (适配TCNAutoencoder)
    print("\n[步骤 3/4] 正在初始化TCN模型...")
    model = TCNAutoencoder(
        input_dim=processor.feature_dims['input_dim'],
        output_dim=processor.feature_dims['input_dim'],
        sequence_length=processor.feature_dims['sequence_length'],
        tcn_channels=config.TCN_CHANNELS,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT_RATE
    )

    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个GPU，使用DataParallel模式。")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 4. 初始化训练器 (使用我们修改后的Trainer)
    trainer = EncoderDecoderTrainer(model, config, device)

    # 5. 训练循环 (现在逻辑更清晰)
    print("\n[步骤 4/4] 🔥 开始训练...")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        train_losses = trainer.train_epoch(train_loader)
        print(f"平均训练损失: {train_losses['total_loss']:.6f}")

        val_losses = trainer.validate(val_loader)
        print(f"平均验证损失: {val_losses['total_loss']:.6f}")

        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            trainer.save_model(config.MODEL_SAVE_PATH)
            print(f"🎉 新的最佳验证损失: {best_val_loss:.6f}. 模型已保存。")

    print("\n训练完成!")
    return trainer


def encode_data(trainer: EncoderDecoderTrainer, config: EncoderDecoderConfig):
    """使用训练好的模型编码所有数据。"""
    print("\n[附加步骤] 正在编码所有数据...")

    # 重新加载完整数据
    processor = DataProcessor(config)
    combined_data = processor.process_data()
    data_tensor = torch.tensor(combined_data, dtype=torch.float32)
    full_dataset = TensorDataset(data_tensor)
    full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    encoded_data = []
    trainer.model.eval()

    # 获取底层的实际模型（处理DataParallel）
    model_to_encode = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model

    with torch.no_grad():
        for (batch,) in tqdm(full_loader, desc="编码中"):
            inputs = batch.to(trainer.device)
            # --- 核心修改：通过模型前向传播获取潜在向量 ---
            _, latent = model_to_encode(inputs)  # TCNAutoencoder返回 (reconstructed, latent)
            encoded_data.append(latent.cpu().numpy())

    encoded_data = np.concatenate(encoded_data, axis=0)

    print(f"\n正在将编码后的数据保存到 {config.ENCODED_DATA_PATH}")
    with open(config.ENCODED_DATA_PATH, 'wb') as f:
        pickle.dump(encoded_data, f)

    return encoded_data


def main():
    config = EncoderDecoderConfig()
    trainer = train(config)
    encoded_data = encode_data(trainer, config)
    print("\n✅ TCN自编码器流程全部完成!")
    print(f"  - 模型已保存至: {config.MODEL_SAVE_PATH}")
    print(f"  - 编码后的数据已保存至: {config.ENCODED_DATA_PATH}")


if __name__ == "__main__":
    main()