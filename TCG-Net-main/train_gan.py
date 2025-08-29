# file: train_gan.py (修改后的完整版本)
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from gan_models import Generator, Discriminator, MappingNetwork
from gan_data import load_encoded_data, create_dataloader, TimeSeriesDataset, scale_data, inverse_scale_data
from gan_config import GANConfig
from gan_trainer import GANTrainer,EarlyStopping



# ========== 加载数据 ==========

def plot_gan_loss_curves(history: list, save_path: str):
    """
    一个新增的函数，用于绘制并保存GAN的损失曲线图。
    """
    print("\n[附加步骤] 正在生成GAN损失曲线图...")

    history_df = pd.DataFrame(history).dropna() # <-- 增加 .dropna() 清理无效数据

    # --- 核心修改：增加防御性检查 ---
    if history_df.empty or len(history_df) < 2:
        print("  - 警告: 有效的历史记录过少，无法生成有意义的损失图。")
        return
    # --- 修改结束 ---
    plt.figure(figsize=(12, 8))
    plt.plot(history_df['d_loss'], label='Discriminator Loss', color='red')
    plt.plot(history_df['g_loss'], label='Generator Loss', color='blue')
    plt.plot(history_df['wasserstein_dist'], label='Wasserstein Distance', color='green', linestyle='--')
    plt.title("GAN Training Losses and Wasserstein Distance per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Distance")
    plt.legend()
    plt.grid(True)

    # --- 核心修改：更稳健地设置y轴范围 ---
    if len(history_df) > 10:
        # 使用clip来限制极端值
        q_low = history_df['g_loss'].quantile(0.1)
        q_high = history_df['d_loss'].quantile(0.9)
        plt.ylim(- 10, +10)
    # --- 修改结束 ---

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ GAN损失曲线图已保存至: {save_path}")



def setup_models(input_dim: int, config: GANConfig) -> tuple[Generator, Discriminator, MappingNetwork]:
    """(已修改) 初始化所有GAN模型。"""
    generator = Generator(
        w_dim=config.W_DIM,
        output_dim=input_dim
    )
    discriminator = Discriminator(
        input_dim=input_dim,
        hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT_RATE,
    )
    mapping_network = MappingNetwork(
        z_dim=config.NOISE_DIM, w_dim=config.W_DIM,
        hidden_layers=config.MAPPING_HIDDEN_LAYERS, hidden_dim=config.MAPPING_HIDDEN_DIM
    )
    return generator, discriminator, mapping_network

# ========== 主训练函数 ==========
def main():
    seed = 2  # 您可以选择任何整数作为种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 为了完全的可复现性，需要牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已固定为: {seed}")
    # --- 核心修改：在这里执行离线数据增强 ---
    # 1. 加载原始的编码后数据
    config = GANConfig()
    original_data = load_encoded_data(config.ENCODED_DATA_PATH)
    print(f"原始数据样本量: {len(original_data)}")
    scaled_data, scaling_params = scale_data(original_data)
    print("✅ 数据已成功缩放到 [-1, 1] 范围。")
    with open('scaling_params.pkl', 'wb') as f:
        pickle.dump(scaling_params, f)
    print(f"💾 缩放参数已保存到 scaling_params.pkl")
    dataset = TimeSeriesDataset(scaled_data,
                                augment_factor=config.AUGMENTATION_FACTOR,
                                noise_std=config.AUGMENT_NOISE_STD)
    print(f"增强数据样本量: {len(dataset)}")


    generator = Generator(
        w_dim=config.W_DIM,
        output_dim=config.INPUT_DIM,
        hidden_dims=config.GENERATOR_HIDDEN_DIMS
    )


    discriminator = Discriminator(
         input_dim=config.INPUT_DIM,
         hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
     )


    mapping_network = MappingNetwork(
        z_dim=config.NOISE_DIM,
        w_dim=config.W_DIM,
        hidden_layers=config.MAPPING_HIDDEN_LAYERS,
        hidden_dim=config.MAPPING_HIDDEN_DIM,
    )
    trainer = GANTrainer(
        generator,
        discriminator,
        mapping_network,
        config,
        dataset=dataset # <--- 修改：传递 dataset
    )
    early_stopper = EarlyStopping(patience=100, min_delta=1e-3, monitor="wasserstein_dist")


    print("\n🚀 开始训练...")
    gan_history = []
    for epoch in range(config.NUM_EPOCHS):
        losses = trainer.train_epoch(epoch)
        gan_history.append(losses)
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} 总结 | "
            f"D_Loss: {losses['d_loss']:.4f}, "
            f"G_Loss: {losses['g_loss']:.4f}, "
            f"W_Dist: {losses['wasserstein_dist']:.4f}"
        )

        if (epoch + 1) % 100 == 0:
            trainer.save_model(config.MODEL_SAVE_PATH)

    print("\n✅ 训练完成，生成合成数据...")
    plot_gan_loss_curves(gan_history, save_path='gan_loss_curves.png')
    synthetic_data_scaled = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)

    with open('scaling_params.pkl', 'rb') as f:
        loaded_scaling_params = pickle.load(f)
    print("✅ 已加载缩放参数用于数据还原。")

    # <-- 核心修改：调用 inverse_scale_data 将数据还原到原始范围 -->
    synthetic_data = inverse_scale_data(synthetic_data_scaled, loaded_scaling_params)
    print("✅ 生成数据已成功还原到原始数据范围。")

    with open(config.SYNTHETIC_DATA_PATH, "wb") as f:
        pickle.dump(synthetic_data, f)
    print(f"💾 合成数据已保存到 {config.SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()
