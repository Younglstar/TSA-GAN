# file: train_gan.py (修改后的最终版本)
import random
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split  # <-- 1. 导入新模块

from gan_models import Generator, Discriminator, MappingNetwork
from gan_data import load_encoded_data, TimeSeriesDataset, scale_data, inverse_scale_data
from gan_config import GANConfig
from gan_trainer import GANTrainer, EarlyStopping


# ========== 绘图函数 (保持不变) ==========
def plot_gan_loss_curves(history: list, save_path: str):
    """
    一个新增的函数，用于绘制并保存GAN的损失曲线图。
    """
    print("\n[附加步骤] 正在生成GAN损失曲线图...")

    history_df = pd.DataFrame(history).dropna()

    if history_df.empty or len(history_df) < 2:
        print("  - 警告: 有效的历史记录过少，无法生成有意义的损失图。")
        return

    plt.figure(figsize=(12, 8))
    plt.plot(history_df['d_loss'], label='Discriminator Loss', color='red')
    plt.plot(history_df['g_loss'], label='Generator Loss', color='blue')
    plt.plot(history_df['wasserstein_dist'], label='Wasserstein Distance', color='green', linestyle='--')
    plt.title("GAN Training Losses and Wasserstein Distance per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Distance")
    plt.legend()
    plt.grid(True)

    if len(history_df) > 10:
        q_low = history_df['g_loss'].quantile(0.1)
        q_high = history_df['d_loss'].quantile(0.9)
        plt.ylim(-2, 5)

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


# ========== 主训练函数 (已修改) ==========
def main():
    seed = 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已固定为: {seed}")

    config = GANConfig()

    # 1. 加载原始的编码后数据
    original_data = load_encoded_data(config.ENCODED_DATA_PATH)
    print(f"原始编码数据样本量: {len(original_data)}")

    # ==================================================================
    # │ 核心修改：在这里分割训练集和测试集                             │
    # ==================================================================
    print("\n[核心步骤] 正在将数据分割为训练集和测试集...")
    train_data, test_data = train_test_split(original_data, test_size=0.2, random_state=seed)

    # 保存测试集以供后续评估使用
    test_data_path = 'test_data_encoded.pkl'
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  - 训练集大小: {len(train_data)}")
    print(f"  - 测试集大小: {len(test_data)}")
    print(f"💾 测试集已保存至: {test_data_path}")
    # ==================================================================

    # 2. 对 *训练集* 进行缩放和数据增强
    # 注意：我们只对训练数据进行缩放，并用同样的参数去处理测试集（在评估阶段）
    scaled_train_data, scaling_params = scale_data(train_data)
    print("✅ 训练数据已成功缩放到 [-1, 1] 范围。")
    with open('scaling_params.pkl', 'wb') as f:
        pickle.dump(scaling_params, f)
    print(f"💾 缩放参数已保存到 scaling_params.pkl")

    dataset = TimeSeriesDataset(scaled_train_data,
                                augment_factor=config.AUGMENTATION_FACTOR,
                                noise_std=config.AUGMENT_NOISE_STD)
    print(f"增强后的训练数据样本量: {len(dataset)}")

    # 动态确定输入维度
    if 'INPUT_DIM' not in config.__dict__:
        config.INPUT_DIM = train_data.shape[1]

    # ... (模型初始化和训练过程保持不变) ...
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
        dataset=dataset
    )


    print("\n🚀 开始训练...")
    gan_history = []
    for epoch in range(config.NUM_EPOCHS):
        losses = trainer.train_epoch(epoch)
        gan_history.append(losses)
        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} 总结 | "
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

    synthetic_data = inverse_scale_data(synthetic_data_scaled, loaded_scaling_params)
    print("✅ 生成数据已成功还原到原始数据范围。")

    with open(config.SYNTHETIC_DATA_PATH, "wb") as f:
        pickle.dump(synthetic_data, f)
    print(f"💾 合成数据已保存到 {config.SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()