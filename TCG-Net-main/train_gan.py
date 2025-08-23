# file: train_gan.py (增加可视化功能)

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt  # <-- 1. 导入绘图库
import pandas as pd  # <-- 导入pandas

from gan_config import GANConfig
from gan_models import Generator, Discriminator, MappingNetwork
from gan_trainer import GANTrainer
from gan_data import (
    load_encoded_data,
    create_dataloader,
    scale_data,
    inverse_scale_data
)


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
        plt.ylim(q_low - 2, q_high + 2)
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


def train_gan(config: GANConfig):
    """GAN的主要训练函数"""
    print("🚀 开始高级GAN (v2) 训练流程...")

    print("\n[步骤 1/4] 加载并准备数据...")
    encoded_data = load_encoded_data(config.ENCODED_DATA_PATH)

    # --- 核心修改：增加数据有效性检查 ---
    if np.isnan(encoded_data).any() or np.isinf(encoded_data).any():
        raise ValueError("错误：加载的编码数据中包含 NaN 或 Inf 值！请检查 CausalVAE 阶段的输出。")
    print("  - 数据有效性检查通过，未发现 NaN 或 Inf。")
    # --- 修改结束 ---

    scaled_data, scaling_params = scale_data(encoded_data)
    dataloader = create_dataloader(scaled_data, config.BATCH_SIZE)
    print("数据集大小:", len(scaled_data), "Batch size:", config.BATCH_SIZE, "迭代步数:", len(dataloader))
    # ... (函数其余部分保持不变) ...
    with open('scaling_params_v2.pkl', 'wb') as f:
        pickle.dump(scaling_params, f)

    print("\n[步骤 2/4] 初始化所有模型...")
    input_dim = encoded_data.shape[1]
    generator, discriminator, mapping_network = setup_models(input_dim, config)

    print("\n[步骤 3/4] 初始化训练器...")
    trainer = GANTrainer(generator, discriminator, mapping_network, config)

    # --- 核心修改：创建列表来记录损失历史 ---
    gan_history = []

    print("\n[步骤 4/4] 🔥 开始训练...")
    for epoch in range(config.NUM_EPOCHS):
        losses = trainer.train_epoch(dataloader, epoch)

        # --- 核心修改：将当轮损失记录到历史中 ---
        gan_history.append(losses)

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} 总结 | "
            f"D_Loss: {losses['d_loss']:.4f}, "
            f"G_Loss: {losses['g_loss']:.4f}, "
            f"W_Dist: {losses['wasserstein_dist']:.4f}"
        )
        # 您可以在这里添加保存最佳模型的逻辑

    print("\n训练完成!")

    # --- 核心修改：在训练结束后调用绘图函数 ---
    plot_gan_loss_curves(gan_history, save_path='gan_loss_curves.png')

    return trainer, scaling_params


def generate_synthetic_data(trainer: GANTrainer, config: GANConfig, scaling_params: dict):
    """使用训练好的GAN生成合成数据"""
    print(f"\n[附加步骤] 正在生成 {config.NUM_SYNTHETIC_SAMPLES} 个合成样本...")
    synthetic_encoded = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)
    synthetic_data = inverse_scale_data(synthetic_encoded, scaling_params)
    with open(config.SYNTHETIC_DATA_PATH, 'wb') as f:
        pickle.dump(synthetic_data, f)
    print(f"合成数据已生成并保存到 {config.SYNTHETIC_DATA_PATH}")
    return synthetic_data


def main():
    """运行完整GAN v2流程的主函数"""

    config = GANConfig()
    trainer, scaling_params = train_gan(config)
    generate_synthetic_data(trainer, config, scaling_params)
    print("\n✅ GAN流程全部完成!")


if __name__ == "__main__":
    main()