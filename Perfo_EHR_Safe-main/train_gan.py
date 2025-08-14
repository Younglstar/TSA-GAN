# file: train_gan.py (调整后，以调用gan_data.py)

import torch
import numpy as np
import pickle
# 移除 torch.utils.data 的导入，因为 gan_data.py 会处理

from gan_config import GANConfig
from gan_models import Generator, Discriminator, MappingNetwork
from gan_trainer import GANTrainer
# --- 核心修改：从 gan_data.py 导入所需的函数 ---
from gan_data import (
    load_encoded_data,
    create_dataloader,
    scale_data,
    inverse_scale_data
)


# --- 修改结束 ---


def setup_models(input_dim: int, config: GANConfig) -> tuple[Generator, Discriminator, MappingNetwork]:
    """初始化所有GAN模型 (此函数保持不变)"""
    generator = Generator(
        w_dim=config.W_DIM,
        output_dim=input_dim,
        hidden_dims=config.GENERATOR_HIDDEN_DIMS,
        attention_channels=config.ATTENTION_CHANNELS,
        attention_seq_len=config.ATTENTION_SEQ_LEN,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    )

    discriminator = Discriminator(
        input_dim=input_dim,
        hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    )

    mapping_network = MappingNetwork(
        z_dim=config.NOISE_DIM,
        w_dim=config.W_DIM,
        hidden_layers=config.MAPPING_HIDDEN_LAYERS,
        hidden_dim=config.MAPPING_HIDDEN_DIM
    )

    return generator, discriminator, mapping_network


# --- 核心修改：重构 train_gan 和 generate_synthetic_data 函数 ---
def train_gan(config: GANConfig):
    """GAN的主要训练函数"""
    print("🚀 开始高级GAN (v2) 训练流程...")

    # 调用 gan_data.py 中的函数来准备数据
    print("\n[步骤 1/4] 加载并准备数据...")
    encoded_data = load_encoded_data(config.ENCODED_DATA_PATH)
    scaled_data, scaling_params = scale_data(encoded_data)
    dataloader = create_dataloader(scaled_data, config.BATCH_SIZE)

    # 保存缩放参数
    with open('scaling_params_v2.pkl', 'wb') as f:
        pickle.dump(scaling_params, f)

    # 初始化模型
    print("\n[步骤 2/4] 初始化所有模型...")
    input_dim = encoded_data.shape[1]
    generator, discriminator, mapping_network = setup_models(input_dim, config)

    # 初始化训练器
    print("\n[步骤 3/4] 初始化训练器...")
    trainer = GANTrainer(generator, discriminator, mapping_network, config)

    # 训练循环
    print("\n[步骤 4/4] 🔥 开始训练...")
    for epoch in range(config.NUM_EPOCHS):
        losses = trainer.train_epoch(dataloader, epoch)
        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} 总结 | "
            f"D_Loss: {losses['d_loss']:.4f}, "
            f"G_Loss: {losses['g_loss']:.4f}, "
            f"W_Dist: {losses['wasserstein_dist']:.4f}"
        )
        # 您可以在这里添加保存最佳模型的逻辑

    print("\n训练完成!")
    return trainer, scaling_params


def generate_synthetic_data(trainer: GANTrainer, config: GANConfig, scaling_params: dict):
    """使用训练好的GAN生成合成数据"""
    print(f"\n[附加步骤] 正在生成 {config.NUM_SYNTHETIC_SAMPLES} 个合成样本...")

    # 在潜在空间中生成样本
    # 注意：generate_samples也需要更新以使用MappingNetwork
    # 我们将在GANTrainer中更新它
    synthetic_encoded = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)

    # 将数据反向缩放回原始范围
    synthetic_data = inverse_scale_data(synthetic_encoded, scaling_params)

    # 保存合成数据
    with open(config.SYNTHETIC_DATA_PATH, 'wb') as f:
        pickle.dump(synthetic_data, f)

    print(f"合成数据已生成并保存到 {config.SYNTHETIC_DATA_PATH}")
    return synthetic_data


# --- 修改结束 ---

def main():
    """运行完整GAN v2流程的主函数"""
    config = GANConfig()

    trainer, scaling_params = train_gan(config)

    generate_synthetic_data(trainer, config, scaling_params)

    print("\n✅ GAN流程全部完成!")


if __name__ == "__main__":
    main()