# file: train_gan.py (修改后的完整版本)
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from gan_models import Generator, Discriminator, MappingNetwork
from gan_data import load_encoded_data, create_dataloader
from gan_config import GANConfig
from gan_trainer import GANTrainer

# ========== 可选导入差分隐私 ==========
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    print("⚠️ 未安装 opacus，差分隐私训练将被禁用。可运行: pip install opacus")
    OPACUS_AVAILABLE = False

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
        plt.ylim(q_low - 2, q_high + 2)
    # --- 修改结束 ---

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ GAN损失曲线图已保存至: {save_path}")

# ========== 主训练函数 ==========
def main():
    config = GANConfig()
    data = load_encoded_data(config.ENCODED_DATA_PATH)
    dataloader = create_dataloader(data, batch_size=config.BATCH_SIZE)
    # ✅ 正确写法
    generator = Generator(
        w_dim=config.W_DIM,
        output_dim=config.INPUT_DIM,
    )

    discriminator = Discriminator(
        input_dim=config.INPUT_DIM,
        hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
        config=config,
    )

    mapping_network = MappingNetwork(
        z_dim=config.NOISE_DIM,
        w_dim=config.W_DIM,
        hidden_layers=config.MAPPING_HIDDEN_LAYERS,
        hidden_dim=config.MAPPING_HIDDEN_DIM,
    )

    trainer = GANTrainer(generator, discriminator, mapping_network, config)

    # ========== 差分隐私 ==========
    if OPACUS_AVAILABLE and getattr(config, "USE_DP", False):
        print("\n🔐 使用差分隐私优化器 (DP-SGD)...")
        privacy_engine = PrivacyEngine()

        # 用 DP-SGD 替换判别器的优化器
        trainer.discriminator, trainer.d_optimizer, dataloader = privacy_engine.make_private(
            module=trainer.discriminator,
            optimizer=trainer.d_optimizer,
            data_loader=dataloader,
            noise_multiplier=config.DP_NOISE_MULTIPLIER,
            max_grad_norm=config.DP_MAX_GRAD_NORM,
        )

    print("\n🚀 开始训练...")
    gan_history = []
    for epoch in range(config.NUM_EPOCHS):
        losses = trainer.train_epoch(dataloader, epoch)
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
    synthetic_data = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)
    with open(config.SYNTHETIC_DATA_PATH, "wb") as f:
        pickle.dump(synthetic_data, f)
    print(f"💾 合成数据已保存到 {config.SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()
