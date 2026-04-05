import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from encdec_model import CausalVAE
from encdec_config import EncoderDecoderConfig


def visualize_causal_structure(model_path='cvae_model.pkl', dims_path='feature_dims.pkl'):
    # 1. 自动获取维度信息
    if not os.path.exists(dims_path):
        print(f"错误: 未找到 {dims_path}")
        return

    with open(dims_path, 'rb') as f:
        dims = pickle.load(f)

    # 从 pkl 中提取所有必需维度
    input_dim = dims['input_dim']
    sequence_length = dims['sequence_length']
    num_total_features = dims['num_total_features']
    # 通常 VAE 的 output_dim 等于 input_dim
    output_dim = input_dim

    # 2. 准备配置和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EncoderDecoderConfig()

    # 3. 实例化模型 (使用所有必需参数)
    try:
        model = CausalVAE(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=config.LATENT_DIM,
            sequence_length=sequence_length,
            num_total_features=num_total_features,
            tcn_channels=config.TCN_CHANNELS
        ).to(device)
    except TypeError as e:
        print(f"模型初始化依然失败，请检查 CausalVAE 定义。错误信息: {e}")
        return

    # ... (后续加载权重和绘图逻辑保持不变)

    # 3. 加载权重
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 4. 提取并处理 A
    # 替换原本报错的那一行
    with torch.no_grad():
        # 方案 A: 提取全局平均因果趋势（查看 causal_head 的最后一层权重）
        # 假设你想看特征之间的映射逻辑，而不是具体样本的输出
        last_layer = model.encoder.causal_head[-1]
        weights = last_layer.weight.cpu().numpy()  # 这会非常大，通常不直接可视化

        # 方案 B: 喂入一个全零或随机输入，获取模型输出的 A_logits
        dummy_input = torch.zeros((1, sequence_length, input_dim)).to(device)
        mu, logvar, A_logits = model.encoder(dummy_input)
        A_prob = torch.sigmoid(A_logits).squeeze(0).cpu().numpy()  # 取出第一张图

    # 5. 可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(A_prob, cmap='magma', vmin=0, vmax=0.05)
    plt.title(f"Causal Adjacency Matrix (Epoch {checkpoint.get('epoch', 'Unknown')})")
    plt.xlabel("Latent Node (Source)")
    plt.ylabel("Latent Node (Target)")

    save_path = 'causal_matrix_final.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"可视化矩阵已保存至: {save_path}")


if __name__ == "__main__":
    visualize_causal_structure()