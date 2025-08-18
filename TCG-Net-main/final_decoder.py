# file: final_decoder.py (最终正确版，适配CausalVAE)

import torch
import pickle
import numpy as np
import os

# --- 核心修改：导入我们所有最新的模块和配置 ---
from encdec_model import CausalVAE  # <-- 导入正确的CausalVAE模型
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig


class FinalDecoder:
    def __init__(self):
        """
        初始化解码器流程，加载所有必需的模型和配置。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载新的配置文件
        self.encdec_config = EncoderDecoderConfig()
        self.gan_config = GANConfig()

        print("📊 正在加载所有必需的数据和模型...")
        try:
            # 加载我们新模型所需的特征维度
            with open('feature_dims.pkl', 'rb') as f:
                self.feature_dims = pickle.load(f)

            # 加载我们新的CausalVAE模型
            print(f"  -> 正在加载CausalVAE模型 ({self.encdec_config.MODEL_SAVE_PATH})...")
            self.model = self._load_causal_vae_model()

            # 加载GAN生成的潜在数据
            with open(self.gan_config.SYNTHETIC_DATA_PATH, 'rb') as f:
                self.synthetic_latent_data = pickle.load(f)

            print("所有文件加载成功！")
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}")
            print("  请确保您已按顺序成功运行了所有训练脚本，并生成了所有必需的文件。")
            raise

    def _load_causal_vae_model(self) -> CausalVAE:
        """
        初始化并加载我们新的CausalVAE模型。
        """
        # --- 核心修改：实例化CausalVAE ---
        model = CausalVAE(
            input_dim=self.feature_dims['input_dim'],
            output_dim=self.feature_dims['input_dim'],
            sequence_length=self.feature_dims['sequence_length'],
            tcn_channels=self.encdec_config.TCN_CHANNELS,
            latent_dim=self.encdec_config.LATENT_DIM,
            # 确保传入了num_total_features
            num_total_features=self.feature_dims['input_dim'],
            dropout=self.encdec_config.DROPOUT_RATE
        )

        model_path = self.encdec_config.MODEL_SAVE_PATH
        checkpoint = torch.load(model_path, map_location=self.device)

        # 兼容单卡和多卡(DataParallel)保存的模型
        saved_state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(key.startswith('module.') for key in saved_state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in saved_state_dict.items():
                name = k[7:]  # 移除 `module.` 前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(saved_state_dict)

        model.to(self.device)
        model.eval()
        return model

    def run_decoding(self):
        """
        完整流程：加载GAN生成的潜在数据，用CausalVAE解码器将其还原，并保存最终结果。
        """
        print("\n--- 开始最终解码流程 ---")

        # 使用CausalVAE模型的解码器部分进行解码
        print("  -> 步骤1: 使用CausalVAE解码器还原数据...")
        with torch.no_grad():
            latent_tensor = torch.FloatTensor(self.synthetic_latent_data).to(self.device)
            # --- 核心修改：调用CausalVAE的解码器 ---
            # 它返回 (recon_data, recon_mask)，我们只需要recon_data
            decoded_data_tensor, _ = self.model.decoder(latent_tensor)

        final_synthetic_data = decoded_data_tensor.cpu().numpy()

        # 保存最终的、解码后的合成数据
        final_output_path = 'final_synthetic_data.pkl'
        print(f"  -> 步骤2: 保存最终的、已解码的合成数据到: {final_output_path}...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(final_synthetic_data, f)

        print("\n✅ 合成数据生成和解码流程全部完成！")
        print(f"   最终的合成数据形状: {final_synthetic_data.shape}")


def main():
    try:
        decoder_process = FinalDecoder()
        decoder_process.run_decoding()
    except Exception as e:
        print(f"\n❌ 执行过程中发生未知错误: {e}")
        raise


if __name__ == "__main__":
    main()