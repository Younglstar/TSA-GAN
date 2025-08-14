# file: final_decoder.py (推荐的文件名)

import torch
import pickle
import numpy as np

# --- 导入我们新的模型和配置 ---
from encdec_model import TCNAutoencoder
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig  # 我们需要从这里获取GAN生成的数据文件名


class FinalDecoder:
    def __init__(self):
        """
        初始化解码器流程，加载所有必需的模型和配置。
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载新的配置文件
        self.encdec_config = EncoderDecoderConfig()
        self.gan_config = GANConfig()

        # 加载我们新模型所需的特征维度
        print("1. 加载特征维度 (feature_dims.pkl)...")
        with open('feature_dims.pkl', 'rb') as f:
            self.feature_dims = pickle.load(f)

        # 加载我们新的TCN自编码器模型
        print(f"2. 加载TCN自编码器模型 ({self.encdec_config.MODEL_SAVE_PATH})...")
        self.model = self._load_tcn_autoencoder()

    def _load_tcn_autoencoder(self) -> TCNAutoencoder:
        """
        初始化并加载我们新的TCN自编码器模型。
        """
        model = TCNAutoencoder(
            input_dim=self.feature_dims['input_dim'],
            output_dim=self.feature_dims['input_dim'],
            sequence_length=self.feature_dims['sequence_length'],
            tcn_channels=self.encdec_config.TCN_CHANNELS,
            latent_dim=self.encdec_config.LATENT_DIM,
            dropout=self.encdec_config.DROPOUT_RATE
        )

        # 加载训练好的模型权重
        model_path = self.encdec_config.MODEL_SAVE_PATH
        checkpoint = torch.load(model_path, map_location=self.device)

        # 兼容单卡和多卡(DataParallel)保存的模型
        # 如果是DataParallel保存的，state_dict的键会带有'module.'前缀
        if isinstance(model, torch.nn.DataParallel):
            model.load_state_dict(checkpoint['model_state_dict'])
        elif any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys()):
            # 从DataParallel保存的模型加载到单卡模型
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:]  # 移除 `module.` 前缀
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            # 单卡模型加载
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(self.device)
        model.eval()
        return model

    def run_decoding(self):
        """
        完整流程：加载GAN生成的潜在数据，用TCN解码器将其还原，并保存最终结果。
        """
        # 1. 加载GAN生成的、处于潜在空间的合成数据
        gan_output_path = self.gan_config.SYNTHETIC_DATA_PATH
        print(f"3. 加载GAN生成的潜在数据 ({gan_output_path})...")
        with open(gan_output_path, 'rb') as f:
            synthetic_latent_data = pickle.load(f)

        # 2. 使用TCN模型的解码器部分进行解码
        print("4. 使用TCN解码器还原数据...")
        with torch.no_grad():
            latent_tensor = torch.FloatTensor(synthetic_latent_data).to(self.device)
            # 直接调用模型的解码器部分
            decoded_data_tensor = self.model.decoder(latent_tensor)

        # 将结果转为Numpy数组
        final_synthetic_data = decoded_data_tensor.cpu().numpy()

        # 3. 保存最终的、可解释的合成数据
        final_output_path = 'final_synthetic_data.pkl'
        print(f"5. 保存最终的、已解码的合成数据到: {final_output_path}...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(final_synthetic_data, f)

        print("\n✅ 合成数据生成和解码流程全部完成！")
        print(f"   最终的合成数据形状: {final_synthetic_data.shape}")


def main():
    try:
        decoder_process = FinalDecoder()
        decoder_process.run_decoding()
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到错误: {e}")
        print("  请确保您已经成功运行了修改后的 'train_encdec.py' 和 'train_gan.py'，")
        print(
            "  并生成了所需的所有模型和数据文件（如 tcn_encoder_decoder_model.pkl, feature_dims.pkl, synthetic_v2_data.pkl）。")
    except Exception as e:
        print(f"\n❌ 执行过程中发生未知错误: {e}")
        raise


if __name__ == "__main__":
    main()