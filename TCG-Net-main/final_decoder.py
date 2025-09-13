# file: final_decoder.py (最终正确版，适配CausalVAE)

import torch
import pickle
import numpy as np
import os
from torch.nn.utils.parametrizations import weight_norm  # ✅ 新接口

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
            num_total_features=self.feature_dims['input_dim'],  # ✅ 确保传入
            dropout=self.encdec_config.DROPOUT_RATE
        )

        model_path = self.encdec_config.MODEL_SAVE_PATH
        # ✅ 使用 weights_only=True (避免 FutureWarning & 提升安全性)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

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

    '''def run_decoding(self):
        """
        完整流程：加载GAN生成的潜在数据，用CausalVAE解码器将其还原，并保存最终结果。
        """
        print("\n--- 开始最终解码流程 ---")
        try:
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(self.synthetic_latent_data).to(self.device)
                decoded_data_tensor, _ = self.model.decoder(latent_tensor)
            final_synthetic_data = decoded_data_tensor.cpu().numpy()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("⚠️ GPU内存不足，切换到CPU运行...")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                with torch.no_grad():
                    latent_tensor = torch.FloatTensor(self.synthetic_latent_data).to(self.device)
                    decoded_data_tensor, _ = self.model.decoder(latent_tensor)
                final_synthetic_data = decoded_data_tensor.numpy()
            else:
                raise

        # 保存最终的、解码后的合成数据
        final_output_path = 'final_synthetic_data.pkl'
        print(f"  -> 步骤2: 保存最终的、已解码的合成数据到: {final_output_path}...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(final_synthetic_data, f)

        print("\n✅ 合成数据生成和解码流程全部完成！")
        print(f"   最终的合成数据形状: {final_synthetic_data.shape}")'''

    # final_decoder.py

    # ... (其他部分代码保持不变) ...

    def run_decoding(self):
        """
        (已优化) 完整流程：分批次加载GAN生成的潜在数据，用CausalVAE解码器将其还原，并保存最终结果。
        """
        print("\n--- 开始最终解码流程 ---")

        # 定义一个合理的批次大小，可以根据你的显存大小调整
        # 从一个较小的值开始，如 64 或 128
        BATCH_SIZE = 64

        # 将原始的 numpy 潜在数据转换为 torch Tensor (暂时放在 CPU)
        latent_data_cpu = torch.FloatTensor(self.synthetic_latent_data)
        num_samples = latent_data_cpu.shape[0]

        print(f"总样本数: {num_samples}, 批次大小: {BATCH_SIZE}")

        # 用于收集每个批次的解码结果
        decoded_results = []

        try:
            # 确保模型在GPU上
            self.model.to(self.device)
            print(f"正在使用设备: {self.device}")

            with torch.no_grad():
                for i in range(0, num_samples, BATCH_SIZE):
                    # 1. 取出一小批数据
                    batch_latent = latent_data_cpu[i: i + BATCH_SIZE]

                    # 2. **只将这一小批数据移动到GPU**
                    batch_latent_gpu = batch_latent.to(self.device)

                    # 3. 在GPU上对这一小批数据进行解码
                    decoded_batch_gpu, _ = self.model.decoder(batch_latent_gpu)

                    # 4. **将解码结果移回CPU**，以便释放显存给下一个批次
                    decoded_results.append(decoded_batch_gpu.cpu())

                    # 打印进度
                    print(f"  已处理批次 {i // BATCH_SIZE + 1} / {-(-num_samples // BATCH_SIZE)}")

            # 将所有在CPU上的小批次结果拼接成一个完整的大张量
            final_synthetic_data_tensor = torch.cat(decoded_results, dim=0)
            final_synthetic_data = final_synthetic_data_tensor.numpy()

        except RuntimeError as e:
            # 如果即使分批处理也内存不足 (例如BATCH_SIZE太大)，打印错误
            if "CUDA out of memory" in str(e):
                print(f"❌ GPU内存不足！即使批次大小为 {BATCH_SIZE}。")
                print("   请尝试进一步减小 BATCH_SIZE 的值。")
                raise
            else:
                raise

        # 保存最终的、解码后的合成数据
        final_output_path = 'final_synthetic_data.pkl'
        print(f"\n-> 步骤2: 保存最终的、已解码的合成数据到: {final_output_path}...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(final_synthetic_data, f)

        print("\n✅ 合成数据生成和解码流程全部完成！")
        print(f"   最终的合成数据形状: {final_synthetic_data.shape}")


# ... (main 函数的调用部分保持不变) ...


def main():
    try:
        decoder_process = FinalDecoder()
        decoder_process.run_decoding()
    except Exception as e:
        print(f"\n❌ 执行过程中发生未知错误: {e}")
        raise


if __name__ == "__main__":
    main()
