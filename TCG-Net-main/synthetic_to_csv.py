# file: synthetic_to_csv.py (最终正确版，适配CausalVAE)

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os

# --- 核心修改：导入我们所有最新的模块和配置文件 ---
from encdec_model import CausalVAE  # <-- 导入正确的CausalVAE模型
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig
from embedding_config import EmbeddingConfig
from config import DataConfig


class SyntheticDataConverter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encdec_conf = EncoderDecoderConfig()
        self.gan_conf = GANConfig()
        self.embed_conf = EmbeddingConfig()
        self.data_conf = DataConfig()
        self.output_dir = 'final_synthetic_output'
        os.makedirs(self.output_dir, exist_ok=True)

        print("📊 正在加载所有必需的数据和模型...")
        try:
            with open('feature_dims.pkl', 'rb') as f:
                self.feature_dims = pickle.load(f)
            with open('processed_data.pkl', 'rb') as f:
                self.processed_data = pickle.load(f)
            with open('normalization_params.pkl', 'rb') as f:
                self.norm_params = pickle.load(f)
            with open(self.embed_conf.EMBEDDING_MODELS_FILE, 'rb') as f:
                self.embed_models = pickle.load(f)
            with open(self.gan_conf.SYNTHETIC_DATA_PATH, 'rb') as f:
                self.synthetic_latent = pickle.load(f)
            # --- 核心修改：调用新的加载函数 ---
            self.decoder_model = self._load_causal_vae_model()
            self.original_df = pd.read_csv(self.data_conf.INPUT_FILE)
            print("所有文件加载成功！")
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到错误: {e}");
            raise

    def _load_causal_vae_model(self) -> CausalVAE:
        """
        (已修改) 初始化并加载我们新的CausalVAE模型。
        """
        print(f"  -> 正在加载CausalVAE模型 ({self.encdec_conf.MODEL_SAVE_PATH})...")

        # --- 核心修改：实例化正确的 CausalVAE 模型 ---
        model = CausalVAE(
            input_dim=self.feature_dims['input_dim'],
            output_dim=self.feature_dims['input_dim'],
            sequence_length=self.feature_dims['sequence_length'],
            tcn_channels=self.encdec_conf.TCN_CHANNELS,
            latent_dim=self.encdec_conf.LATENT_DIM,
            num_total_features=self.feature_dims['input_dim'],
            dropout=self.encdec_conf.DROPOUT_RATE
        )

        checkpoint = torch.load(self.encdec_conf.MODEL_SAVE_PATH, map_location=self.device)
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

        model.to(self.device);
        model.eval();
        return model

    def _decode_latent_data(self) -> np.ndarray:
        """使用CausalVAE解码器还原数据。"""
        print("\n   - 步骤1: 使用CausalVAE解码器还原数据...")
        with torch.no_grad():
            latent_tensor = torch.FloatTensor(self.synthetic_latent).to(self.device)
            # CausalVAE的解码器返回 (recon_data, recon_mask)，我们只需要前者
            decoded_tensor, _ = self.decoder_model.decoder(latent_tensor)
        return decoded_tensor.cpu().numpy()

    def _convert_tensor_to_dataframe(self, decoded_tensor: np.ndarray) -> pd.DataFrame:
        """将解码后的3D张量转换回原始的DataFrame格式 (此函数逻辑正确，保持不变)"""
        print("   - 步骤2: 将解码后的张量逆向工程为DataFrame...")
        real_static_df = self.processed_data['static_data']
        static_dim_real = real_static_df.shape[1]
        decoded_static_features = decoded_tensor[:, 0, :static_dim_real]
        synth_df = pd.DataFrame()
        current_pos = 0
        static_cols_numerical = real_static_df.select_dtypes(include=np.number).columns
        static_cols_categorical = [col for col in self.embed_models['label_encoders'].keys() if
                                   col in real_static_df.columns]

        print("    -> 正在还原数值型特征...")
        for col in static_cols_numerical:
            if col in self.norm_params:
                params = self.norm_params[col]
                synth_df[col] = decoded_static_features[:, current_pos] * params['std'] + params['mean']
                current_pos += 1

        print("    -> 正在还原类别型特征...")
        for col in static_cols_categorical:
            model_emb = self.embed_models['embedding_models'][col]
            le = self.embed_models['label_encoders'][col]
            original_embeddings = model_emb.embedding.weight.detach().cpu().numpy()
            embed_dim = original_embeddings.shape[1]
            synth_embeddings = decoded_static_features[:, current_pos:current_pos + embed_dim]
            if synth_embeddings.shape[1] != embed_dim:
                raise ValueError(f"特征 '{col}' 的维度不匹配！")
            distances = np.dot(synth_embeddings, original_embeddings.T)
            indices = np.argmax(distances, axis=1)
            synth_df[col] = le.inverse_transform(indices)
            current_pos += embed_dim

        synth_df['DOSSIER_HASH'] = [f'SYNTH_{i:06d}' for i in range(len(synth_df))]
        original_cols = [col for col in self.original_df.columns if col in synth_df.columns]
        return synth_df.reindex(columns=original_cols)

    def generate_and_save_csv(self):
        decoded_tensor = self._decode_latent_data()
        synthetic_df = self._convert_tensor_to_dataframe(decoded_tensor)
        output_path = os.path.join(self.output_dir, 'synthetic_data_final.csv')
        print(f"   - 步骤3: 正在将最终的DataFrame保存为CSV文件到: {output_path}")
        synthetic_df.to_csv(output_path, index=False)
        print("\n✅ 流程完成！")


def main():
    try:
        converter = SyntheticDataConverter()
        converter.generate_and_save_csv()
    except Exception as e:
        print(f"\n❌ 主程序执行时出错: {e}")
        raise


if __name__ == "__main__":
    main()