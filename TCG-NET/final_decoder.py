# file: final_decoder_argparse.py
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch

from encdec_model import CausalVAE
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig


class FinalDecoder:
    def __init__(
        self,
        synthetic_pkl: Optional[str] = None,
        out_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        feature_dims_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encdec_config = EncoderDecoderConfig()
        self.gan_config = GANConfig()

        self.synthetic_pkl = synthetic_pkl or getattr(self.gan_config, "SYNTHETIC_DATA_PATH", "synthetic_final_data.pkl")
        self.model_path = model_path or getattr(self.encdec_config, "MODEL_SAVE_PATH", "cvae_model.pkl")
        self.feature_dims_path_override = feature_dims_path

        self.output_dir = out_dir or os.path.join(
            getattr(self.encdec_config, "CACHE_DIR", "."),
            "decode_outputs",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.decoded_data_path = os.path.join(self.output_dir, "decoded_data.pkl")
        self.mask_logits_path = os.path.join(self.output_dir, "mask_logits.pkl")
        self.mask_probs_path = os.path.join(self.output_dir, "mask_probs.pkl")
        self.mask_binary_path = os.path.join(self.output_dir, "mask_binary.pkl")
        self.meta_path = os.path.join(self.output_dir, "decode_meta.json")

        print("正在加载解码所需的模型与数据...")
        self.feature_dims = self._load_feature_dims()
        self.model = self._load_causal_vae_model()
        self.synthetic_latent_data = self._load_synthetic_latent()

        print("加载完成。")
        print(f"  device: {self.device}")
        print(f"  latent path: {self.synthetic_pkl}")
        print(f"  latent shape: {self.synthetic_latent_data.shape}")
        print(f"  decoder ckpt: {self.model_path}")
        print(f"  decoder output dir: {self.output_dir}")

    def _load_feature_dims(self) -> Dict[str, Any]:
        if self.feature_dims_path_override:
            candidate_paths = [self.feature_dims_path_override]
        else:
            cache_dir = getattr(self.encdec_config, "CACHE_DIR", ".")
            feature_dims_file = getattr(self.encdec_config, "FEATURE_DIMS_FILE", "feature_dims.pkl")
            candidate_paths = [
                os.path.join(cache_dir, feature_dims_file),
                feature_dims_file,
            ]

        for path in candidate_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                print(f"  -> feature_dims loaded from: {path}")
                return obj

        raise FileNotFoundError(f"未找到 feature_dims.pkl。尝试过: {candidate_paths}")

    def _load_checkpoint_payload(self, path: str, map_location: torch.device):
        load_errors = {}

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            return payload
        except Exception as e:
            load_errors["pickle"] = repr(e)

        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            return payload
        except Exception as e:
            load_errors["torch.load"] = repr(e)

        raise RuntimeError(
            f"无法加载 checkpoint: {path}\n"
            f"pickle 错误: {load_errors.get('pickle')}\n"
            f"torch.load 错误: {load_errors.get('torch.load')}"
        )

    def _normalize_state_dict(self, saved_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if any(key.startswith("module.") for key in saved_state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in saved_state_dict.items():
                new_state_dict[k[7:]] = v
            return new_state_dict
        return saved_state_dict

    def _load_causal_vae_model(self) -> CausalVAE:
        input_dim = int(self.feature_dims["input_dim"])
        sequence_length = int(self.feature_dims["sequence_length"])
        num_total_features = int(self.feature_dims.get("num_total_features", input_dim))

        model = CausalVAE(
            input_dim=input_dim,
            output_dim=input_dim,
            sequence_length=sequence_length,
            tcn_channels=self.encdec_config.TCN_CHANNELS,
            latent_dim=self.encdec_config.LATENT_DIM,
            num_total_features=num_total_features,
            dropout=self.encdec_config.DROPOUT_RATE,
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到 encoder-decoder checkpoint: {self.model_path}")

        print(f"  -> loading CausalVAE checkpoint: {self.model_path}")
        checkpoint = self._load_checkpoint_payload(self.model_path, self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved_state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            saved_state_dict = checkpoint
        else:
            raise TypeError(f"checkpoint 类型异常: {type(checkpoint)}")

        saved_state_dict = self._normalize_state_dict(saved_state_dict)
        model.load_state_dict(saved_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_synthetic_latent(self) -> np.ndarray:
        if not os.path.exists(self.synthetic_pkl):
            raise FileNotFoundError(f"未找到 GAN synthetic latent 文件: {self.synthetic_pkl}")

        print(f"  -> loading synthetic latent: {self.synthetic_pkl}")
        with open(self.synthetic_pkl, "rb") as f:
            latent = pickle.load(f)

        latent = np.asarray(latent, dtype=np.float32)

        if latent.ndim != 2:
            raise ValueError(f"synthetic latent 必须是二维数组 [N, D]，当前 shape={latent.shape}")

        expected_dim = int(self.encdec_config.LATENT_DIM)
        if latent.shape[1] != expected_dim:
            raise ValueError(
                f"latent 维度不匹配：synthetic latent dim={latent.shape[1]} "
                f"vs encdec LATENT_DIM={expected_dim}"
            )

        return latent

    def _decode_in_batches(self, latent_np: np.ndarray, batch_size: int = 256, mask_threshold: float = 0.5):
        num_samples = latent_np.shape[0]
        num_batches = math.ceil(num_samples / batch_size)

        decoded_data_chunks = []
        mask_logits_chunks = []
        mask_probs_chunks = []
        mask_binary_chunks = []

        print(f"\n开始解码：N={num_samples}, batch_size={batch_size}, num_batches={num_batches}")
        print(f"使用设备: {self.device}")

        latent_cpu = torch.from_numpy(latent_np)

        try:
            self.model.to(self.device)
            self.model.eval()

            with torch.no_grad():
                for bi in range(num_batches):
                    s = bi * batch_size
                    e = min((bi + 1) * batch_size, num_samples)

                    z_batch = latent_cpu[s:e].to(self.device, non_blocking=True)
                    decoded_batch, mask_logits_batch = self.model.decoder(z_batch)

                    mask_probs_batch = torch.sigmoid(mask_logits_batch)
                    mask_binary_batch = (mask_probs_batch > mask_threshold).to(decoded_batch.dtype)

                    decoded_data_chunks.append(decoded_batch.cpu())
                    mask_logits_chunks.append(mask_logits_batch.cpu())
                    mask_probs_chunks.append(mask_probs_batch.cpu())
                    mask_binary_chunks.append(mask_binary_batch.cpu())

                    if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                        print(f"  已处理批次 {bi + 1}/{num_batches}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(
                    f"GPU 显存不足，即使使用 batch_size={batch_size} 仍失败。"
                    f"请继续减小 batch_size。原始错误: {e}"
                )
            raise

        decoded_data = torch.cat(decoded_data_chunks, dim=0).numpy()
        mask_logits = torch.cat(mask_logits_chunks, dim=0).numpy()
        mask_probs = torch.cat(mask_probs_chunks, dim=0).numpy()
        mask_binary = torch.cat(mask_binary_chunks, dim=0).numpy()

        return decoded_data, mask_logits, mask_probs, mask_binary

    def run_decoding(self, batch_size: int = 256, mask_threshold: float = 0.5):
        print("\n--- 开始最终 decode 流程 ---")
        decoded_data, mask_logits, mask_probs, mask_binary = self._decode_in_batches(
            latent_np=self.synthetic_latent_data,
            batch_size=batch_size,
            mask_threshold=mask_threshold,
        )

        with open(self.decoded_data_path, "wb") as f:
            pickle.dump(decoded_data, f)
        with open(self.mask_logits_path, "wb") as f:
            pickle.dump(mask_logits, f)
        with open(self.mask_probs_path, "wb") as f:
            pickle.dump(mask_probs, f)
        with open(self.mask_binary_path, "wb") as f:
            pickle.dump(mask_binary, f)

        meta = {
            "encoder_model_path": self.model_path,
            "synthetic_latent_path": self.synthetic_pkl,
            "feature_dims_path_override": self.feature_dims_path_override,
            "latent_shape": list(self.synthetic_latent_data.shape),
            "decoded_data_shape": list(decoded_data.shape),
            "mask_logits_shape": list(mask_logits.shape),
            "mask_probs_shape": list(mask_probs.shape),
            "mask_binary_shape": list(mask_binary.shape),
            "latent_dim_expected": int(self.encdec_config.LATENT_DIM),
            "input_dim": int(self.feature_dims["input_dim"]),
            "sequence_length": int(self.feature_dims["sequence_length"]),
            "num_total_features": int(self.feature_dims.get("num_total_features", self.feature_dims["input_dim"])),
            "batch_size": int(batch_size),
            "mask_threshold": float(mask_threshold),
            "output_dir": self.output_dir,
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("\ndecode 完成。输出文件：")
        print(f"  decoded_data : {self.decoded_data_path}")
        print(f"  mask_logits  : {self.mask_logits_path}")
        print(f"  mask_probs   : {self.mask_probs_path}")
        print(f"  mask_binary  : {self.mask_binary_path}")
        print(f"  meta         : {self.meta_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description="最终解码脚本：把 GAN synthetic latent 解码成 decoded tensors")
    parser.add_argument("--synthetic_pkl", type=str, default=None, help="GAN 生成的 synthetic latent pkl 路径")
    parser.add_argument("--out_dir", type=str, default=None, help="decode 输出目录")
    parser.add_argument("--model_path", type=str, default=None, help="encoder-decoder checkpoint 路径")
    parser.add_argument("--feature_dims_path", type=str, default=None, help="feature_dims.pkl 路径")
    parser.add_argument("--batch_size", type=int, default=256, help="decode batch size")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="mask 二值化阈值")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu；默认自动检测")
    return parser


def main():
    args = build_argparser().parse_args()
    decoder = FinalDecoder(
        synthetic_pkl=args.synthetic_pkl,
        out_dir=args.out_dir,
        model_path=args.model_path,
        feature_dims_path=args.feature_dims_path,
        device=args.device,
    )
    decoder.run_decoding(batch_size=args.batch_size, mask_threshold=args.mask_threshold)


if __name__ == "__main__":
    main()
