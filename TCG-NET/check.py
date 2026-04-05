import numpy as np
import os

m_path = os.path.join("./cache_encdec", "encdec_M.u8")
M = np.memmap(m_path, dtype="uint8", mode="r")
print(f"Mask 形状: {M.shape}")
print(f"有效观测点数量 (1的个数): {np.sum(M == 1)}")
print(f"缺失点数量 (0的个数): {np.sum(M == 0)}")

if np.sum(M == 1) == 0:
    print("错误：生成的 Mask 全为 0，模型无法计算重构误差！")