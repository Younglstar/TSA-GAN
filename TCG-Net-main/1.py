import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm  # 用于显示美观的进度条

# --- 1. 定义输入和输出文件名 ---
pkl_file_path = 'cvae_encoded_data.pkl'
csv_file_path = 'final_synthetic_data_long_format.csv'

print(f"开始转换文件: {pkl_file_path}")

# --- 2. 加载 .pkl 文件 ---
try:
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    # 确认数据是 NumPy 数组
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("Pickle文件中的数据不是一个3维NumPy数组!")

    # --- 3. 将3D数据转换为适合DataFrame的格式 ---
    n_samples, n_timesteps, n_features = data.shape
    print(f"数据形状: {n_samples}个样本, {n_timesteps}个时间步, {n_features}个特征")

    # 创建一个列表来存储每一行的数据
    rows_list = []

    # 使用tqdm来可视化处理进度
    for sample_id in tqdm(range(n_samples), desc="处理样本中"):
        for time_step in range(n_timesteps):
            # 获取当前时间步的所有特征值
            features = data[sample_id, time_step, :]
            # 创建行数据：样本ID, 时间步ID, 特征值...
            row = [sample_id, time_step] + list(features)
            rows_list.append(row)

    # --- 4. 创建列名 ---
    feature_columns = [f'feature_{i + 1}' for i in range(n_features)]
    columns = ['sample_id', 'time_step'] + feature_columns

    # --- 5. 创建Pandas DataFrame并保存为CSV ---
    print("正在创建DataFrame...")
    df = pd.DataFrame(rows_list, columns=columns)

    print(f"正在将DataFrame保存到: {csv_file_path}")
    # 使用 index=False 来避免在CSV文件中写入多余的行号索引
    df.to_csv(csv_file_path, index=False)

    print("\n转换成功！")
    print("生成的CSV文件预览:")
    print(df.head())

except FileNotFoundError:
    print(f"错误: 文件 '{pkl_file_path}' 未找到。")
except Exception as e:
    print(f"发生错误: {e}")