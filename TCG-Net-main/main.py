# main.py (已修改，以适应通用化的数据加载流程)

import pandas as pd
import numpy as np
from config import DataConfig
from data_loader import DataLoader
from missing_pattern_analyzer import MissingPatternAnalyzer
import pickle


def main():
    # 1. 初始化配置和数据加载器
    config = DataConfig()
    loader = DataLoader(config)

    # 2. 加载和拆分数据
    print("--- 步骤 1: 加载并预处理数据 ---")
    static_data, temporal_data = loader.split_static_temporal()

    # 3. 分析缺失模式
    # 注意：我们将在下一步修改 MissingPatternAnalyzer 以适应新的数据格式
    print("\n--- 步骤 2: 分析数据缺失模式 ---")
    # 将 config 传入，以便 analyzer 获取 ID 列名
    analyzer = MissingPatternAnalyzer(static_data, temporal_data, config)
    missing_rates_static, missing_rates_temporal = analyzer.calculate_missing_rates()
    missing_patterns = analyzer.generate_missing_patterns()

    # 4. 打印基本统计信息 (已通用化)
    subject_id_col = config.IDDATA_COLS[0] if config.IDDATA_COLS else 'subject'
    print("\n--- 基本统计信息 ---")
    print(f"独立主体 ({subject_id_col}) 的数量: {len(static_data)}")

    if not missing_rates_static.empty:
        print("\n静态特征的缺失率:")
        print(missing_rates_static)
    else:
        print("\n没有静态特征需要分析缺失率。")

    if not missing_rates_temporal.empty:
        print("\n时序特征的缺失率:")
        # 由于 temporal_data 不再是字典，我们直接打印其缺失率
        print(missing_rates_temporal)
    else:
        print("\n没有时序特征需要分析缺失率。")

    # 5. 保存处理后的数据
    print(f"\n--- 步骤 3: 保存处理后的数据 ---")
    processed_data = {
        'static_data': static_data,
        'temporal_data': temporal_data,
        'missing_patterns': missing_patterns,
        'missing_rates': {
            'static': missing_rates_static,
            'temporal': missing_rates_temporal
        }
    }

    with open(config.PROCESSED_FILE, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"\n处理完成的数据已保存至: {config.PROCESSED_FILE}")
    return processed_data


if __name__ == "__main__":
    # 捕获并打印更清晰的错误信息
    try:
        processed_data = main()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n在预处理过程中发生错误: {e}")