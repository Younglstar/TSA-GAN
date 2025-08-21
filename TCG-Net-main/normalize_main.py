# normalize_main.py (已修改，为通用化的 processor 做准备)
import pickle
import os
from normalizer_config import NormalizerConfig
from normalizer_processor import DataNormalizationProcessor
from config import DataConfig  # 引入主配置文件


def main():
    print("--- 开始数据归一化流程 ---")

    # 1. 检查并加载预处理数据
    processed_file = 'processed_data.pkl'
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"{processed_file} 未找到。请先运行主预处理脚本 (main.py)。")

    print(f"1. 正在从 {processed_file} 加载预处理数据...")
    with open(processed_file, 'rb') as f:
        processed_data = pickle.load(f)

    # 2. 初始化配置和处理器
    # 注意：我们将在下一步修改 DataNormalizationProcessor 来接收这两个config
    norm_config = NormalizerConfig()
    # 从主配置文件获取特征定义，这将使 normalizer 更具动态性
    data_config = DataConfig()
    processor = DataNormalizationProcessor(norm_config, data_config)

    # 3. 归一化静态特征
    print("\n2. 正在归一化静态特征...")
    normalized_static = processor.normalize_static_features(processed_data['static_data'])
    print("静态特征归一化完成。")

    # 4. 归一化时序特征
    print("\n3. 正在归一化时序特征...")
    # 传入的 temporal_data 已经是我们需要的单个 DataFrame 格式
    normalized_temporal = processor.normalize_temporal_features(processed_data['temporal_data'])
    print("时序特征归一化完成。")

    # 5. 保存归一化参数
    print("\n4. 正在保存归一化参数...")
    processor.save_normalization_params()

    # 6. 保存归一化后的数据
    print("\n5. 正在保存归一化后的数据...")
    normalized_data = {
        'static_data': normalized_static,
        'temporal_data': normalized_temporal,
        # 保留缺失模式和比率，以备后续使用
        'missing_patterns': processed_data.get('missing_patterns'),
        'missing_rates': processed_data.get('missing_rates')
    }

    output_file = 'normalized_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(normalized_data, f)

    print("\n--- 归一化流程全部完成！ ---")
    print(f"- 归一化数据已保存至: {output_file}")
    print(f"- 归一化参数已保存至: {norm_config.NORMALIZATION_PARAMS_FILE}")

    return normalized_data


if __name__ == "__main__":
    try:
        normalized_data = main()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n在归一化过程中发生错误: {e}")