# embedding_main.py (已修改，为通用化的 processor 做准备)
import pickle
import os
from embedding_config import EmbeddingConfig
from embedding_processor import CategoricalEmbeddingProcessor
from config import DataConfig  # 引入主配置文件


def main():
    print("--- 开始类别特征嵌入流程 ---")

    # 1. 检查并加载归一化后的数据
    normalized_file = 'normalized_data.pkl'
    if not os.path.exists(normalized_file):
        raise FileNotFoundError(f"{normalized_file} 未找到。请先运行归一化脚本 (normalize_main.py)。")

    print(f"1. 正在从 {normalized_file} 加载数据...")
    with open(normalized_file, 'rb') as f:
        normalized_data = pickle.load(f)

    # 2. 初始化配置和处理器
    # 注意：我们将在下一步修改 CategoricalEmbeddingProcessor 来接收这两个config
    embed_config = EmbeddingConfig()
    data_config = DataConfig()
    processor = CategoricalEmbeddingProcessor(embed_config, data_config)

    # 3. 处理静态类别特征
    print("\n2. 正在处理静态类别特征...")
    static_embeddings = processor.fit_transform_static_features(
        normalized_data['static_data']
    )
    print("静态类别特征处理完成。")

    # 4. 处理时序类别特征
    print("\n3. 正在处理时序类别特征...")
    temporal_embeddings = processor.fit_transform_temporal_features(
        normalized_data['temporal_data']
    )
    print("时序类别特征处理完成。")

    # 5. 保存嵌入模型
    print("\n4. 正在保存嵌入模型...")
    processor.save_models()

    # 6. 保存生成的嵌入向量
    # 我们将嵌入结果与原始数据（数值部分）合并，而不是替换
    # 注意：这个逻辑将在 processor 中实现，这里只负责保存 processor 的输出
    print("\n5. 正在保存生成的嵌入向量...")

    # 我们创建一个新的字典来存储最终结果，而不是覆盖原有数据
    # 这个字典将包含原始的数值数据和新生成的嵌入向量
    final_embedded_data = {
        'static_data': static_embeddings,  # 这将是包含数值和嵌入的DataFrame
        'temporal_data': temporal_embeddings,  # 这也将是包含数值和嵌入的DataFrame
        'missing_patterns': normalized_data.get('missing_patterns'),
        'missing_rates': normalized_data.get('missing_rates')
    }

    with open(embed_config.EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(final_embedded_data, f)

    print("\n--- 类别特征嵌入流程全部完成！ ---")
    print(f"- 嵌入模型已保存至: {embed_config.EMBEDDING_MODELS_FILE}")
    print(f"- 嵌入数据已保存至: {embed_config.EMBEDDINGS_FILE}")

    return final_embedded_data


if __name__ == "__main__":
    try:
        embeddings_data = main()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n在嵌入过程中发生错误: {e}")