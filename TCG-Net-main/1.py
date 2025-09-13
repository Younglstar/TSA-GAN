import dask
import dask.dataframe as dd
import numpy as np


def drop_columns_with_missing_data_optimized(file_path, missing_threshold, blocksize):
    """
    使用 Dask 高效加载 CSV 文件，并删除缺失值比例超过阈值的列。
    此版本经过优化，可减少内存消耗和计算时间。

    参数:
    file_path (str): CSV 文件的路径。
    missing_threshold (float): 缺失值比例的阈值 (0 到 1 之间)。
    blocksize (str): Dask 读取文件时每个数据块的大小。为大文件设置一个合理的值
                     (如 '64MB' 或 '128MB') 对控制内存至关重要。

    返回:
    dask.dataframe.core.DataFrame: 一个新的、已删除相应列的 Dask DataFrame。
    """
    print(f"正在从 '{file_path}' 加载数据，设置块大小为 {blocksize}...")
    # 假设空字符串是缺失值，这在很多真实数据中很常见
    ddf = dd.read_csv(file_path, blocksize=blocksize, assume_missing=True)

    print(f"原始 DataFrame 有 {len(ddf.columns)} 列。")

    # 步骤 1: 定义计算任务
    null_counts = ddf.isnull().sum()
    total_rows = len(ddf)

    # 步骤 2: 使用 dask.compute() 一次性执行元数据计算
    print("正在单次遍历数据以计算缺失值总数和总行数...")
    computed_nulls, computed_rows = dask.compute(null_counts, total_rows)

    if computed_rows == 0:
        print("警告：CSV 文件中没有数据行。")
        return ddf

    # 步骤 3: 确定要删除的列
    print("正在确定需要删除的列...")
    missing_ratios = computed_nulls / computed_rows
    cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()

    if not cols_to_drop:
        print("没有找到需要删除的列。")
        return ddf

    print(f"找到 {len(cols_to_drop)} 列需要删除，因为它们的缺失值比例 > {missing_threshold * 100}%。")
    print("要删除的列名:", cols_to_drop)

    # 步骤 4: 创建最终的 DataFrame（仍然是惰性操作）
    print("正在删除这些列...")
    ddf_cleaned = ddf.drop(columns=cols_to_drop)

    print(f"处理计划已就绪。新的 DataFrame 有 {len(ddf_cleaned.columns)} 列。")

    return ddf_cleaned


if __name__ == '__main__':
    # --- 参数定义 ---
    input_file = './1s_data.csv'
    output_file = './1s_cleaned.csv'  # <--- 1. 定义输出文件的路径
    threshold = 0.90
    mem_blocksize = '1GB'

    # --- 调用函数 ---
    # 这一步仍然是惰性求值，只是构建了计算计划
    cleaned_ddf_optimized = drop_columns_with_missing_data_optimized(
        input_file,
        missing_threshold=threshold,
        blocksize=mem_blocksize
    )

    # --- 保存文件 ---
    # 2. 调用 .to_csv() 方法来执行所有计算并保存结果
    # 这将花费最长的时间，因为它需要处理所有数据
    print(f"\n正在将处理后的数据保存到 '{output_file}'...")
    print("这个过程可能会很长，取决于文件大小和计算资源...")

    # single_file=True: 将所有分区的结果合并成一个单独的 CSV 文件
    # index=False: 不在 CSV 文件中写入 DataFrame 的索引
    cleaned_ddf_optimized.to_csv(output_file, single_file=True, index=False)

    print("文件保存成功！")