from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np


def drop_columns_with_missing_data_optimized(file_path, missing_threshold=0.5, blocksize='64MB'):
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
    ddf = dd.read_csv(
        file_path,
        blocksize=blocksize,
        assume_missing=True,
        dtype={'drugname': 'object',
               'responsetotherapy': 'object',
               'treatment': 'object',
               'drugrate': 'object',
               'labresulttext': 'object',
               'loadingdose': 'object',
               'antibiotic': 'object',
               'sensitivitylevel': 'object',
               }
    )

    print(f"原始 DataFrame 有 {len(ddf.columns)} 列。")

    # 步骤 1: 定义两个独立的计算任务（此时不执行）
    # 任务一：计算每列的缺失值总数
    null_counts = ddf.isnull().sum()
    # 任务二：计算 DataFrame 的总行数
    total_rows = len(ddf)

    # 步骤 2: 使用 dask.compute() 一次性执行所有计算
    # 这是关键的优化步骤。Dask会构建一个单一的计算图，
    # 只需遍历一次数据即可同时完成 null_counts 和 total_rows 的计算。
    print("正在单次遍历数据以计算缺失值总数和总行数...")
    computed_nulls, computed_rows = dask.compute(null_counts, total_rows)

    # 检查总行数是否为零，以避免除零错误
    if computed_rows == 0:
        print("警告：CSV 文件中没有数据行。")
        return ddf

    # 步骤 3: 在内存中计算比例并确定要删除的列
    # computed_nulls 和 computed_rows 都已经是计算好的具体值（Pandas Series 和 int），
    # 接下来的操作在内存中进行，速度非常快。
    print("正在确定需要删除的列...")
    missing_ratios = computed_nulls / computed_rows
    cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()

    if not cols_to_drop:
        print("没有找到需要删除的列。")
        return ddf

    print(f"找到 {len(cols_to_drop)} 列需要删除，因为它们的缺失值比例 > {missing_threshold * 100}%。")
    print("要删除的列名:", cols_to_drop)

    # 步骤 4: 从 Dask DataFrame 中删除这些列（惰性操作）
    print("正在删除这些列...")
    ddf_cleaned = ddf.drop(columns=cols_to_drop)

    print(f"处理完成。新的 DataFrame 有 {len(ddf_cleaned.columns)} 列。")

    return ddf_cleaned


if __name__ == '__main__':
    # 同样使用之前的示例进行演示

    p = Path('eicu-collaborative-research-database-2.0')
    for item in p.iterdir():
        if item.is_file():
            print(f"  [文件] {item.name}")
            sample_csv_path='eicu-collaborative-research-database-2.0/'+item.name
            cleaned_ddf_optimized = drop_columns_with_missing_data_optimized(
                sample_csv_path,
                missing_threshold=0.6,
                blocksize='1GB'  # 对于小文件，这个值很小，但演示了用法
            )

            output_path = 'eicu_cleandata/'+sample_csv_path
            print(f"\n正在将结果保存到 '{output_path}'...")
            cleaned_ddf_optimized.to_csv(output_path, single_file=True, index=False)
            print("保存完成！")

    # 调用优化后的函数

