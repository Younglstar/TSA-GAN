'''import pickle
import pandas as pd
import numpy as np
import os


def check_missing_values_in_pkl(file_path: str):
    """
    加载一个 pickle (.pkl) 文件，并检查其中包含的数据是否存在缺失值 (NaN)。
    该函数可以处理 Pandas DataFrame, NumPy 数组, 以及包含这两者的字典。

    参数:
    file_path (str): 要检查的 .pkl 文件的路径。
    """
    print("-" * 50)
    print(f"正在检查文件: '{file_path}'")

    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        print("-" * 50)
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("文件加载成功。正在分析数据结构...")

        # --- 核心检查逻辑 ---
        inspect_data_object(data)

    except (pickle.UnpicklingError, EOFError) as e:
        print(f"错误：无法加载 pickle 文件。文件可能已损坏或为空。({e})")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print("-" * 50)


def inspect_data_object(data_obj, name="顶层对象"):
    """
    递归地检查数据对象（DataFrame, NumPy数组, 字典）中的缺失值。
    """
    # 1. 检查是否为 Pandas DataFrame
    if isinstance(data_obj, pd.DataFrame):
        print(f"\n分析对象 '{name}' (类型: Pandas DataFrame)...")
        missing_count = data_obj.isnull().sum().sum()
        if missing_count > 0:
            print(f"  - 发现 {missing_count} 个缺失值 (NaN)！")
            print("  - 各列的缺失值数量:")
            # 只显示包含缺失值的列
            missing_details = data_obj.isnull().sum()
            print(missing_details[missing_details > 0])
        else:
            print("  - 未发现缺失值。")

    # 2. 检查是否为 NumPy 数组
    elif isinstance(data_obj, np.ndarray):
        print(f"\n分析对象 '{name}' (类型: NumPy Array)...")
        missing_count = np.isnan(data_obj).sum()
        if missing_count > 0:
            print(f"  - 发现 {missing_count} 个缺失值 (NaN)！")
        else:
            print("  - 未发现缺失值。")

    # 3. 检查是否为字典
    elif isinstance(data_obj, dict):
        print(f"\n分析对象 '{name}' (类型: 字典)...")
        print("将递归检查字典中的每一个值:")
        if not data_obj:
            print("  - 字典为空。")
            return
        for key, value in data_obj.items():
            # 递归调用自身来检查字典中的每个元素
            inspect_data_object(value, name=f"字典键 '{key}'")

    # 4. 其他数据类型
    else:
        print(f"\n跳过对象 '{name}' (类型: {type(data_obj).__name__})，非 DataFrame, NumPy Array 或字典。")


if __name__ == "__main__":
    # --- 如何使用 ---
    # 1. 将此脚本与你的 .pkl 文件放在同一个目录下。
    # 2. 在下面的列表中填入你想要检查的 .pkl 文件的文件名。

    files_to_check = [
        'processed_data.pkl',
        'normalized_data.pkl',
        # 在你的项目中，嵌入后的数据文件名可能叫 'embedded_data.pkl' 或 'embeddings.pkl'
        # 请根据实际情况修改
        'categorical_embeddings.pkl'
    ]

    print("开始对项目中的关键 Pickle 文件进行缺失值检查...")

    for file in files_to_check:
        check_missing_values_in_pkl(file)

    print("\n检查完成。")'''
import pandas as pd


def filter_csv_by_date(input_filename, output_filename, date_column='localminute'):
    """
    读取CSV文件，并删除指定日期之后的所有记录。

    参数:
    input_filename (str): 输入的CSV文件名。
    output_filename (str): 输出的CSV文件名。
    date_column (str): 包含日期时间的列名。
    """
    try:
        # 1. 读取CSV文件
        print(f"正在读取文件: {input_filename}...")
        df = pd.read_csv(input_filename)

        # 2. 将日期列转换为datetime对象，以便进行比较
        # pandas可以自动处理 "YYYY-MM-DD HH:MM:SS-ZZ" 这种格式
        # errors='coerce' 会将无法转换的格式变为NaT(Not a Time)，避免程序出错
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # 3. 删除转换失败的行 (如果存在)
        original_rows = len(df)
        df.dropna(subset=[date_column], inplace=True)
        if original_rows > len(df):
            print(f"警告: 移除了 {original_rows - len(df)} 行，因为日期格式无法识别。")

        # 4. 定义筛选条件：保留所有小于 '2019-10-16' 的记录
        # 这会包含10月15日当天所有时间点的数据
        filter_condition = df[date_column] < '2019-10-16'

        # 5. 应用筛选条件，创建新的DataFrame
        filtered_df = df[filter_condition]

        # 6. 将结果保存到新的CSV文件
        # index=False 表示在保存时不额外添加一列行号
        filtered_df.to_csv(output_filename, index=False, encoding='utf-8')

        print("\n处理完成！")
        print(f"原始数据共有 {original_rows} 行。")
        print(f"筛选后剩余 {len(filtered_df)} 行。")
        print(f"结果已保存至: {output_filename}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_filename}'。请检查文件名和路径是否正确。")
    except KeyError:
        print(f"错误：在CSV文件中找不到名为 '{date_column}' 的列。请检查列名是否正确。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


# --- 如何使用 ---
if __name__ == "__main__":
    # 设置你的输入文件名和希望输出的文件名
    input_csv = '1s_data_newyork_file4.csv'  # <--- 请将这里替换成你的文件名
    output_csv = '1s_data.csv'  # <--- 这是处理后生成的文件名

    filter_csv_by_date(input_csv, output_csv)