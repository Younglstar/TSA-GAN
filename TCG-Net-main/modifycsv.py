import pandas as pd
import os


def update_id(row):
    """
    一个应用于DataFrame每一行的函数，用于安全地更新 dataid。
    """
    data_id = row['dataid']
    date_str = row['localminute']

    try:
        # --- 核心修改在这里 ---
        # 我们明确地告诉pandas日期的格式，包括时区。
        # %z 用于解析像 -0600 或 +0800 这样的时区偏移。
        # pandas的to_datetime可以智能处理像 "-06" 这样的缩写格式。
        date_format = '%Y-%m-%d %H:%M:%S%z'

        # 注意：不再需要 utc=True，因为格式字符串中已经包含了时区信息
        dt_object = pd.to_datetime(date_str, format=date_format)

        # 格式化月份和日期为两位数 'MMDD'
        month_day_str = dt_object.strftime('%m%d%H%M')
        data_id=int(data_id)
        # 将原始ID和日期字符串拼接，并转换为整数
        new_id = float(str(data_id) + month_day_str)

        return new_id

    except (ValueError, TypeError):
        # 如果日期转换失败，保持原始ID不变
        # print(f"警告：无法使用格式 '{date_format}' 解析日期 '{date_str}'")
        return data_id


def create_new_dataid_robust(input_file: str, output_file: str):
    """
    (最终修正版) 读取CSV，并根据'dataid'和'local_15min'创建一个新的ID。
    """
    print(f"--- 开始处理文件: {input_file} (最终版) ---")

    try:
        print("1. 正在加载CSV文件...")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到。")
        return

    required_columns = ['dataid', 'localminute']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：CSV文件中必须包含 {required_columns} 列。")
        return

    # 在应用之前，确保 'local_15min' 列是字符串类型，避免潜在的类型错误
    df['localminute'] = df['localminute'].astype(str)

    print("2. 正在逐行生成新的 DataID...")
    df['dataid'] = df.apply(update_id, axis=1)

    print("3. 正在保存修改后的文件...")
    try:
        df.to_csv(output_file, index=False)
        print(f"--- 处理完成！修改后的文件已保存至: {output_file} ---")
    except Exception as e:
        print(f"错误：保存文件时出错。{e}")


if __name__ == '__main__':
    # --- 请在这里修改你的输入和输出文件名 ---
    input_csv_path = '1s_cleaned.csv'
    output_csv_path = '1s_modified.csv'

    create_new_dataid_robust(input_csv_path, output_csv_path)

    # (可选) 读取并打印修改后的文件内容进行验证
    if os.path.exists(output_csv_path):
        print("\n--- 修改后的数据预览 ---")
        try:
            df_modified = pd.read_csv(output_csv_path)
            # 检查是否有任何 dataid 被成功修改了
            if df_modified['dataid'].dtype != 'int64' or df_modified['dataid'].max() > 10000:  # 假设原始ID不会这么大
                print("检测到 dataid 列已被成功修改。")
            else:
                print("警告：dataid 列似乎没有被修改。请检查日期格式是否完全匹配。")
            print(df_modified.head())
        except Exception as e:
            print(f"无法读取输出文件进行验证: {e}")