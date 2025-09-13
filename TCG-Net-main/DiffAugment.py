import pandas as pd
import glob
import os


def merge_csv_files(path_to_csvs, output_filename):
    """
    合并指定路径下的所有 CSV 文件到一个文件中。

    参数:
    path_to_csvs (str): 存放多个 CSV 文件的文件夹路径。
    output_filename (str): 合并后输出的 CSV 文件名（包含路径）。
    """
    # 1. 检查输入路径是否存在
    if not os.path.isdir(path_to_csvs):
        print(f"错误：提供的路径 '{path_to_csvs}' 不存在或不是一个文件夹。")
        return

    # 2. 匹配路径下所有的 csv 文件
    # 使用 glob 模块找到所有以 .csv 结尾的文件
    csv_files = glob.glob(os.path.join(path_to_csvs, "*.csv"))

    if not csv_files:
        print(f"错误：在路径 '{path_to_csvs}' 下没有找到任何 CSV 文件。")
        return

    print(f"找到以下文件进行合并: {len(csv_files)} 个")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # 3. 使用列表推导式读取所有 CSV 文件为 DataFrame
    # 这种方式比传统的 for 循环更简洁高效
    try:
        df_list = [pd.read_csv(file) for file in csv_files]
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 4. 合并所有的 DataFrame
    # ignore_index=True 会重新生成索引，避免合并后出现重复的索引值
    combined_df = pd.concat(df_list, ignore_index=True)

    # 5. 将合并后的 DataFrame 保存为新的 CSV 文件
    # index=False 表示在输出的 CSV 文件中不包含 DataFrame 的索引列
    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"\n成功！所有 CSV 文件已合并到 '{output_filename}'")
        print(f"合并后的数据共有 {len(combined_df)} 行。")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 只需要修改下面这两个变量
    # 变量1: 存放你的CSV文件的文件夹路径
    # 例如 'C:/Users/YourUser/Desktop/MyCSVs' 或 './data/reports'
    folder_path = './1min'  # 请替换为你的文件夹路径
    print(111)
    # 变量2: 合并后输出的文件名
    output_file = '1min.csv'
    L1=['air1','air2','air3','airwindowunit1','aquarium1','bathroom1','bathroom2','bedroom1',
        'bedroom2','bedroom3','bedroom4','bedroom5','battery1','car1','car2','circpump1',
        'clotheswasher1','clotheswasher_dryg1','diningroom1','diningroom2','dishwasher1',
        'disposal1','drye1','dryg1','freezer1','furnace1','furnace2','garage1','garage2',
        'grid','heater1','heater2','heater3','housefan1','icemaker1','jacuzzi1','kitchen1',
        'kitchen2','kitchenapp1','kitchenapp2','lights_plugs1','lights_plugs2','lights_plugs3',
        'lights_plugs4','lights_plugs5','lights_plugs6','livingroom1','livingroom2','microwave1',
        'office1','outsidelights_plugs1','outsidelights_plugs2','oven1','oven2','pool1','pool2',
        'poollight1','poolpump1','pump1','range1','refrigerator1','refrigerator2','security1',
        'sewerpump1','shed1','solar','solar2','sprinkler1','sumppump1','utilityroom1','venthood1',
        'waterheater1','waterheater2','wellpump1','winecooler1','leg1v','leg2v']
    l2=['air2', 'air3', 'airwindowunit1', 'aquarium1', 'bathroom2', 'bedroom2', 'bedroom3', 'bedroom4', 'bedroom5', 'battery1', 'car2', 'circpump1', 'diningroom1', 'diningroom2', 'freezer1', 'furnace2', 'garage2', 'heater1', 'heater2', 'heater3', 'housefan1', 'icemaker1', 'jacuzzi1', 'kitchen1', 'kitchen2', 'lights_plugs3', 'lights_plugs4', 'lights_plugs5', 'lights_plugs6', 'livingroom2', 'office1', 'outsidelights_plugs1', 'outsidelights_plugs2', 'oven2', 'pool1', 'pool2', 'poollight1', 'poolpump1', 'pump1', 'refrigerator2', 'security1', 'sewerpump1', 'shed1', 'solar2', 'sprinkler1', 'sumppump1', 'utilityroom1', 'waterheater2', 'wellpump1', 'winecooler1']
    for i in l2:
        L1.remove(i)

    print(L1)
    # 创建一个示例文件夹和一些CSV文件用于测试
    # 调用函数执行合并
    #merge_csv_files(folder_path, output_file)