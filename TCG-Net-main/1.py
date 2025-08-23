import pickle
import numpy as np

with open("encoded_data.pkl", "rb") as f:
    data = pickle.load(f)

print("类型:", type(data))
if isinstance(data, np.ndarray):
    print("数组形状:", data.shape)
elif isinstance(data, dict):
    for k, v in data.items():
        print(f"{k}: {type(v)}", getattr(v, "shape", None))
elif isinstance(data, list):
    print("列表长度:", len(data))
    if len(data) > 0:
        print("第一个元素类型:", type(data[0]), getattr(data[0], "shape", None))
