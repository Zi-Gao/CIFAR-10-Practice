import numpy as np
from utils import read_data
pic=read_data.get_train_pic().astype(np.float32)/255
mean_np = np.mean(pic, axis=(0, 2, 3))
std_np = np.std(pic, axis=(0, 2, 3))
print(f"NumPy Mean: {mean_np}")
print(f"NumPy Std:  {std_np}")