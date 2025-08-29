# write by deepseek v3.1

from utils.read_data import *
from sklearn.model_selection import train_test_split
import numpy as np

# 假设已有以下数据
train_pic = get_train_pic()  # 特征数据
train_label = get_train_label()  # 标签数据
p = 0.85  # 训练集比例

# 使用分层抽样划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    train_pic, 
    train_label, 
    train_size=p,           # 训练集比例
    random_state=42,       # 随机种子，确保结果可复现
    stratify=train_label   # 关键参数：按标签进行分层抽样
)

# 打印划分结果信息
print(f"原始数据集大小: {len(train_pic)}")
print(f"训练集大小: {len(X_train)} ({len(X_train)/len(train_pic):.1%})")
print(f"测试集大小: {len(X_test)} ({len(X_test)/len(train_pic):.1%})")

# 检查类别分布是否保持一致
print("\n各类别在原始数据中的分布:")
unique, counts = np.unique(train_label, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"类别 {cls}: {count} ({count/len(train_label):.2%})")

print("\n各类别在训练集中的分布:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for cls, count in zip(unique_train, counts_train):
    print(f"类别 {cls}: {count} ({count/len(y_train):.2%})")

print("\n各类别在测试集中的分布:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for cls, count in zip(unique_test, counts_test):
    print(f"类别 {cls}: {count} ({count/len(y_test):.2%})")

np.save('data/train_pic_train.npy',X_train)
np.save('data/train_label_train.npy',y_train)
np.save('data/train_pic_test.npy',X_test)
np.save('data/train_label_test.npy',y_test)