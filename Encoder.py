import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = "dataset/diabetes_prediction_dataset_clean.csv"

# 读取数据集
dataset_df = pd.read_csv(dataset_path)

# 为 'gender' 列初始化编码器
gender_encoder = LabelEncoder()

# 拟合并转换 'gender' 列，并在数据框中替换
dataset_df['gender'] = gender_encoder.fit_transform(dataset_df['gender'])

# 为 'smoking_history' 列初始化一个新的编码器，以避免两个列之间的类别混淆
smoking_history_encoder = LabelEncoder()

# 拟合并转换 'smoking_history' 列，并在数据框中替换
dataset_df['smoking_history'] = smoking_history_encoder.fit_transform(dataset_df['smoking_history'])

# 查看标签编码后数据集类别分布
print(dataset_df['gender'].value_counts())

# 查看标签编码后数据集类别分布
print(dataset_df['smoking_history'].value_counts())

# 保存标签编码后的数据集
dataset_df.to_csv("dataset/diabetes_prediction_dataset_label_encoded.csv", index=False)




