import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = "dataset/diabetes_prediction_dataset.csv"

# 读取数据集
dataset_df = pd.read_csv(dataset_path)

# 查看属性
print(dataset_df.columns)

# 查看数据集类别分布
print(dataset_df['diabetes'].value_counts())

# 绘制数据集类别类别分布图
plt.figure()
# 绘制柱状图
dataset_df['diabetes'].value_counts().plot(kind='bar')
# 设置标题
plt.title('Diabetes Distribution')
# 设置x轴标签
plt.xlabel('Diabetes')
# 设置y轴标签
plt.ylabel('Count')
# 显示图片
plt.show()

# 检查缺失值
print(dataset_df.isnull().sum())

# 统计重复数据的数量
duplicated_amt = dataset_df.duplicated().sum()
print(duplicated_amt)

# 数据的形状
print(dataset_df.shape)

# 数据的数量
dataset_amt = dataset_df.shape[0]


# 删除重复数据
dataset_df.drop_duplicates(inplace=True)

# 剔除重复数据后的数据形状
print(dataset_df.shape)

# 剔除重复数据后的数据量
drop_dataset_amt = dataset_df.shape[0]

# 检查是否成功删除重复数据
if(drop_dataset_amt == dataset_amt - duplicated_amt):
    print("Drop duplicated data successfully!")

# 查看数据属性的数据类型
print(dataset_df.dtypes)

# 数据的gender属性的类别
print(dataset_df['gender'].value_counts())

# 剔除gender属性为Other的数据
dataset_df = dataset_df[dataset_df['gender'] != 'Other']

# 查看剔除gender属性为Other的数据后的gender属性的类别
print(dataset_df['gender'].value_counts())

# 查看smoking_history属性的类别
print(dataset_df['smoking_history'].value_counts())

# 保存处理后的数据集
dataset_df.to_csv("dataset/diabetes_prediction_dataset_clean.csv", index=False)

