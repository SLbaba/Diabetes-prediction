import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 数据集路径
dataset_path = "dataset/diabetes_prediction_dataset.csv"

# 读取数据集
dataset_df = pd.read_csv(dataset_path)

# 定义一个函数来绘制数据分布
def plot_distributions(df, save_path):
    matplotlib.rcParams['font.family'] = 'SimHei'  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    plt.suptitle('处理后分布情况一览', fontsize=20, y=0.99)  # 添加整个图像的标题

    # 列到绘图类型的映射
    cat_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
    num_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    for ax, column in zip(axes.flatten(), df.columns):
        if column in cat_columns:
            sns.countplot(x=column, data=df, ax=ax)
            ax.set_title(f'{column.replace("_", " ")}的分布')  # 替换下划线以改进显示
            ax.set_xlabel(column.replace("_", " "))
            ax.set_ylabel('数量')
        else:
            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_title(f'{column.replace("_", " ")}的分布')
            ax.set_xlabel(column.replace("_", " "))
            ax.set_ylabel('密度')

    plt.tight_layout()
    plt.savefig(save_path)

    return fig

# 绘制处理前数据分布
plot_distributions(dataset_df, 'figure/Initial-Data-Distributions.png')

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
# 保存图片
plt.savefig('figure/Diabetes-Distribution.png')

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

print(dataset_df['diabetes'].value_counts())

# 绘制处理后数据分布
plot_distributions(dataset_df, 'figure/Cleaned-Data-Distributions.png')
