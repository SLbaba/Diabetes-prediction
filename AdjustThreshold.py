import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from Module import load_dataset, dataset_path
from tqdm import tqdm  # 导入tqdm

# 加载数据
data = load_dataset(dataset_path)

# 分离特征和目标变量
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20216074)

# 应用SMOTEENN
smote_enn = SMOTEENN(random_state=20216074, sampling_strategy=0.15)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# 初始化随机森林模型
model = RandomForestClassifier(random_state=20216074)

# 训练模型
model.fit(X_resampled, y_resampled)

# 预测概率
y_scores = model.predict_proba(X_test)[:, 1]

# 阈值数组
thresholds = np.arange(0, 1.01, 0.01)

# 初始化性能指标列表
accuracies, precisions_0, recalls_0, f1_scores_0, precisions_1, recalls_1, f1_scores_1 = [], [], [], [], [], [], []

# 遍历不同阈值
for thresh in tqdm(thresholds, desc="Processing Thresholds"):
    y_pred = (y_scores > thresh).astype(int)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions_0.append(precision_score(y_test, y_pred, pos_label=0))
    recalls_0.append(recall_score(y_test, y_pred, pos_label=0))
    f1_scores_0.append(f1_score(y_test, y_pred, pos_label=0))
    precisions_1.append(precision_score(y_test, y_pred, pos_label=1))
    recalls_1.append(recall_score(y_test, y_pred, pos_label=1))
    f1_scores_1.append(f1_score(y_test, y_pred, pos_label=1))

# 绘制性能指标随阈值变化的图
plt.figure(figsize=(10, 10))
plt.plot(thresholds, accuracies, label='Accuracy')
plt.plot(thresholds, precisions_0, label='Precision Class 0')
plt.plot(thresholds, recalls_0, label='Recall Class 0')
plt.plot(thresholds, f1_scores_0, label='F1 Score Class 0')
plt.plot(thresholds, precisions_1, label='Precision Class 1')
plt.plot(thresholds, recalls_1, label='Recall Class 1')
plt.plot(thresholds, f1_scores_1, label='F1 Score Class 1')
plt.title('Performance Metrics by Classification Threshold')
plt.xlabel('Classification Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
# 保存图片
plt.savefig('figure/Threshold-Performance-Metrics.png')
# 保存npy
np.save("result/Threshold-Performance-Metrics.npy", [accuracies, precisions_0, recalls_0, f1_scores_0, precisions_1, recalls_1, f1_scores_1])
# 显示图片
plt.show()