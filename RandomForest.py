import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from Module import load_dataset, dataset_path

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

# 选择一个阈值
threshold = 0.6  # 可调整以优化性能

# 生成基于阈值的预测
y_pred = (y_scores > threshold).astype(int)

# 计算测试指标
accuracy = accuracy_score(y_test, y_pred)
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
f1_score_0 = f1_score(y_test, y_pred, pos_label=0)
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
f1_score_1 = f1_score(y_test, y_pred, pos_label=1)
cm = confusion_matrix(y_test, y_pred)

# 输出性能指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision Class 0: {precision_0:.4f}, Precision Class 1: {precision_1:.4f}")
print(f"Recall Class 0: {recall_0:.4f}, Recall Class 1: {recall_1:.4f}")
print(f"F1 Score Class 0: {f1_score_0:.4f}, F1 Score Class 1: {f1_score_1:.4f}")
print("Confusion Matrix:")
print(cm)
print("测试集上的性能指标:\n", {
    "Accuracy": accuracy,
    "Precision 0": precision_0,
    "Recall 0": recall_0,
    "F1 Score 0": f1_score_0,
    "Precision 1": precision_1,
    "Recall 1": recall_1,
    "F1 Score 1": f1_score_1
})
print("测试集上的分类报告:\n", classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

# 可视化混淆矩阵
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar(cax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_xticklabels([''] + ['Non-Diabetes', 'Diabetes'])
ax.set_yticklabels([''] + ['Non-Diabetes', 'Diabetes'])
# 在矩阵中标注数值
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')
# 保存图片
plt.savefig('figure/Confusion-Matrix.png')
# 显示图片
plt.show()