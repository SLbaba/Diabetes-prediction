import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def load_dataset(dataset_path):
    """
    加载数据集。
    参数:
        dataset_path: 数据集的文件路径。
    返回:
        pandas DataFrame: 加载的数据集。
    """
    return pd.read_csv(dataset_path)

def calculate_metrics(y_true, y_pred):
    """
    计算并返回各项性能指标。
    参数:
        y_true: 真实标签。
        y_pred: 预测标签。
    返回:
        各项性能指标的字典。
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)
    precision_1 = precision_score(y_true, y_pred, pos_label=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)
    return {
        'accuracy': accuracy,
        'precision_0': precision_0,
        'recall_0': recall_0,
        'precision_1': precision_1,
        'recall_1': recall_1
    }

def perform_cross_validation(X, y, model):
    """
    执行十折交叉验证，并打印每一折的性能指标。
    参数:
        X: 特征数据集。
        y: 目标变量数据集。
        model: 用于训练和预测的模型实例。
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    metrics = []

    for train_index, test_index in kfold.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        fold_metrics = calculate_metrics(y_test_fold, y_pred_fold)
        metrics.append(fold_metrics)

    avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
    print(f"十折交叉验证平均性能指标: {avg_metrics}")

def evaluate_on_test_set(X_test, y_test, model):
    """
    在测试集上评估模型。
    参数:
        X_test: 测试集特征。
        y_test: 测试集目标变量。
        model: 训练完成的模型实例。
    """
    y_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)

    print("测试集上的性能指标:\n", test_metrics)
    print("测试集上的分类报告:\n", classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

def main(model, dataset_path):
    """
    主函数，加载数据，执行十折交叉验证和测试集评估。
    参数:
        model: 用于训练和预测的模型实例。
        dataset_path: 数据集的文件路径。
    """
    # 加载数据集
    dataset_df = load_dataset(dataset_path)

    # 分离特征和目标变量
    X = dataset_df.drop('diabetes', axis=1).values
    y = dataset_df['diabetes'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20216074)

    # 执行十折交叉验证
    perform_cross_validation(X_train, y_train, model)

    # 在测试集上评估模型
    evaluate_on_test_set(X_test, y_test, model)

# 实例化一个XGBoost分类器
xgb_classifier = xgb.XGBClassifier(
    use_label_encoder=False,  # 避免使用标签编码器的警告
    eval_metric='logloss',    # 评估模型性能的指标，对于二分类问题，'logloss'是一个好选择
)

rf_classifier = RandomForestClassifier(n_estimators=100)
lr_classifier = LogisticRegression(max_iter=10000)
dt_classifier = DecisionTreeClassifier()
mlp_classifier = MLPClassifier(max_iter=1000)

# 数据集路径
dataset_path = "dataset/diabetes_prediction_dataset_label_encoded.csv"

# 将模型和其名称放入一个字典中
models = {
    "XGBoost": xgb_classifier,
    "Random Forest": rf_classifier,
    "Logistic Regression": lr_classifier,
    "Decision Tree": dt_classifier,
    "MLP": mlp_classifier
}

# xgboost模型,随机森林模型,逻辑回归模型,决策树模型,支持向量机模型,多层感知机模型的性能比较
for model_name, model in models.items():
    print(f"{model_name}模型:")
    # 循环遍历每个模型，调用main函数
    main(model, dataset_path)
    print("-------")

