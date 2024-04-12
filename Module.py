import numpy as np
import pandas as pd
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, TomekLinks, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.cluster import KMeans

imbalance_method_list = [
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "SMOTENC",
    "KMeansSMOTE",
    "RandomUnderSampler",
    "NearMiss",
    "TomekLinks",
    "OneSidedSelection",
    "NeighbourhoodCleaningRule",
    "InstanceHardnessThreshold",
    "SMOTETomek",
    "SMOTEENN"
]

# 数据集路径
dataset_path = "dataset/diabetes_prediction_dataset_label_encoded.csv"

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
    f1_score_0 = f1_score(y_true, y_pred, pos_label=0)
    precision_1 = precision_score(y_true, y_pred, pos_label=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1)
    f1_score_1 = f1_score(y_true, y_pred, pos_label=1)

    return {
        'accuracy': accuracy,
        'precision_0': precision_0,
        'recall_0': recall_0,
        'f1_score_0': f1_score_0,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f1_score_1': f1_score_1
    }

def perform_cross_validation(X, y, model, imbalance_method=None):
    """
    执行十折交叉验证，并打印每一折的性能指标。
    参数:
        X: 特征数据集。
        y: 目标变量数据集。
        model: 用于训练和预测的模型实例。
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=20216074)
    metrics = []

    stand = preprocessing.StandardScaler()
    X = stand.fit_transform(X)
    # kmeans_estimator = KMeans(n_clusters=200 , random_state=20216074)

    # 实例化KMeansSMOTE，并通过kmeans_estimator参数传递KMeans估计器
    # B_method = under_sampling.InstanceHardnessThreshold(random_state=20216074,n_jobs=-1)
    # B_method = over_sampling.KMeansSMOTE(random_state=20216074, kmeans_estimator=KMeans(n_clusters=35, random_state=20216074),sampling_strategy=0.15)
    # B_method = over_sampling.SMOTE(random_state=20216074, sampling_strategy=0.2)
    # B_method = under_sampling.RandomUnderSampler(random_state=20216074, sampling_strategy=0.15)
    # B_method = combine.SMOTEENN(random_state=20216074, sampling_strategy=0.10)
    if imbalance_method == 'SMOTE':
        imbalance_processor = SMOTE(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'ADASYN':
        imbalance_processor = ADASYN(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'BorderlineSMOTE':
        imbalance_processor = BorderlineSMOTE(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'SVMSMOTE':
        imbalance_processor = SVMSMOTE(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'SMOTENC':
        imbalance_processor = SMOTENC(random_state=20216074,categorical_features=[0,4,3,5], sampling_strategy=0.15)
    elif imbalance_method == 'KMeansSMOTE':
        imbalance_processor = KMeansSMOTE(random_state=20216074, kmeans_estimator=KMeans(n_clusters=35, random_state=20216074),sampling_strategy=0.15)
    elif imbalance_method == 'RandomUnderSampler':
        imbalance_processor = RandomUnderSampler(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'NearMiss':
        imbalance_processor = NearMiss(sampling_strategy=0.15)
    elif imbalance_method == 'TomekLinks':
        imbalance_processor = TomekLinks()
    elif imbalance_method == 'OneSidedSelection':
        imbalance_processor = OneSidedSelection()
    elif imbalance_method == 'NeighbourhoodCleaningRule':
        imbalance_processor = NeighbourhoodCleaningRule()
    elif imbalance_method == 'InstanceHardnessThreshold':
        imbalance_processor = InstanceHardnessThreshold(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'SMOTETomek':
        imbalance_processor = SMOTETomek(random_state=20216074, sampling_strategy=0.15)
    elif imbalance_method == 'SMOTEENN':
        imbalance_processor = SMOTEENN(random_state=20216074, sampling_strategy=0.15)
    else:
        imbalance_processor = None

    if imbalance_processor != None:
        #通过fit_resample方法对数据集进行过采样
        X, y = imbalance_processor.fit_resample(X, y)

    y = pd.Series(y)
    print(y.value_counts())
    # y_KSMT的pie图
    # plt.figure()
    # y = pd.Series(y)
    # y.value_counts().plot(kind='bar')
    # plt.title('Diabetes Distribution')
    # plt.show()



    for train_index, test_index in kfold.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        # X_train_fold, y_train_fold = B_method.fit_resample(X_train_fold, y_train_fold)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        fold_metrics = calculate_metrics(y_test_fold, y_pred_fold)
        metrics.append(fold_metrics)

    avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
    print(f"十折交叉验证平均性能指标: {avg_metrics}")
    return avg_metrics
def evaluate_on_test_set(X_test, y_test, model):
    """
    在测试集上评估模型。
    参数:
        X_test: 测试集特征。
        y_test: 测试集目标变量。
        model: 训练完成的模型实例。
    """
    # X_test标准化
    stand = preprocessing.StandardScaler()
    X_test = stand.fit_transform(X_test)

    y_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)

    print("测试集上的性能指标:\n", test_metrics)
    print("测试集上的分类报告:\n", classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))

    # 返回各种指标
    return test_metrics