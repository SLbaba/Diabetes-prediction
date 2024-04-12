import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from Module import perform_cross_validation, evaluate_on_test_set, imbalance_method_list, load_dataset, dataset_path



def main(model, dataset_path, imbalance_method=None):
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
    perform_cross_validation(X_train, y_train, model, imbalance_method)

    # 在测试集上评估模型
    evaluate_on_test_set(X_test, y_test, model)

# 实例化一个XGBoost分类器
xgb_classifier = xgb.XGBClassifier(
    use_label_encoder=False,  # 避免使用标签编码器的警告
    eval_metric='logloss',    # 评估模型性能的指标，对于二分类问题，'logloss'是一个好选择
)


rf_classifier = RandomForestClassifier(n_estimators=50)
lr_classifier = LogisticRegression(max_iter=10000)
dt_classifier = DecisionTreeClassifier()
mlp_classifier = MLPClassifier(max_iter=1000)


# 将模型和其名称放入一个字典中
models = {
    "XGBoost": xgb_classifier,
    "Random Forest": rf_classifier,
    "Logistic Regression": lr_classifier,
    "Decision Tree": dt_classifier,
    "MLP": mlp_classifier
}

main(xgb_classifier, dataset_path)

# for imbalance_method in imbalance_method_list:
#     print(f"Imbalance method: {imbalance_method}")
#     main(xgb_classifier, dataset_path, imbalance_method)
#     print("-------")










# # xgboost模型,随机森林模型,逻辑回归模型,决策树模型,支持向量机模型,多层感知机模型的性能比较
# for model_name, model in models.items():
#     print(f"{model_name}模型:")
#     # 循环遍历每个模型，调用main函数
#     main(model, dataset_path)
#     print("-------")

