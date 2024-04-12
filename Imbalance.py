from sklearn.linear_model import LogisticRegression

from Module import perform_cross_validation, evaluate_on_test_set, imbalance_method_list, load_dataset, dataset_path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def evaluate_methods(X, y, methods_list):
    results = []

    for method in methods_list:
        print(f"Evaluating {method}...")
        model = RandomForestClassifier(random_state=20216074)
        # model = LogisticRegression(max_iter=10000)
        metrics = perform_cross_validation(X, y, model, imbalance_method=method)
        results.append((method, metrics['f1_score_0'], metrics['f1_score_1'],(metrics['f1_score_0'] + metrics['f1_score_1']) /2 , metrics['accuracy']))
    return results

def plot_results(results):
    methods = [result[0] for result in results]
    f1_scores_positive = [result[1] for result in results]
    f1_scores_negative = [result[2] for result in results]
    f1_scores_overall = [result[3] for result in results]
    accuracy_list = [result[4] for result in results]

    plt.figure(figsize=(16, 8))
    plt.plot(methods, f1_scores_positive, label='Positive F1 Score', marker='o')
    plt.plot(methods, f1_scores_negative, label='Negative F1 Score', marker='o')
    plt.plot(methods, f1_scores_overall, label='F1 Score', marker='o')
    plt.plot(methods, accuracy_list, label='Accuracy', marker='o')
    # 显示方格
    plt.grid()
    plt.xlabel('Imbalance Handling Method')
    plt.ylabel('Metrics')
    plt.title('Model Performance by Imbalance Handling Method')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # 保存图片
    plt.savefig('figure/Imbalance-Handling-Method.png')

# 加载数据集
dataset_df= load_dataset(dataset_path)

# 分离特征和目标变量
X = dataset_df.drop('diabetes', axis=1).values
y = dataset_df['diabetes'].values

# 这里用X_train, y_train 替换 X, y 来演示
results = evaluate_methods(X, y, imbalance_method_list)
plot_results(results)
print("Imbalance.py executed successfully!")
# 将results保存为npy
import numpy as np
np.save("result/Imbalance-Handling-Method_results.npy", results)
