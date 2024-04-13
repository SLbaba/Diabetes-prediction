import numpy as np
import matplotlib.pyplot as plt
import copy

train_metrics = np.load('result/train_metrics.npy', allow_pickle=True)
test_metrics = np.load('result/test_metrics.npy', allow_pickle=True)

def plot_imbalance_method(results):
    # Initialize a dictionary to hold the computed averages for each method
    method_metrics = {}

    # Iterate through each imbalance method group in results
    for group in results:
        for result in group:
            method = result['imbalance_method']
            if method not in method_metrics:
                method_metrics[method] = {'f1_score_0': [], 'f1_score_1': [], 'accuracy': []}

            method_metrics[method]['f1_score_0'].append(result['f1_score_0'])
            method_metrics[method]['f1_score_1'].append(result['f1_score_1'])
            method_metrics[method]['accuracy'].append(result['accuracy'])

    # Compute averages
    for method, metrics in method_metrics.items():
        method_metrics[method]['f1_score_0'] = np.mean(metrics['f1_score_0'])
        method_metrics[method]['f1_score_1'] = np.mean(metrics['f1_score_1'])
        method_metrics[method]['accuracy'] = np.mean(metrics['accuracy'])
        method_metrics[method]['f1_score_overall'] = (method_metrics[method]['f1_score_0'] + method_metrics[method]['f1_score_1']) / 2

    # Extracting data for plotting
    methods = list(method_metrics.keys())
    f1_score_0 = [method_metrics[method]['f1_score_0'] for method in methods]
    f1_score_1 = [method_metrics[method]['f1_score_1'] for method in methods]
    f1_score_overall = [method_metrics[method]['f1_score_overall'] for method in methods]
    accuracy = [method_metrics[method]['accuracy'] for method in methods]

    # Plotting
    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the size of the figure here
    ax.bar(x - 1.5 * width, f1_score_0, width, label='F1 Score Class 0')
    ax.bar(x - 0.5 * width, f1_score_1, width, label='F1 Score Class 1')
    ax.bar(x + 0.5 * width, f1_score_overall, width, label='F1 Score Overall')
    ax.bar(x + 1.5 * width, accuracy, width, label='Accuracy')

    # Highlight the maximum values for each metric
    ax.axhline(y=max(f1_score_0), color='blue', linestyle='--', label='Max F1 Score 0')
    ax.axhline(y=max(f1_score_1), color='green', linestyle='--', label='Max F1 Score 1')
    ax.axhline(y=max(f1_score_overall), color='red', linestyle='--', label='Max F1 Score Overall')
    ax.axhline(y=max(accuracy), color='purple', linestyle='--', label='Max Accuracy')

    # Adjust y-axis to enhance visibility of differences
    all_values = f1_score_0 + f1_score_1 + f1_score_overall + accuracy
    ax.set_ylim(min(all_values) * 0.95, max(all_values) * 1.05)  # Adjust the y-axis limits

    ax.set_xlabel('Imbalance Method', fontsize=15)
    ax.set_ylabel('Average Scores', fontsize=15)
    ax.set_title('Performance Metrics by Imbalance Method', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    legend = ax.legend()
    legend.set_title('Metrics', prop={'size': 10})  # Adjust the legend title font size
    for text in legend.get_texts():
        text.set_fontsize('small')  # Adjust the font size of legend text

    # Save the plot
    plt.tight_layout()
    plt.savefig('figure/Performance Metrics by Imbalance Method.png')

def plot_model(results):
    model_names = ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "MLP"]
    # Define unique colors for each metric's maximum line
    colors = {
        'Accuracy': 'red',
        'Precision 0': 'blue',
        'Recall 0': 'green',
        'F1 Score 0': 'orange',
        'Precision 1': 'purple',
        'Recall 1': 'pink',
        'F1 Score 1': 'cyan'
    }

    for method_index, group in enumerate(results):
        fig, ax = plt.subplots(figsize=(16, 12))
        x = np.arange(len(model_names))  # x positions for the bars
        width = 0.12  # width of the bars

        metrics_data = {
            'Accuracy': [],
            'Precision 0': [],
            'Recall 0': [],
            'F1 Score 0': [],
            'Precision 1': [],
            'Recall 1': [],
            'F1 Score 1': []
        }

        # Gather data for each metric
        for model_results in group:
            for key in metrics_data.keys():
                metric_key = key.lower().replace(' ', '_')  # Make sure this matches exactly with your data keys
                metrics_data[key].append(model_results[metric_key])

        # Plot each metric
        offset = -3 * width  # Initial offset from the center
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + offset + i * width, values, width, label=metric)

        # Highlight max values for each metric
        for metric, values in metrics_data.items():
            max_value = max(values)
            ax.axhline(y=max_value, color=colors[metric], linestyle='--', linewidth=1.2, label=f'Max {metric}')

        # Adjust y-axis scale to make the 0.8-1 range clearer
        ax.set_ylim(0.45, 1.0)

        ax.set_xlabel('Model', fontsize=15)
        ax.set_ylabel('Values', fontsize=15)
        ax.set_title(f"Metrics for method: {group[0]['imbalance_method']}", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        legend = ax.legend(fontsize='small', title='Metrics', title_fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(f'figure/Metrics_{group[0]["imbalance_method"]}.png')
        plt.show()

# Assuming results is populated with the correct data structure
# plot_model(results)



# 画图
# plot_imbalance_method(test_metrics)
# plot_model(test_metrics)


models = [
    "XGBoost",
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "MLP"
]

def save_model_metrics(data, name=None):
    res = []
    for item in data:
        for i in range(len(item)):
            # print(item[i])
            temp = copy.deepcopy(item[i])
            # 数据类型保留四位小数
            temp.update({"model": models[i]})
            for key in temp.keys():
                if key != "imbalance_method" and key != "model":
                    temp.update({key: round(temp[key], 4)})
            res.append(temp)
    # 保存为npy
    np.save(f"result/model_{name}_metrics.npy", res)

    return res
res1 = save_model_metrics(train_metrics, "train")
res2 = save_model_metrics(test_metrics, "test")

for item in res1:
    print(item)
    print("------")

print("------------------------------------------------")

for item in res2:
    print(item)
    print("------")