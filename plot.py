import numpy as np

# 读取np
print("train_metrics:-----------------")
train_metrics = np.load('result/train_metrics.npy', allow_pickle=True)
print("test_metrics:-----------------")
test_metrics = np.load('result/test_metrics.npy', allow_pickle=True)
print(test_metrics)