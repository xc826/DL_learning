import sys
import os
import pickle

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=True, one_hot_label=False
)

print(np.random.choice(60000, 10))


# mini_batch版交叉熵误差实现（one-hot标签）
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# mini_batch版交叉熵误差实现（非one-hot标签）
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # 使用高级索引提取每个样本真实标签对应的预测概率
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
