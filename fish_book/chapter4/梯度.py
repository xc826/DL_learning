import sys
import os
import pickle

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


# def numerical_gradient(f, x):
#     """
#     使用中心差分法计算函数在给定点的梯度
#     参数:
#         f: 目标函数
#         x: 需要计算梯度的点（numpy数组）
#     返回:
#         grad: 梯度向量（与x同形状）
#     """
#     h = 1e-2  # 差分步长（0.01）
#     grad = np.zeros_like(x)  # 生成和x形状相同的全零数组

#     # 对每个维度分别计算偏导数
#     for idx in range(x.size):
#         tmp_val = x[idx]  # 保存原始值

#         # 计算f(x+h)
#         x[idx] = tmp_val + h
#         fxh1 = f(x)

#         # 计算f(x-h)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         # 使用中心差分公式计算偏导数
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#         x[idx] = tmp_val  # 恢复原始值

#     return grad


# 梯度下降法实现
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    使用梯度下降法寻找函数最小值
    参数:
        f: 目标函数
        init_x: 初始点
        lr: 学习率（默认0.01）
        step_num: 迭代次数（默认100）
    返回:
        x: 优化后的参数
    """
    x = init_x.copy()  # 创建初始点的副本

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


# 测试梯度计算
if __name__ == "__main__":
    # 在不同点计算梯度
    print("梯度测试结果:")
    test1 = numerical_gradient(function_2, np.array([3.0, 4.0]))  # [6, 8]
    test2 = numerical_gradient(function_2, np.array([0.0, 2.0]))  # [0, 4]
    test3 = numerical_gradient(function_2, np.array([3.0, 0.0]))  # [6, 0]
    print(f"(3,4)处梯度: {test1}")
    print(f"(0,2)处梯度: {test2}")
    print(f"(3,0)处梯度: {test3}")

    # 使用梯度下降寻找最小值
    print("\n梯度下降求最小值:")
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
    print(f"初始点: {init_x} -> 优化结果: {result}")


# ------------------------ 神经网络梯度示例 ------------------------

# 神经网络的梯度计算


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# 测试神经网络梯度
if __name__ == '__main__':
    print("\n神经网络梯度示例:")
    net = simpleNet()
    print(f"初始权重:\n{net.W}")

    # 输入样本
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(f"预测输出: {p}")

    # 获取最大概率索引
    agm = np.argmax(p)
    print(f"预测类别: {agm}")

    # 正确标签（one-hot编码）
    t = np.array([0, 0, 1])
    los = net.loss(x, t)
    print(f"损失值: {los:.4f}")


    # 计算梯度
    def f(W):
        """包装函数用于计算损失"""
        return net.loss(x, t)


    # 计算损失函数关于权重的梯度
    dw = numerical_gradient(f, net.W)
    print(f"权重梯度:\n{dw}")