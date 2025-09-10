import numpy as np
import sys
import os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


# 加载数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 训练参数设置
iters_num = 10000
batch_size = 100
learning_rate = 0.1
train_size = x_train.shape[0]


# 初始化记录变量
train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

# 神经网络初始化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 随机选择批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度 - 使用反向传播替代数值梯度计算以提高效率
    grad = network.gradient(x_batch, t_batch)

    # 更新参数（权重和偏置）
    for key in ("w1", "b1", "w2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 记录当前批次的损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc,test acc | " + str(train_acc) + ", " + str(test_acc))

# 结果可视化
# 绘制训练和测试准确率曲线
plt.figure(figsize=(10, 6))
x = np.arange(len(train_acc_list))  # 创建epoch数轴
plt.plot(x, train_acc_list, label="train acc", marker="o")  # 训练准确率曲线
plt.plot(
    x, test_acc_list, label="test acc", linestyle="--", marker="s"
)  # 测试准确率曲线
plt.xlabel("Epochs")  # x轴标签
plt.ylabel("Accuracy")  # y轴标签
plt.ylim(0, 1.0)  # 设置y轴范围(0-100%)
plt.legend(loc="lower right")  # 图例位置
plt.title("Training and Test Accuracy over Epochs")  # 图表标题
plt.grid(True, linestyle="--", alpha=0.7)  # 添加网格线
plt.show()  # 显示图表
