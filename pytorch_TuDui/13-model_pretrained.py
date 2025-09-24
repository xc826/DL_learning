

import torchvision
import torch.nn as nn


vgg16_false=torchvision.models.vgg16()
vgg16_true=torchvision.models.vgg16(weights='DEFAULT')

# print(vgg16_true)

# 加载CIFAR10训练数据集（10分类小图片数据集）
dataset = torchvision.datasets.CIFAR10(root='./CIFAR10_dataset',train=True,download=True,transform=torchvision.transforms.ToTensor())

# 修改预训练模型（输出1000类）以适应CIFAR10的10分类任务
# 在classifier末尾添加一个新的全连接层（1000->10）
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# 打印修改后的模型结构
print(vgg16_true)


print("----------------------------------1----------------------------------------------")
print(vgg16_false)
vgg16_false.classifier[6]=nn.Linear(4096,10)
print("-----------------------------------2---------------------------------------------")
print(vgg16_false)