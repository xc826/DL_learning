import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

"""
# 创建CIFAR10训练数据集
train_set = torchvision.datasets.CIFAR10(
    root='./CIFAR10_DATA',  # 数据集存储路径
    train=True,  # 加载训练集
    transform=dataset_transform,  # 应用定义的预处理变换
    download=True  # 如果本地不存在则下载数据集
)
"""

train_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10_dataset", train=True, transform=dataset_transform, download=True
)
test_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10_dataset", train=True, transform=dataset_transform, download=True
)

# print(test_set[0])
# print(test_set.classes)

img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])

writer = SummaryWriter("CIFAR10_log")

for i in range(10):
    img, target = test_set[i]
    writer.add_image("CIFAR10", img, i)

writer.close()
