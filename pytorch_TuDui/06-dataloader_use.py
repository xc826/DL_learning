import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
# 使用CIFAR10数据集，参数说明：
# root：数据集存储路径
# train=False：使用测试集（非训练集）
# transform：将PIL图像转换为PyTorch张量
# download=True：如果本地不存在则自动下载
test_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)


# 定义测试数据加载器（覆盖前一个定义），参数说明：
# dataset：使用的数据集
# batch_size=64：每批加载64张图像
# shuffle=True：打乱数据顺序
# num_workers=0：使用主进程加载数据（无子进程）
# drop_last=False：保留最后不足批量的样本
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 获取测试数据集中的第一张图像及其标签
img, target = test_set[0]
# 打印图像张量的形状（通道数×高度×宽度）
print(img.shape)  # 输出：torch.Size([3, 32, 32])
# 打印图像的类别标签（0-9之间的整数）
print(target)     # 输出：3（代表猫类别）

# 创建SummaryWriter实例
writer = SummaryWriter("dataloader_log")

# epoch=0

for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        writer.add_images("epoch: {}".format(epoch),imgs,step)
        step+=1

writer.close()