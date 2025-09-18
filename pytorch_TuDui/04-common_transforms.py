# 文档说明：介绍常见的transforms操作
"""
常见的transforms
输入：PIL Image.open()   # 输入是PIL库打开的图像
输出：tensor ToTensor()  # 输出通过ToTensor()转为张量
作用：narrays cv.imread() # 作用类似OpenCV的imread读取numpy数组
"""

from torchvision import transforms  # pytorch图像预处理模块
from PIL import Image  # python图像处理库
from torch.utils.tensorboard import SummaryWriter  # 训练可视化工具
import numpy as np


writer = SummaryWriter("transform_log")

img_path = "/home/xc/AI_learning/pytorch_TuDui/pytorch_learning_dataset/test_dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)

# 1. ToTensor转换：将PIL图像转为PyTorch张量
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)  # 执行转换
writer.add_image("tensor_image", img_tensor)  # 将原始张量图像写入TensorBoard

# 2. Normalize标准化：对图像进行归一化处理,输入输出均为tensor类型
# 参数说明：[0.5,0.5,0.5]是RGB三通道均值，[0.5,0.5,0.5]是标准差
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)  # 对张量执行归一化
writer.add_image("Normalize", img_norm, 0)  # 将归一化后的图像写入TensorBoard
trans_norm = transforms.Normalize([6, 1.5, 8.5], [0.345, 0.567, 0.545])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 1)   
trans_norm = transforms.Normalize([0.523, 0.565, 0.765], [0.345, 0.565, 0.245])
img_norm = trans_norm(img_tensor)  
writer.add_image("Normalize", img_norm, 2) 
print(type(img_norm))

# 3. Resize调整大小：将图像调整为指定尺寸
'''输出仍是PIL图像，输入一般也是PIL图像'''
# 创建Resize变换，目标尺寸为512x512像素
# 如果只输入一个尺寸数据，保持图像的宽高比，将图像的短边调整为该整数，长边按比例缩放
trans_resize = transforms.Resize((512, 512))
# 对原始PIL图像执行调整大小操作
img_resize = trans_resize(img_PIL)  
# 将调整大小后的PIL图像转换为张量
img_resize = trans_totensor(img_resize)
# 将调整后的图像写入TensorBoard，step=0表示第一步
writer.add_image('Resize', img_resize, 0)

# 4. Compose组合变换：创建包含多个变换的管道
# 创建只指定宽度的Resize变换（高度会按比例自动调整）
trans_resize_2 = transforms.Resize(512)
# 创建组合变换：先调整大小，再转换为张量
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# 应用组合变换（一步完成调整大小+张量转换）
img_resize_2 = trans_compose(img_PIL)
# 将结果写入TensorBoard，step=1表示第二步
writer.add_image('Resize', img_resize_2, 1)

# 5. RandomCrop随机裁剪：从图像中随机裁剪区域
'''输入输出均为PIL'''
# 创建随机裁剪变换：裁剪高度为512像素，宽度为1000像素
trans_random = transforms.RandomCrop(512, 1000)
# 创建组合变换：先随机裁剪，再转换为张量
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
# 循环10次生成10个不同的随机裁剪样本
for i in range(10):
    img_crop = trans_compose_2(img_PIL)  # 每次生成不同位置的裁剪
    writer.add_image('RandomCrop', img_crop, i)  # 将每个样本写入TensorBoard

writer.close()