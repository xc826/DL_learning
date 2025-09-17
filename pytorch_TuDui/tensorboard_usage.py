from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("tensorboard_log")

image_path="/home/xc/AI_learning/pytorch_TuDui/pytorch_learning_dataset/test_dataset/train/ants_image/0013035.jpg"
img_PIL=Image.open(image_path)
# print(type(img_PIL))
# print(img_PIL)
img_array=np.array(img_PIL)
# print(img_array.shape)
# print(img_array)
# img_array=img_array.transpose(2,0,1)#转成通道数在前
# print(img_array.shape)
# print(img_array)
# print(img_array)
          

writer.add_image("test",img_array,1,dataformats="HWC")#HWC表示高宽通道数

for i in range(100):
    writer.add_scalar("y=x",i,i)
    writer.add_scalar("y=x^2",i**2,i)
    writer.add_scalar("y=x^3",i**3,i)


writer.close()