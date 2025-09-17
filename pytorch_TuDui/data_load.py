from torch.utils.data import Dataset
from PIL import Image
import os


# help(Dataset)

class myData(Dataset):
    def __init__(self,root_dir,label_dir):
        """初始化函数，传入数据集的根目录和标签目录
        Args:
            root_dir (str): 数据集根目录
            label_dir (str): 标签目录
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)#列出路径下的所有文件名列表

    def __getitem__(self, index):
        """根据索引获取数据集中的一个样本
        Args:
            index (int): 样本索引
        Returns:
            tuple: (image, label) 图像和对应的标签
        """
        img_name = self.img_path[index]#根据索引获取文件名
        img_item_path = os.path.join(self.path,img_name)#拼接成完整路径
        img = Image.open(img_item_path)#打开图像
        label = self.label_dir#标签就是文件夹名称
        return img,label
    
    def __len__(self):
        """返回数据集的大小
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.img_path)


root_dir="/home/xc/AI_learning/pytorch_TuDui/pytorch_learning_dataset/test_dataset/train"
ants_label_dir="ants_image"
bees_label_dir="bees_image"

myData_ants=myData(root_dir,ants_label_dir)
myData_bees=myData(root_dir,bees_label_dir)

print(myData_ants.img_path)

train_data=myData_ants+myData_bees#合并两个数据集
print("训练数据集的长度：",len(train_data))#打印数据集的长度

img,label=train_data[36]#获取数据集中的某一个样本
print(img,label)#显示图像
img.show()
