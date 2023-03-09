import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.low_img_dir = img_dir                              # './data/finetune'
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):   # 类似于广度优先遍历
            for name in names:                  # names是一个图片列表，取图片出来
                self.train_low_data_names.append(os.path.join(root, name))  # 每个图片的路径

        self.train_low_data_names.sort()              # 升序排列
        self.count = len(self.train_low_data_names)   # 获取图片数量

        transform_list = []
        transform_list += [transforms.ToTensor()]     # 将PIL和numpy格式的数据从[0,255]范围转换到[0,1]，数据归一化
        self.transform = transforms.Compose(transform_list)  # 将多个数据预处理的操作整合在一起，是一个列表存放的操作


    def __getitem__(self, index):
        im = Image.open(self.train_low_data_names[index]).convert('RGB')
        low = self.transform(im)

        # h = low.shape[1]  # 长
        # w = low.shape[2]  # 宽
        #
        # h_offset = random.randint(0, max(0, h - batch_h - 1))
        # w_offset = random.randint(0, max(0, w - batch_w - 1))
        # low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        img_name = self.train_low_data_names[index]
        return low, img_name

    def __len__(self):
        return self.count



# 拿到图片的路径数量，对每张图片进行处理，获取图像的通道数，转化为tensor格式（c,h,w）
# def getitem(self, index) 是Python中的一个特殊方法，用于实现对象的索引访问。当我们使用类似 obj[index] 的方式访问对象时，
# Python会自动调用该方法，并将索引值作为参数传递给它。该方法需要返回对应索引位置的元素值