from model.unet_model import Network
# from utils.dataset import ISBI_Loader
from multi_read_data import MemoryFriendlyLoader
from torch import optim
import torch.nn as nn
import torch
import os
import sys
import time
import glob
import numpy as np
import torch
# import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import utils



model_path = 'EXP/model_epochs/'
def main():
    # gpu不能用，直接退出
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(2)  # 随机种子 2
    cudnn.benchmark = True
    torch.manual_seed(2)
    cudnn.enabled = True
    torch.cuda.manual_seed(2)

    # 网络架构
    model = Network(stage=3)
    # 将模型加载到GPU上
    model = model.cuda()
    # 优化器，传入参数们，学习率，权重衰退
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=3e-4)


    # 训练集
    train_low_data_names = './data/difficult1'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names)

    # # 测试集
    # test_low_data_names = './data/medium'
    # TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names)

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False)
    # generator=torch.Generator(device='cuda')

    # test_queue = torch.utils.data.DataLoader(
    #     TestDataset, batch_size=1,
    #     pin_memory=True, num_workers=0, shuffle=True, generator=torch.Generator(device='cuda'))

    for epoch in range(5):
        model.train()
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            input = Variable(input, requires_grad=False).cuda()
            optimizer.zero_grad()
            loss = model._loss(input)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            logging.info('train-epoch %03d %03d %f', epoch, batch_idx, loss)

        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))



if __name__ == "__main__":
    main()