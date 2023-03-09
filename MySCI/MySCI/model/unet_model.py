""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

from .unet_parts import *
from .loss import LossFunction

class UNet(nn.Module):                # (3,3)
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        illu = torch.clamp(logits, 0.0001, 1)  # 区间收敛到[0.0001,1]


        return logits


# 网络结构
class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage      # epochs  default=3
        self.enhance = UNet(3, 3)   # 只有三层？ 输入层，卷积，输出
        self._criterion = LossFunction()                     # 损失函数

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):                        # 每一个epoch使用三次自校准模块
            inlist.append(input_op)                        # 输入添加到列表中
            i = self.enhance(input_op)                     # 输入放进光照估计网络得到输出  光照图
            r = input / i                                  # 想要的清晰图像 z=y/x     g_o 1
            r = torch.clamp(r, 0, 1)                       # 区间收敛到[0,1]
            # att = self.calibrate(r)                        # 自校准的map s          g_o 2
            input_op = r                         # 自校准的map s ，将其添加到低光观测上，g_o 3
            ilist.append(i)                                # 光照图列表，三个
            rlist.append(r)                                # 输出清晰图像列表，三个
            attlist.append(torch.abs(r))                 # 残差图列表  map s

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])   # 输入的低光观测，输出的光照图  求两个的损失
        return loss





if __name__ == '__main__':
    net = Network(stage=3)
    print(net)


