import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import sys
sys.path.append('/home/zzw/code/NWPU-MOC-main')
from misc.utils import *

from config import cfg
# from models.Transformers.ST import ST
# from models.FPN.FPN_Head import FPN

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()
        self.fusion = cfg.MM   #use nir
        
        # 定义第一列卷积网络，大感受野 (卷积核 9x9)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4),  # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，下采样
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3), # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，再次下采样
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3), # 保持当前大小
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=7, stride=1, padding=3), # 保持当前大小
            nn.ReLU()
        )

        # 定义第二列卷积网络，中等感受野 (卷积核 7x7)
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, stride=1, padding=3),  # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，下采样
            nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2), # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，再次下采样
            nn.Conv2d(40, 20, kernel_size=5, stride=1, padding=2), # 保持当前大小
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=5, stride=1, padding=2), # 保持当前大小
            nn.ReLU()
        )

        # 定义第三列卷积网络，小感受野 (卷积核 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=2),  # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，下采样
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1), # 保持输入大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2 池化，再次下采样
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1), # 保持当前大小
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1), # 保持当前大小
            nn.ReLU()
        )

        # 最终融合
        self.fuse = nn.Conv2d(30, 6, kernel_size=1)  # 合并三个分支的输出通道

    def forward(self, rgb, nir):

        if self.fusion:
            x = torch.cat([rgb,nir], dim = 1)
        else:
            x = rgb
        # 分别通过三列网络
        x1 = self.branch1(x)  # 大感受野分支
        x2 = self.branch2(x)  # 中感受野分支
        x3 = self.branch3(x)  # 小感受野分支

        # 拼接三列的输出
        x = torch.cat((x1, x2, x3), 1)

        # 融合后输出密度图
        x = self.fuse(x)
        return x

# 测试输入图像
if __name__ == '__main__':
    model = MCNN()
    input_image = torch.randn(1, 3, 512, 512)  # 输入图像为 256x256 大小
    output = model(input_image)
    print("Output size:", output.size())  # 输出特征图大小
