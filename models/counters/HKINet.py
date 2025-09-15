import torch
from torch import nn
from torch.utils import model_zoo
import pdb
from torchvision import models
# MFEN-v3

class HKINet(nn.Module):
    def __init__(self):
        super(HKINet, self).__init__()
        self.ff = FEnd()
        self.dmp = BackEnd()

    def forward(self, input):
        input = self.ff(input)
        dmp_out = self.dmp(input)

        return dmp_out


class FEnd(nn.Module):
    def __init__(self):
        super(FEnd, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features.children())

        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])

    def forward(self, x):
        x = self.features1(x)
        x_1_2 = self.features2(x)
        x_1_4 = self.features3(x_1_2)
        x_1_8 = self.features4(x_1_4)
        # 得到三个不同尺度的特征图
        return x_1_2, x_1_4, x_1_8

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()

        self.ms1 = nn.Sequential(
            BaseConv(128, 64, 3, padding=1, use_bn=False),
        #     BaseConv(64, 64, 3, padding=1, use_bn=False),
        #     BaseConv(64, 64, 3, padding=1, use_bn=False)
        )
        self.ms2 = nn.Sequential(
            BaseConv(256, 128, 3, padding=1, use_bn=False),
            # BaseConv(128, 128, 3, padding=1, use_bn=False)
        )
        self.ms3 = BaseConv(512, 256, 1, padding=0, use_bn=False)

        self.hrb1 = HRB(128)
        self.hrb2 = HRB(128)
        self.hrb3 = HRB(128)
        # self.hrb4 = HRB(128)

        self.ssm = SSF1(128)
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.ms4 = BaseConv(64, 32, 1, padding=0)
        self.pre = BaseConv(32, 6, 1, padding =0, activation=None)
        # self.pre = BaseConv(32, 6, 1, padding =0)

        # self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.ff1 = BaseConv(256, 128, 1, padding=0)
        # self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.ff2 = BaseConv(128, 64, 1, padding=0)
        # self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.pre = BaseConv(32, 1, 1, padding=0, activation=None)

    def forward(self, input):
        # 尺度从大到小，通道数从小到大
        conv2_2, conv3_3, conv4_3 = input
        # 降通道数为64，128，256
        x11 = self.ms1(conv2_2)
        x21 = self.ms2(conv3_3)
        x31 = self.ms3(conv4_3)
        x1, x2, x3 = self.hrb1(x11, x21, x31)
        x1, x2, x3 = self.hrb2(x1, x2, x3)
        x1, x2, x3 = self.hrb3(x1, x2, x3)
        # x1, x2, x3 = self.hrb4(x1, x2, x3)
        x1 += x11
        x2 += x21
        x3 += x31
        x = self.ssm(x1, x2, x3)
        # pdb.set_trace()

        x = self.up(x)
        # x = self.ms4(x)
        x = self.pre(x)
        return x


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, padding=1, activation=nn.ReLU(), use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation=dilation)
        self.conv.weight.data.normal_(0, 0.01)
        #self.conv.weight.data.kaiming_uniform_()
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class CALayer(nn.Module):
    def __init__(self, num, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                BaseConv(channel, channel // num, 1, padding =0),
                BaseConv(channel // num, channel, 1, padding =0, activation=nn.Sigmoid())
        )
    def forward(self, x):
        y = self.avg_pool(x)
        #pdb.set_trace()
        y = self.ca(y)
        return x * y


# scale selection attention
class SSAtt(nn.Module):
    def __init__(self, num, channel):
        super(SSAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            BaseConv(channel, channel // num, 1, padding=0),
            BaseConv(channel // num, channel, 1, padding=0, activation=nn.Sigmoid())
        )
        # self.conv1 = BaseConv(channel, channel//num, 1, padding=0)
        # self.conv2 = BaseConv(channel // num, channel, 1, padding=0, activation=nn.Sigmoid())
        # self.conv3 = BaseConv(channel // num, channel, 1, padding=0, activation=nn.Sigmoid())
        # self.conv4 = BaseConv(channel // num, channel, 1, padding=0, activation=nn.Sigmoid())
        # self.conv3 = BaseConv(channel // num, channel, 1, padding=0, activation=nn.Sigmoid())
        self.conv = BaseConv(3*channel, channel, 1, padding=0)

    def forward(self, x1, x2, x3):
        x_sum = x1 + x2 + x3
        y = self.avg_pool(x_sum)
        #pdb.set_trace()
        y = self.ca(y)
        # y = self.conv1(y)
        # y1 = self.conv2(y)
        # y2 = self.conv3(y)
        # y3 = self.conv4(y)
        # y3 = self.conv1(y)
        x = torch.cat([x1 * y, x2 * y, x3 * y], 1)
        x = self.conv(x)
        return x

class SSF1(nn.Module):
    def __init__(self, in_channels):
        super(SSF1, self).__init__()
        self.conv = BaseConv(in_channels//2, 64, 1, padding=0)
        self.up1 = nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.up1 = BaseConv(in_channels, 64, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(2*in_channels, 64, 4, stride=4, padding=0, output_padding=0, bias=True)
        # self.up2 = BaseConv(2*in_channels, 64, 3, padding=1)
        self.ssa = SSAtt(8, 64)

    def forward(self, x1, x2, x3):
        x1 = self.conv(x1)
        # x2 = self.up1(x2)
        # x2 = nn.functional.interpolate(x2, scale_factor=2)
        x2 = self.up1(x2)
        # x3 = self.up2(x3)
        # x3 = nn.functional.interpolate(x3, scale_factor=4)
        x3 = self.up2(x3)
        # pdb.set_trace()
        x = self.ssa(x1, x2, x3)
        return x

class SSF2(nn.Module):
    def __init__(self, in_channels):
        super(SSF2, self).__init__()
        self.up1 = nn.ConvTranspose2d(2*in_channels, 128, 4, stride=2, padding=1, output_padding=0, bias=True)
        # self.up1 = BaseConv(2*in_channels, 128, 3, padding=1)
        self.conv = BaseConv(in_channels, 128, 1, padding=0)
        self.down1 = BaseConv(in_channels//2, 128, 3, stride=2, padding=1)
        self.ssa = SSAtt(8, 128)

    def forward(self, x1, x2, x3):
        # x3 = self.up1(x3)
        # x3 = nn.functional.interpolate(x3, scale_factor=2)
        x3 = self.up1(x3)
        x2 = self.conv(x2)
        x1 = self.down1(x1)
        x = self.ssa(x1, x2, x3)
        return x

class SSF3(nn.Module):
    def __init__(self, in_channels):
        super(SSF3, self).__init__()
        self.down1 = BaseConv(in_channels//2, 256, 3, stride=4, padding=0)
        self.down2 = BaseConv(in_channels, 256, 3, stride=2, padding=1)
        self.conv = BaseConv(in_channels*2, 256, 1, padding=0)
        self.ssa = SSAtt(8, 256)

    def forward(self, x1, x2, x3):
        x1 = self.down1(x1)
        x2 = self.down2(x2)
        x3 = self.conv(x3)
        # pdb.set_trace()
        x = self.ssa(x1, x2, x3)
        return x

class RAB(nn.Module):
    def __init__(self, in_channels):
        super(RAB, self).__init__()
        self.conv1 = BaseConv(in_channels, in_channels, 3, padding=1)
        # self.conv3 = BaseConv(in_channels, in_channels, 3, padding=1)
        self.conv2 = BaseConv(in_channels, 1, 1, padding=0, activation=nn.Sigmoid())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x1 * x2
        x2 += x
        return x2


class HRB(nn.Module):
    def __init__(self, in_channels):
        super(HRB, self).__init__()
        self.ssf1 = SSF1(in_channels)
        self.ssf2 = SSF2(in_channels)
        self.ssf3 = SSF3(in_channels)
        self.rab1 = RAB(64)
        self.rab2 = RAB(128)
        self.rab3 = RAB(256)

    def forward(self, x1, x2, x3):
        x1 = self.ssf1(x1, x2, x3)
        x2 = self.ssf2(x1, x2, x3)
        x3 = self.ssf3(x1, x2, x3)
        x1 = self.rab1(x1)
        x2 = self.rab2(x2)
        x3 = self.rab3(x3)
        return x1, x2, x3


if __name__ == '__main__':
    input = torch.randn(4, 3, 512, 512).cuda()
    model = HKINet().cuda()
    output = model(input)
    # output, attention = model(input)
    print(input.size())
    print(output.size())
    # print(attention.size())
