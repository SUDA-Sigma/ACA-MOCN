import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
#from torchsummary import summary
import collections
from torchvision import models
import os

# from .BaseModel.MCC_BaseModel import MCC_BaseModel
from torch import optim
from torch.optim.lr_scheduler import StepLR
# from misc.mcmse_loss import MCMSE_Loss

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()

 
    def forward(self, x):
        out = self.relu1(self.fc1(self.avg_pool(x)))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class BranchNet(nn.Module):
    def __init__(self,in_channels = 512,out_channels = 512):
        super(BranchNet, self).__init__()
        out_channel1 = int(0.1875 * out_channels)
        out_channel2 = int(0.25 * out_channels)
        out_channel3 = int(0.375 * out_channels)

        self.atten=SEAttention()
        self.SPatten=SpatialAttention()

        self.b1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel1, kernel_size=1, groups=1),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b1_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel3, kernel_size=1, groups=1),

            nn.ReLU(),
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, padding=2, dilation=2),
        )
        # 1x1 conv -> 5x5 conv branch
        self.b1_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel2, kernel_size=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, padding=3, dilation=3),
        )

        self.b1_4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channel1, kernel_size=1, groups=1),
        )
    def forward(self, x):
       
        f1 = self.b1_1(x)
        f2 = self.b1_2(x)
        f3 = self.b1_3(x)
        f4 = self.b1_4(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        # output = self.atten(output)* output + output + x
        output = torch.add((self.atten(output) * output + output),self.SPatten(x))
        return output
        
class MFANet(nn.Module):

    def __init__(self, load_weights=False):

        super(MFANet, self).__init__()
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,512]
        self.branch1 = BranchNet(in_channels = 512,out_channels = 512)
        self.branch2 = BranchNet(in_channels = 512,out_channels = 512)
        self.branch3 = BranchNet(in_channels = 512,out_channels = 512)
        self.branch4 = BranchNet(in_channels = 512,out_channels = 512)
        self.branch5 = BranchNet(in_channels = 512,out_channels = 512)
        self.branch6 = BranchNet(in_channels = 512,out_channels = 512)
        self.output_layer = nn.Conv2d(64, 6, kernel_size=1)
        self.relu = nn.ReLU()
        self.features = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation = 1),
            nn.ReLU(),          
            nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation = 1), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels= 64,  kernel_size=9, stride=8, padding=1,output_padding=1), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation = 1),       
            nn.ReLU(),  
  
        )
        self.relu = nn.ReLU(inplace=True)
        self.frontend = make_layers(self.frontend_feat)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

        #------- init weights --------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #-----------------------------#

    def forward(self, input):
        fv = self.frontend(input)
        # 过一层
        x1 = self.branch1(fv)
        # x2 = self.branch2(x1)
        # x3 = self.branch3(x2)
        # x4 = self.branch4(x3)
        # x5 = self.branch5(x4)
        # x6 = self.branch6(x5)
        x = self.features(x1) 
        x = self.output_layer(x)     
        return x
        
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):  #dialtion扩张
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# class MFANet(MCC_BaseModel):
#     def __init__(self, dataloader, cfg, dataset_cfg, pwd):
#         super(MFANet, self).__init__(dataloader, cfg, dataset_cfg, pwd)

#         self.net = Net()
#         self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
#         # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
#         self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

#         if torch.cuda.device_count() > 1:
#             self.net = nn.DataParallel(self.net, device_ids=cfg.GPU_ID)

#         self.net.to(self.device)

#         if len(self.gpus) >= 1:
#             self.loss_mse_fn = nn.MSELoss().to(self.device)

if __name__=="__main__":
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GBSNet().to(device)
    model = MFANet()
    input_img=torch.ones((1,3,224,224)).to("cuda" if torch.cuda.is_available() else "cpu")
    out = model(input_img)
    print(out.shape)
    # torch.save(model, 'net1.pkl')
    #summary(model, (3,640,480))   
   
    
