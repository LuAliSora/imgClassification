import torch
from torch import nn
from torch.nn import functional as F

import torchvision

from pathlib import Path

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)#stride=2
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            # print("Conv1X1(stride=2):")
            # print("Before:",X.shape)
            X = self.conv3(X)
            # print("After:",X.shape)
        # print(X[0][0][0][0])
        # print(Y[0][0][0][0])
        Y += X
        # print(Y[0][0][0][0])
        return F.relu(Y)

class ResNet_main(nn.Module):
    def resnet_block(self, input_channels, num_channels, num_residuals,
                    first_block=False):
        blk = []#
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk
    
    def __init__(self, input_channels, tagNum):
        super().__init__()
        # print(input_channels,tagNum)
        b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, tagNum))

    def forward(self,x):
        return self.net(x)
    
def ResNet_transL(tagNum, only_fc=True):
    model=torchvision.models.resnet18(weights="IMAGENET1K_V1")
    if only_fc:
        for param in model.parameters():
            param.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, tagNum, bias=True)
    return model


def modelLoad(tagNum, fileSave, device, lr):
    # model=ResNet_main(input_channels=3, tagNum=tagNum).to(device)
    model=ResNet_transL(tagNum=tagNum).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    baseEpoch=0
    if Path(fileSave).is_file():
        saveState=torch.load(fileSave, weights_only=True)
        model.load_state_dict(saveState['model_state'])
        optimizer.load_state_dict(saveState['optim_state'])
        baseEpoch=saveState["epoch"]
        print("model_load!")
    return model, optimizer, baseEpoch