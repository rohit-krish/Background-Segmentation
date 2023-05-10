'''
Created on Thu May  04 17:08:06 2023
@src: https://github.com/yassouali/pytorch-segmentation/blob/8b8e3ee20a3aa733cb19fc158ad5d7773ed6da7f/models/segnet.py
'''

from torch import nn
from torchvision import models
import torch.nn.functional as F

from warnings import filterwarnings
filterwarnings('ignore')


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels//4, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels//4)
        self.conv2 = nn.ConvTranspose2d(
            in_channels//4, in_channels//4, kernel_size=2, stride=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.conv3 = nn.Conv2d(in_channels//4, in_channels//2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, in_channels//2, kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(in_channels//2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LastBottleneck(nn.Module):
    def __init__(self, in_channels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels//4, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels//4)
        self.conv2 = nn.Conv2d(
            in_channels//4, in_channels // 4, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.conv3 = nn.Conv2d(in_channels//4, in_channels//4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels//4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet50_Weights.DEFAULT

class SegResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        encoder = list(resnet50.children())

        if in_channels != 3:
            encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1)

        encoder[3].return_indices = True  # MaxPool2D

        # encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)

        # decoder
        resnet50_untrained = models.resnet50(pretrained=False)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)

        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(
                *new_block, DecoderBottleneck(channels[i])
            ))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        # encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # decoder
        x = self.decoder(x)
        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)

        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


if __name__ == '__main__':
    from torch import randn
    x = randn(1, 3, 224, 224)
    print(SegResNet(2)(x).shape)

    # x = randn(1, 2048, 7, 7)
    # print(DecoderBottleneck(2048)(x))