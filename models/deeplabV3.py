'''
Created on Sat May  06 00:53:00 2023
@author: rohit krishna
resources: 
    - https://towardsdatascience.com/witnessing-the-progression-in-semantic-segmentation-deeplab-series-from-v1-to-v3-4f1dd0899e6e
    - https://github.com/kkatsy/DeepLabV3Plus/blob/main/deeplab.py
'''

# IN DEEPLAB_V3+ the backbone is Xception

from torch import nn
from torchvision import models
import torch


class DeepLavV3(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        # encoder
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # freeze pretrained layers
        for param in resnet.parameters():
            param.requires_grad = False

        # pre-residual layers
        self.in_conv = resnet.conv1
        self.in_bn = resnet.bn1
        self.in_relu = resnet.relu
        self.in_maxpool = resnet.maxpool
        self.begin_resnet_layers = nn.ModuleList([
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        ])

        # resnet high-level
        self.resnet_4_layers = nn.ModuleList([
            resnet.layer1, resnet.layer2, resnet.layer3  # , resnet.layer4
        ])
        features4 = self.begin_resnet_layers + self.resnet_4_layers
        self.resnet_4 = nn.Sequential(*features4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print(self.resnet_4_layers)

        # ASPP
        self.aspp_conv1 = nn.Conv2d(1024, 256, kernel_size=1, padding='same', bias=False)
        self.aspp_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.aspp_conv3_r6 = nn.Conv2d(256, 256, 3, padding=6, dilation=6, bias=False)
        self.aspp_conv3_r12 = nn.Conv2d(256, 256, 3, padding=12, dilation=12, bias=False)
        self.aspp_conv3_r18 = nn.Conv2d(256, 256, 3, padding=18, dilation=18, bias=False)
        self.aspp_pool = nn.AdaptiveAvgPool2d((14, 14))

        # ENCODER OUT
        # concat ASPP
        self.conv1_encoder = nn.Conv2d(5 * 256, 256, 1, bias=False)
        self.b1_encoder = nn.BatchNorm2d(256)

        # resnet low-level
        self.resnet_2_layers = nn.ModuleList([resnet.layer1]) #, resnet.layer2])
        features_2 = self.begin_resnet_layers + self.resnet_2_layers
        self.renset_2 = nn.Sequential(*features_2)

        # DECODER
        self.low_level_conv = nn.Conv2d(256, 256, 1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(256)
        
        # concat resnet + ASPP
        self.conv3_decoder = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.bn_decoder = nn.BatchNorm2d(256)

        # upsample by 4
        self.upsample4 = nn.Upsample(scale_factor=4)#, mode='bilinear')

        self.final_conv = nn.Conv2d(256, out_channels, 1)
    
    def forward(self, x):
        # encoder
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_relu(x)
        x = self.in_maxpool(x)
        pre_resnet = x
        
        for layer in self.resnet_4_layers:
            x = layer(x)
        
        x = self.aspp_conv1(x)
        x = self.aspp_bn(x)
        x = self.relu(x)
        aspp_conv1_output = x

        x = self.aspp_conv3_r6(x)
        x = self.aspp_bn(x)
        x = self.relu(x)
        aspp_conv3_r6_output = x

        x = self.aspp_conv3_r12(x)
        x = self.aspp_bn(x)
        x = self.relu(x)
        aspp_conv3_r12_output = x

        x = self.aspp_conv3_r18(x)
        x = self.aspp_bn(x)
        x = self.relu(x)
        aspp_conv3_r18_output = x

        aspp_pool_output = self.aspp_pool(x)

        aspp_pyramid = [
            aspp_conv1_output, aspp_conv3_r6_output,
            aspp_conv3_r12_output, aspp_conv3_r18_output,
            aspp_pool_output
        ]

        aspp_concat = torch.cat(aspp_pyramid, dim=1)

        x = self.conv1_encoder(aspp_concat)
        x = self.b1_encoder(x)
        x = self.relu(x)

        upsampled_encoder = self.upsample4(x)

        for layer in self.resnet_2_layers:
            pre_resnet = layer(pre_resnet)
        
        x = self.low_level_conv(pre_resnet)
        x = self.low_level_bn(x)
        x = self.relu(x)

        decoder_concat = torch.cat([x, upsampled_encoder], dim=1)

        x = self.conv3_decoder(decoder_concat)
        x = self.bn_decoder(x)
        x = self.relu(x)

        x = self.upsample4(x)
        
        return self.final_conv(x)


if __name__ == '__main__':
    model = DeepLavV3(1)
    x = torch.rand((1, 3, 224, 224))
    print(model(x).shape)
