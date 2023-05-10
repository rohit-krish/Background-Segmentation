'''
Created on Thu May  04 12:02:59 2023
@src: https://github.com/yassouali/pytorch-segmentation/blob/8b8e3ee20a3aa733cb19fc158ad5d7773ed6da7f/models/segnet.py
'''

from torch import nn
from torchvision import models
from warnings import filterwarnings

filterwarnings('ignore')


class SegNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False):
        super().__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        # adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1
            )

        # encoder, vgg without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [
            i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)
        ]
        # replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # when reveersing, we also reversed conv->batchN->relu, correct it
        decoder = [
            item for i in range(0, len(decoder), 3)
            for item in decoder[i:i+3][::-1]
        ]
        # replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(
                        module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1
                    )

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(
            *decoder[33:],
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(
            self.stage1_decoder, self.stage2_decoder,
            self.stage3_decoder, self.stage4_decoder, self.stage5_decoder
        )

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.weight.data.zero_()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):
        # encoder
        x = self.stage1_encoder(x)
        x1_shape = x.shape
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_shape = x.shape
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_shape = x.shape
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_shape = x.shape
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_shape = x.shape
        x, indices5 = self.pool(x)

        # decoder
        x = self.unpool(x, indices=indices5, output_size=x5_shape)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_shape)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_shape)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_shape)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_shape)
        x = self.stage5_decoder(x)

        return x


if __name__ == '__main__':
    from torch import rand
    model = SegNet(num_classes=2)
    x = rand(10, 3, 224, 224)
    print(model(x).shape)
