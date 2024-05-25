import torch
import torch.nn as nn


class MelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_for_identity=None):
        super(MelResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.conv_for_identity = conv_for_identity
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        x = self.layer1(x)

        x = self.layer2(x)
        if self.conv_for_identity:
            residual = self.conv_for_identity(residual)
        x += residual
        x = self.relu(x)

        return x


class MelResNet(nn.Module):
    # in_channels: For res_blocks
    def __init__(self, in_channels=64, layer1_kernel_size=10, layer1_stride=2,
                 max_pool1_kernel=3, max_pool1_stride=3,
                 res_unit1_out=64, res_unit2_out=32, res_unit3_out=32,
                 res_unit1_stride=1, res_unit2_stride=2, res_unit3_stride=2,
                 res_unit1_blocks=3, res_unit2_blocks=3, res_unit3_blocks=3):
        super(MelResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=layer1_kernel_size, stride=layer1_stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=max_pool1_kernel, stride=max_pool1_stride)
        self.in_channels = in_channels
        self.res_unit1 = self._make_layer(MelResBlock, res_unit1_out, res_unit1_blocks, stride=res_unit1_stride)
        self.res_unit2 = self._make_layer(MelResBlock, res_unit2_out, res_unit2_blocks, stride=res_unit2_stride)
        # self.res_unit3 = self._make_layer(MelResBlock, res_unit3_out, res_unit3_blocks, stride=res_unit3_stride)
        # self.res_unit4 = self._make_layer(MelResBlock, 16, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 64, 1)

    def _make_layer(self, res_block, out_channels, num_blocks, stride=1):
        # 1x1 convolution for identity skip connection
        conv_for_identity = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels))

        # The first res_block
        layers = [res_block(self.in_channels, out_channels, stride, conv_for_identity)]
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(res_block(self.in_channels, out_channels, conv_for_identity=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.res_unit1(x)
        # print(x.shape)
        x = self.res_unit2(x)
        # print(x.shape)
        # x = self.res_unit3(x)
        # print(x.shape)
        # x = self.res_unit4(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
