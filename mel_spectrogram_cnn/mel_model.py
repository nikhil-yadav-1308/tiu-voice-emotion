import torch
import torch.nn as nn


# Define a simple CNN model
class MelCNN(nn.Module):
    def __init__(self):
        super(MelCNN, self).__init__()
        self.relu = nn.ReLU()

        layer1_kernel_size = 10
        layer1_stride = 4
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=layer1_kernel_size, stride=layer1_stride)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        layer1_output = ((512 - layer1_kernel_size) // layer1_stride + 1, (683 - layer1_kernel_size) // layer1_stride + 1)

        max_pool1_kernel = 4
        max_pool1_stride = 4
        self.max_pool1 = nn.MaxPool2d(kernel_size=max_pool1_kernel, stride=max_pool1_stride)
        max_pool1_output = ((layer1_output[0] - max_pool1_kernel)//max_pool1_stride + 1, (layer1_output[1] - max_pool1_kernel) // max_pool1_stride + 1)

        layer2_kernel_size = 5
        layer2_stride = 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=layer2_kernel_size, stride=layer2_stride)
        self.batch_norm2 = nn.BatchNorm2d(num_features=16)
        layer2_output = ((max_pool1_output[0] - layer2_kernel_size)//layer2_stride + 1, (max_pool1_output[1] - layer2_kernel_size) // layer2_stride + 1)

        max_pool2_kernel = 2
        max_pool2_stride = 2
        self.max_pool2 = nn.MaxPool2d(kernel_size=max_pool2_kernel, stride=max_pool2_stride)
        max_pool2_output = ((layer2_output[0] - max_pool2_kernel)//max_pool2_stride + 1, (layer2_output[1] - max_pool2_kernel) // max_pool2_stride + 1)

        self.fc1 = nn.Linear(16 * max_pool2_output[0] * max_pool2_output[1], 32)
        self.batch_norm3 = nn.BatchNorm1d(num_features=32)

        self.fc3 = nn.Linear(32, 1)
        self.float()

    def forward(self, x):
        # identity = x
        # print(x.shape)
        x = self.layer1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.max_pool1(x)
        x = self.batch_norm1(x)
        # print(x.shape)

        x = self.layer2(x)
        # x += identity
        x = self.relu(x)
        # print(x.shape)
        x = self.max_pool2(x)
        x = self.batch_norm2(x)

        # print(x.shape)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)

        x = self.batch_norm3(x)
        # print(x.shape)
        x = self.fc3(x)
        x = self.relu(x)
        # print(x.shape)
        return x


class MelCNNAdapt(nn.Module):
    def __init__(self, layer1_kernel_size=12, layer1_stride=2, layer1_out_channels=128,
                 max_pool1_kernel=3, max_pool1_stride=3,
                 layer2_kernel_size=5, layer2_stride=2, layer2_out_channels=32,
                 adapt_pool1=8, adapt_pool2=8,
                 fc1_out=32):
        super(MelCNNAdapt, self).__init__()
        self.relu = nn.ReLU()

        self.layer1 = nn.Conv2d(in_channels=1, out_channels=layer1_out_channels, kernel_size=layer1_kernel_size, stride=layer1_stride)
        self.batch_norm1 = nn.BatchNorm2d(num_features=layer1_out_channels)

        self.max_pool1 = nn.MaxPool2d(kernel_size=max_pool1_kernel, stride=max_pool1_stride)

        self.layer2 = nn.Conv2d(in_channels=layer1_out_channels, out_channels=layer2_out_channels, kernel_size=layer2_kernel_size, stride=layer2_stride)
        self.batch_norm2 = nn.BatchNorm2d(num_features=layer2_out_channels)

        self.adapt_pool = nn.AdaptiveAvgPool2d((adapt_pool1, adapt_pool2))

        self.fc1 = nn.Linear(layer2_out_channels * adapt_pool1 * adapt_pool2, fc1_out)
        self.batch_norm3 = nn.BatchNorm1d(num_features=fc1_out)

        self.fc2 = nn.Linear(fc1_out, 1)
        self.float()

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        # print(x.shape)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.adapt_pool(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x
