import torch
import torch.nn as nn
from config import config

class CNN(nn.module):
    def __init__(self, config, *args):
        super().__init__()


        self.conv1 = nn.Sequential([
                nn.Conv2d(in_channels = 1,
                            out_channels = config.out_channels_1,
                            kernel_size = config.kernel_size,
                            #stride = config.CNN.stride,
                            #padding=config.CNN.padding,
                            #dilation=config.CNN.dilation
                            ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=config.max_pool,
                            #  stride=config.stride,
                            #  padding=config.padding,
                            #  dilation=config.dilation
                            ),
                nn.BatchNorm1d(num_features=config.out_channels_1)
                ])
        
        self.conv2 = 0
        


    