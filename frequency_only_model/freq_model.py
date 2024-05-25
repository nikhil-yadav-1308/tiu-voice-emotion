import torch
import torch.nn as nn


# Define a simple CNN model
class FreqCNN(nn.Module):
    def __init__(self):
        super(FreqCNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=10, stride=4)
        self.batch_norm1 = nn.BatchNorm1d(num_features=128)

        self.layer2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=10, stride=2)
        self.batch_norm2 = nn.BatchNorm1d(num_features=32)

        self.layer3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, stride=1)
        self.batch_norm3 = nn.BatchNorm1d(num_features=16)
        #
        # self.fc1 = nn.Linear(16 * 27, 32)
        # self.batch_norm4 = nn.BatchNorm1d(num_features=32)

        self.fc3 = nn.Linear(16 * 12, 1)
        self.double()

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.relu(x)
        # print(x.shape)
        x = nn.MaxPool1d(kernel_size=3)(x)
        x = self.batch_norm1(x)
        # print(x.shape)

        x = self.layer2(x)
        x = self.relu(x)
        # print(x.shape)
        x = nn.MaxPool1d(kernel_size=3)(x)
        x = self.batch_norm2(x)

        x = self.layer3(x)
        x = self.relu(x)
        # print(x.shape)
        x = nn.MaxPool1d(kernel_size=3)(x)
        x = self.batch_norm3(x)

        # print(x.shape)

        x = x.view(x.size(0), -1)  # Flatten
        # x = self.fc1(x)
        # x = self.relu(x)
        #
        # x = self.batch_norm4(x)
        # print(x.shape)
        x = self.fc3(x)
        x = self.relu(x)
        # print(x.shape)
        return x
