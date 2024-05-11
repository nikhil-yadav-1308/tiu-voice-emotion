import torch
import torch.nn as nn


# Define a simple CNN model
class FreqCNN(nn.Module):
    def __init__(self):
        super(FreqCNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=4)
        self.batch_norm1 = nn.BatchNorm1d(num_features=64)

        self.layer2 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=10, stride=2)
        self.batch_norm2 = nn.BatchNorm1d(num_features=16)

        self.fc1 = nn.Linear(16 * 17, 32)
        self.batch_norm3 = nn.BatchNorm1d(num_features=32)

        self.fc3 = nn.Linear(32, 1)
        self.double()


    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        x = self.relu(x)
        # print(x.shape)
        x = nn.MaxPool1d(kernel_size=5)(x)
        x = self.batch_norm1(x)
        # print(x.shape)

        x = self.layer2(x)
        x = self.relu(x)
        # print(x.shape)
        x = nn.MaxPool1d(kernel_size=4)(x)
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
