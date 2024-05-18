import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from mel_config import MelConfig
from mel_dataloader import data_generator, MelTestDataset
from mel_model import MelCNN, MelCNNAdapt
from mel_resnet import ResNet
from time import time

config = MelConfig()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Call spectrograms_with_csv to create the padded spectrograms and csv
input_dir = os.path.join("../train")  # Directory to raw audio files
output_dir = "../mel_spectrograms"  # Directory to store padded spectrograms
csv_path = "mel_spectrograms.csv"  # Path name to CSV file that holds paths and label
test_csv_path = "mel_spectrograms_test.csv"
train_loader, val_loader, test_loader = data_generator(spectrogram_csv_path=csv_path, test_csv_path=test_csv_path)


def train(model, criterion, optimizer, num_epochs):
    print("=" * 14, " Training started ", "=" * 14)
    print(f'Device: {config.device}')
    train_loss_list = []
    val_loss_list = []
    best_loss = 100
    # Training loop
    for epoch in range(num_epochs):
        train_loss_epoch = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            targets = targets.unsqueeze(1).float()
            inputs = F.normalize(inputs, dim=0)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= (len(train_loader))
        print(f"Training Set: Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_epoch:.4f}")
        train_loss_list.append(train_loss_epoch)

        with torch.no_grad():
            val_loss_epoch = 0
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(config.device), val_targets.to(config.device)
                val_targets = val_targets.unsqueeze(1)
                val_inputs = F.normalize(val_inputs, dim=0)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_loss_epoch += val_loss.item()
            val_loss_epoch /= (len(val_loader))
            print(f"Validation Set: Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss_epoch:.4f}")
            val_loss_list.append(val_loss_epoch)

        if val_loss_epoch < best_loss:
            data = []
            with torch.no_grad():
                for test_inputs, files in test_loader:
                    test_inputs = test_inputs.to(config.device)
                    test_inputs = F.normalize(test_inputs, dim=0)
                    test_outputs = model(test_inputs)
                    test_outputs = np.array(test_outputs.squeeze(1).cpu())
                    data.extend(list(zip(files, test_outputs)))
            data = pd.DataFrame(data, columns=['ID', 'Label'])
            data.to_csv('results.csv', index=False)
            best_loss = val_loss_epoch

    print(val_outputs.reshape(1, -1))
    print(val_targets.reshape(1, -1))
    return train_loss_list, val_loss_list, best_loss


def tune_models():
    # Setting Hyperparameters
    num_epochs = 60
    for i in range(100):

        lr = np.random.choice([0.01, 0.001, 0.005, 0.0001])
        weight_decay = np.random.choice([0.001, 0.005, 0.0001])

        layer1_kernel_size = np.random.choice([8,10,12,15, 20])
        layer1_stride = np.random.choice([2,3,4,5])
        layer1_out_channels = np.random.choice([16,32,64,128])
        max_pool1_kernel = np.random.choice([3,4,5])
        max_pool1_stride = np.random.choice([2,3])
        layer2_kernel_size = np.random.choice([3,5,8,10])
        layer2_stride = np.random.choice([2,3,4])
        layer2_out_channels = np.random.choice([16,32,64])
        adapt_pool1 = np.random.choice([4,8,10])
        adapt_pool2 = np.random.choice([4,8,10])
        fc1_out = np.random.choice([8, 16, 32])

        with open("output.txt", "a") as f:
            print("===================== Training =======================", file=f)
            print(f"Learning Rate: {lr}", file=f)
            print(f"Weight Decay: {weight_decay}", file=f)

            print("\nLayer 1:", file=f)
            print(f"Kernel size: {layer1_kernel_size}", file=f)
            print(f"Stride: {layer1_stride}", file=f)
            print(f"Output channels: {layer1_out_channels}", file=f)

            print("\nMax-pooling 1:", file=f)
            print(f"Kernel size: {max_pool1_kernel}", file=f)
            print(f"Stride: {max_pool1_stride}", file=f)

            print("\nLayer 2:", file=f)
            print(f"Kernel size: {layer2_kernel_size}", file=f)
            print(f"Stride: {layer2_stride}", file=f)
            print(f"Output channels: {layer2_out_channels}", file=f)

            print("\nAdaptive pooling:", file=f)
            print(f"Adapt pool 1 size: {adapt_pool1}", file=f)
            print(f"Adapt pool 2 size: {adapt_pool2}", file=f)

            print("\nFully connected layer:", file=f)
            print(f"Output units: {fc1_out}", file=f)

            print(f"Output units: {fc1_out}", file=f)

        model = MelCNNAdapt(layer1_kernel_size=layer1_kernel_size, layer1_stride=layer1_stride, layer1_out_channels=layer1_out_channels,
                            max_pool1_kernel=max_pool1_kernel, max_pool1_stride=max_pool1_stride,
                            layer2_kernel_size=layer2_kernel_size, layer2_stride=layer2_stride, layer2_out_channels=layer2_out_channels,
                            adapt_pool1=adapt_pool1, adapt_pool2=adapt_pool2,
                            fc1_out=fc1_out).to(config.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        try:
            train_loss_list, val_loss_list, best_loss = train(model, criterion, optimizer, num_epochs)

            with open("output.txt", "a") as f:
                print(f'\nBest Loss: {best_loss}', file=f)

            # plt.plot(range(num_epochs-5), train_loss_list[5:])
            # plt.plot(range(num_epochs-5), val_loss_list[5:])
            # plt.show()
        except:
            continue


tune_models()


# data = []
# with torch.no_grad():
#     for test_inputs, files in test_loader:
#         test_inputs = test_inputs.to(config.device)
#         test_inputs = F.normalize(test_inputs, dim=0)
#         test_outputs = model(test_inputs)
#         # print('asdsad')
#         test_outputs = np.array(test_outputs.squeeze(1).cpu())
#         data.extend(list(zip(files, test_outputs)))
#
# data = pd.DataFrame(data, columns=['ID', 'Label'])
# data.to_csv('results.csv', index=False)
# print(data)
