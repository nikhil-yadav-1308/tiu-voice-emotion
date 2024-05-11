import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from freq_model import FreqCNN
from freq_dataloader import FrequencyTrainDataset, FrequencyTestDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd

# torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

data = FrequencyTrainDataset()
train_dataset, val_dataset = random_split(data, [int(0.8*len(data))+1, len(data)-int(0.8*len(data))-1])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


def train(model, criterion, optimizer, num_epochs):
    train_loss_list = []
    val_loss_list = []

    # Training loop
    for epoch in range(num_epochs):
        train_loss_epoch = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = F.normalize(inputs, dim=0)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= (len(train_loader))
        print(f"Training Set: Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss_epoch:.4f}")
        train_loss_list.append(train_loss_epoch)

        with torch.no_grad():
            val_loss_epoch = 0
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_inputs = F.normalize(val_inputs, dim=0)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_loss_epoch += val_loss.item()
            val_loss_epoch /= (len(val_loader))
            print(f"Validation Set: Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss_epoch:.4f}")
            val_loss_list.append(val_loss_epoch)

        if val_loss_epoch <= 0.575:
            break
    print(val_outputs.reshape(1, -1))
    print(val_targets.reshape(1, -1))
    return train_loss_list, val_loss_list


# Setting Hyperparameters
num_epochs = 30
model = FreqCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.005)

train_loss_list, val_loss_list = train(model, criterion, optimizer, num_epochs)

plt.plot(range(num_epochs-5), train_loss_list[5:])
plt.plot(range(num_epochs-5), val_loss_list[5:])
plt.show()

data = []
test_data = FrequencyTestDataset()
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
with torch.no_grad():
    for test_inputs, files in test_loader:
        test_inputs = test_inputs.to(device)
        test_inputs = F.normalize(test_inputs, dim=0)
        test_outputs = model(test_inputs)
        test_outputs = np.array(test_outputs.squeeze(1).cpu())
        data.extend(list(zip(files, test_outputs)))

data = pd.DataFrame(data, columns=['ID', 'Label'])
data.to_csv('results.csv', index=False)
print(data)


# print(next(iter(train_loader))[0].shape)