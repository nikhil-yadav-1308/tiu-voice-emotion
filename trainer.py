import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import config
config = config()

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

def trainer(train_dataloader):
    # Setup model, optimizer, and criterion
    model = torch.nn.LSTM(input_size=config.num_mels,
                        hidden_size=config.hidden_size, 
                        batch_first=True,
                        device=config.device)
    
    regression_layer = nn.Linear(config.hidden_size, 1)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    train_loss = []
    # Iterate over epochs
    for i in range(config.num_epcohs):
        epoch_train_loss = model_train(train_dataloader, model, regression_layer, optimizer, criterion)
        if i%5==0:
            print(f"Epoch {i}\nTrain loss: {epoch_train_loss:.4f}")
        train_loss.append(epoch_train_loss)
    
    
    plt.plot(np.arange(1, len(train_loss)+1), train_loss)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    

def model_train(train_dataloader, model, regression_layer, optimizer, criterion):
    train_loss = 0
    batch_idx = np.arange(64)

    model.train()
    model.parameters()

    for batch, labels in train_dataloader:
        # >> a batch is (batch_size, num_mels, global_max_seq_length) e.g. (64, 128, 278)
            # global_max_seq_length refers to the longest sequence out of all spectrograms
        optimizer.zero_grad()

        batch = batch.transpose(1, -1)
        # >> (batch_size, global_max_seq_length, num_mels)

        labels = labels.to(config.device).to(torch.float32)
        batch_packed = get_packed_padded_sequence(batch).to(config.device)
        # print(f"batch packed: {batch_packed.data.shape}")
        # >> (num_timesptes, num_mels), concatenates all non-padded timesteps

        outputs_packed, (hidden, cell_state) = model(batch_packed)
        outputs, lengths = pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = outputs.to(config.device)
        # print(f"outputs shape: {outputs.data.shape}")
        # >> (batch_size, max_seq_length, hidden_size)
            # Note that max_seq_length is the longest sequence within this batch, not the global max        

        # To derive the last output for batch i you must select the sequence index of length[i]-1 
            # e.g.  outputs[12, lengths[12]-1, :]  returns the 12th batch last output with shape hidden_size

        last_outputs = outputs[batch_idx, lengths-1, :]
        # >> (batch_size, hidden_size), contains the last output of each batch
        
        predictions = regression_layer(last_outputs)
        # >> (batch_size, 1), contains a single valence prediction for each batch 
        predictions = predictions.squeeze(1)
        # >> (batch_size)
        loss = criterion(predictions, labels)
        
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return train_loss/len(train_dataloader)

def get_packed_padded_sequence(batch):
    # batch comes in shape (batch_size, max_seq_length, num_mels)
    # Compute the length of each batch, defined as the number of entries in the last dimension that are not 0
    lengths = (batch[:, :, 0]!=0).sum(dim=-1) # >> (64) where each entry contains the length of batch n 
    # print(lengths)
    # print(f"max sequence length: {max(lengths)}")
    # Clip the sequence length dimension to be as long as the longest sequence within batch
    batch = batch[:, :max(lengths), :]

    batch_packed = pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)
    
    return batch_packed.float()
    # # print(train_spectogram_packed)
    # # print(len(train_spectogram_packed.data))
    # assert len(train_spectogram_packed.batch_sizes) == train_spectogram.shape[1]

    # # Recover the tensors
    # train_spectogram, lens_unpacked = pad_packed_sequence(train_spectogram_packed, batch_first=True)
    # train_spectogram = train_spectogram.transpose(1, -1) # >> (B, num_mels, T)
