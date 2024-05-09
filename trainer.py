import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from config import config
config = config()

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

def trainer(train_dataloader, val_dataloader):
    """ Loops over all training epochs, saves the training and validation losses """
    start_time = datetime.now()
    print("="*14, " Training started ", "="*14)
    # Setup model, optimizer, and criterion
    model = nn.LSTM(input_size=config.num_mels, hidden_size=config.hidden_size, proj_size=1, bidirectional=False, batch_first=True, device=config.device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []
    # Iterate over epochs
    for i in range(1, config.num_epochs+1):
        # model_train returns the epochs average training loss
        epoch_train_loss = model_train(train_dataloader, model, optimizer, criterion)
        # model_evaluate returns the epochs average validation loss
        epoch_val_loss = model_evaluate(val_dataloader, model, criterion)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        if i%5==0:
            print(f"Epoch {i}:\nTrain loss: {epoch_train_loss:.4f}\t|\tVal loss: {epoch_val_loss:.4f}")
    print("="*14, f" Training finished ", "="*14)
    
    # Save the model, losses, and a plot of the training and validation loss
    output_path = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f"Saving model and losses to: {output_path}")

    # Saving model
    os.makedirs(os.path.join(output_path, "model"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join(output_path, "model", "ckp_last.pt"))

    # Saving losses and a plot
    os.makedirs(os.path.join(output_path, "losses"), exist_ok=True)
    torch.save(torch.tensor(train_loss), f=os.path.join(output_path, "losses", "train.pt"))
    torch.save(torch.tensor(val_loss), f=os.path.join(output_path, "losses", "val.pt"))
    
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(train_loss)+1), train_loss, color="#0074D9", label="train loss")
    plt.plot(np.arange(1, len(val_loss)+1), val_loss, color="#FF851B", label="val loss")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Losses over Training')
    plt.legend()
    plt.savefig(os.path.join(output_path, "losses", "loss_plot.png"))
    
    print(f"Training time: {datetime.now()-start_time}")


def model_train(train_dataloader, model, optimizer, criterion):
    """ Runs one epoch of training by looping over batches in train_dataloader, 
        computing the loss and updating the models parameters   """
    
    train_loss = 0
    batch_idx = np.arange(config.batch_size) # Needed for indexing
    model.train() # Put the model in training mode

    for i, (batch, labels) in enumerate(train_dataloader, start=1):
        # >> a batch is (batch_size, num_mels, global_max_seq_length)
        # global_max_seq_length refers to the longest sequence out of all spectrograms
        
        optimizer.zero_grad() # Zero out gradients

        labels = labels.to(config.device).to(torch.float32)
        batch = batch.transpose(1, -1).to(config.device)
        # >> (batch_size, global_max_seq_length, num_mels)

        # Get a PackedSequence object of this batch 
        batch_packed = get_packed_padded_sequence(batch).to(config.device)
        # print(f"Shape of packed batch: {batch_packed.data.shape}")

        # Call the LSTM model on the packed batch, returns a packed batch
        outputs_packed, (h_t, c_t) = model(batch_packed)

        # Unpack the packed outputs 
        outputs, lengths = pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = outputs.to(config.device)
        # print(f"Shape of unpacked outputs: {outputs.shape}")
        # >> (batch_size, max_seq_length, proj_size)

        # To derive the last output for batch i, select the index: length[i]-1
        predictions = outputs[batch_idx, lengths-1, 0]
        # print(f"Shape of predictions: {predictions.shape}")
        # >> (batch_size, 1)

        # Compute the MSE loss
        loss = criterion(predictions, labels)
        train_loss += loss.item()
        # print(f"Batch {i} train loss: {loss:.4f}")
        
        # Compute gradients and update model parameters
        loss.backward()
        optimizer.step()

    # Normalize total MSE by the number of batches
    return train_loss/i

def model_evaluate(val_dataloader, model, criterion):
    """ Evaluates the model's performance on the validation set """

    batch_idx = np.arange(config.batch_size) # Needed for indexing
    val_loss = 0
    model.eval() # Put the model in evaluation mode

    with torch.no_grad():
        for i, (batch, labels) in enumerate(val_dataloader, start=1):
            labels = labels.to(torch.float32).to(config.device)
            batch = batch.transpose(1, -1).to(config.device)
            # >> (batch_size, global_max_seq_length, num_mels)

            # Get a PackedSequence object of this batch 
            batch_packed = get_packed_padded_sequence(batch).to(config.device)
            # print(f"Shape of packed batch: {batch_packed.data.shape}")

            # Call the LSTM model on the packed batch, returns a packed batch
            outputs_packed, (h_t, c_t) = model(batch_packed)

            # Unpack the packed outputs 
            outputs, lengths = pad_packed_sequence(outputs_packed, batch_first=True)
            outputs = outputs.to(config.device)
            # print(f"Shape of unpacked outputs: {outputs.shape}")
            # >> (batch_size, max_seq_length, proj_size)

            # To derive the last output for batch i, select the index: length[i]-1
            predictions = outputs[batch_idx, lengths-1, 0]

            # Compute the MSE loss
            loss = criterion(predictions, labels)
            val_loss += loss.item()

        # Normalize MSE by the number of batches
        return val_loss/i

def get_packed_padded_sequence(batch):
    """ Takes a batch of shape (batch_size, max_seq_length, num_mels) 
        and turns it into a PackedSequence object for efficient training """
    
    # Compute the length of non-padded entries for each batch
    lengths = (batch[:, :, 0]!=0).sum(dim=-1).long()
    lengths = lengths.cpu() # Must be on the cpu
    # >> (64) where each entry contains the non-padded length of that batch
    # print(f"max sequence length for this batch: {max(lengths)}")

    # Clip the sequence length to be as long as the longest sequence within this batch
    batch = batch[:, :max(lengths), :]

    # Call pack_padded_sequence with the batch and lengths
    batch_packed = pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)
    
    return batch_packed.float()
    # The resulting batch_packed is of shape: (timesteps, num_mels) 
    # Where timesteps is the total number of non-padded timesteps over all samples in the batch