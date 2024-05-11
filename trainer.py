import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
config = config()

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)

class EarlyStopping:
    """ Early stopping to stop training when the validation loss doesn't improve """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        self.best_loss = val_loss

class LSTMCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels=128, kernel_size=3):
        super(LSTMCNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=num_channels,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels * 2,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * 2, 1)

    def forward(self, x):
        # LSTM layer
        packed_output, (h_t, c_t) = self.lstm(x)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        output = output.transpose(1, 2)

        # Convolutional layers
        x = torch.relu(self.conv1(output))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.mean(x, dim=2)  # Or max pooling

        # Fully connected layer
        x = self.fc(x)
        return x


def trainer(train_dataloader, val_dataloader):
    """ Loops over all training epochs, saves the training and validation losses """
    start_time = datetime.now()
    print("="*14, " Training started ", "="*14)
    # Setup model, optimizer, and criterion
    model = LSTMCNN(input_size=config.num_mels, hidden_size=config.hidden_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []
    early_stopping = EarlyStopping(patience=25, delta=0.005)

    # Iterate over epochs
    for epoch in tqdm(range(1, config.num_epochs + 1), desc="Training Progress", unit="epoch"):
        # model_train returns the epoch's average training loss
        epoch_train_loss = model_train(train_dataloader, model, optimizer, criterion)
        # model_evaluate returns the epoch's average validation loss
        epoch_val_loss = model_evaluate(val_dataloader, model, criterion)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        print(f"Epoch {epoch}:\nTrain loss: {epoch_train_loss:.4f}\t|\tVal loss: {epoch_val_loss:.4f}")

        # Check early stopping
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after epoch {epoch}")
            break

    print("="*14, f" Training finished ", "="*14)
    
    # Save the model, losses, and a plot of the training and validation loss
    output_path = f"runs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
    train_loss = 0
    batch_idx = np.arange(config.batch_size)  # Needed for indexing
    model.train()  # Put the model in training mode

    for i, (batch, labels) in tqdm(enumerate(train_dataloader, start=1), desc="Steps", unit="batch"):
        optimizer.zero_grad()  # Zero out gradients

        labels = labels.to(config.device).to(torch.float32)
        batch = batch.transpose(1, -1).to(config.device)
        # >> (batch_size, global_max_seq_length, num_mels)

        # Get a PackedSequence object of this batch 
        batch_packed = get_packed_padded_sequence(batch).to(config.device)

        # Call the LSTM model on the packed batch, returns predictions directly
        predictions = model(batch_packed)

        # Compute the MSE loss
        loss = criterion(predictions.squeeze(), labels)
        train_loss += loss.item()
        
        # Compute gradients and update model parameters
        loss.backward()
        optimizer.step()

    # Normalize total MSE by the number of batches
    return train_loss / i


def model_evaluate(val_dataloader, model, criterion):
    """ Evaluates the model's performance on the validation set """
    val_loss = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for i, (batch, labels) in tqdm(enumerate(val_dataloader), desc="Validation Progress", unit="batch"):
            labels = labels.to(config.device).to(torch.float32)
            batch = batch.transpose(1, -1).to(config.device)
            
            # Get a PackedSequence object of this batch
            batch_packed = get_packed_padded_sequence(batch).to(config.device)

            # Call the LSTM model on the packed batch, which returns predictions directly
            predictions = model(batch_packed)

            # Since model outputs a single tensor, adjust labels and predictions to match dimensions
            predictions = predictions.squeeze()  # Ensure predictions are the right shape
            loss = criterion(predictions, labels)
            val_loss += loss.item()

        # Normalize MSE by the number of batches
        return val_loss / len(val_dataloader)

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