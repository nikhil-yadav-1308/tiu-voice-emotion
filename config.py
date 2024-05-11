import torch

class config():
    def __init__(self):
        self.identity = "crisis"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 132
        self.num_epochs = 1000
        self.num_mels = 128
        self.hidden_size = 256

        # Dataloader parameters
        self.val_size = 0.1
        self.batch_size = 32
        self.drop_last = True