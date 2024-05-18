import torch


class MelConfig():
    def __init__(self):
        self.identity = "crisis"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        self.num_epochs = 30
        self.num_mels = 256
        self.hop_length = 512

        # Dataloader parameters
        self.val_size = 0.2
        self.batch_size = 64
        self.drop_last = True
