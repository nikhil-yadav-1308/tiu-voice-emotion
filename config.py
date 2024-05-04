class config():
    def __init__(self):
        self.identity = "crisis"
        
        self.seed = 132

        # Dataloader parameters
        self.val_size = 0.2
        self.batch_size = 64
        self.drop_last = True
