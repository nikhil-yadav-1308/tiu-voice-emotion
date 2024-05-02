class config():
    def __init__(self):
        self.identity = "crisis"
        
        # Dataloader parameters
        self.val_size = 0.2
        self.batch_size = 64
        self.drop_last = True
