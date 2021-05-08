import numpy as np
import torch

class Trainer:
    """ Trainer class responsible for training the model """
    def __init__(self,
                 train_dataset,             # An instance of torch's Dataset class
                 validation_dataset,        # An instance of torch's Dataset class
                 model,                     # The model to use for training
                 epochs,                    # Number of epochs to train the model
                 validation_interval,       # Validation interval
                 save_dir,                  # Where to save the models
                 save_period,               # Save interval
                 tensorboard,               # Use Tensorboard to monitor the training ?
                 log_dir                    # The log directory for tensorboard logs
                 ):
        pass

    def start(self):
        pass