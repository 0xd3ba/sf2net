# BaseModel defines a common interface for the models to implement.
# Each PyTorch's model needs to be wrapped with its own implementation
# of BaseModel and override its methods. This was needed due to heterogeneity of
# various model types and their own training procedures

import torch

class BaseModel:
    """ Common interface for the models to implement """

    def __init__(self, model, optimizer, lr_scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.model.to(self.device)

    def train(self, data, target, snr_diff=None):
        """ Trains the model after feeding in the batch
        """
        raise NotImplementedError

    def evaluate(self, data, target, snr_diff=None):
        """ Performs a validation on the given dataset
        """
        target = target.to(self.device)
        labels = self.predict(data)
        n_correct = torch.where(labels == target, 1.0, 0.0)
        return n_correct.mean().cpu().item()

    def log(self, log_dir):
        """ Saves the training logs if tensorboard monitoring is enabled """
        raise NotImplementedError

    def predict(self, data):
        """
        Predicts the frames of every audio sample as 0 or 1
            0: Frame doesn't require enhancement
            1: Frame requires enhancement
        """
        raise NotImplementedError
