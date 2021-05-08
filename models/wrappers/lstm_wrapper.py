import torch
import models.base


class LSTM_Wrapper(models.base.BaseModel):
    """ Wrapper class for ANN model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)

    def train(self, batch_noisy, batch_clean):
        """ Trains the model after feeding in the batch """
        pass

    def evaluate(self, noisy_samples, clean_samples):
        """ Performs a validation on the given dataset """
        pass

    def log(self, log_dir):
        """ Saves the training logs if tensorboard monitoring is enabled """
        pass

    def predict(self, noisy_samples):
        """
        Predicts the frames of every audio sample as 0 or 1
            0: Frame doesn't require enhancement
            1: Frame requires enhancement
        """
        pass
