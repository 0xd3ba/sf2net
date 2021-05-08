# BaseModel defines a common interface for the models to implement.
# Each PyTorch's model needs to be wrapped with its own implementation
# of BaseModel and override its methods. This was needed due to heterogeneity of
# various model types and their own training procedures


class BaseModel:
    """ Common interface for the models to implement """

    def __init__(self, model, optimizer, lr_scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train(self, batch_noisy, batch_clean):
        """ Trains the model after feeding in the batch """
        raise NotImplementedError

    def evaluate(self, noisy_samples, clean_samples):
        """ Performs a validation on the given dataset """
        raise NotImplementedError

    def log(self, log_dir):
        """ Saves the training logs if tensorboard monitoring is enabled """
        raise NotImplementedError

    def predict(self, noisy_samples):
        """
        Predicts the frames of every audio sample as 0 or 1
            0: Frame doesn't require enhancement
            1: Frame requires enhancement
        """
        raise NotImplementedError
