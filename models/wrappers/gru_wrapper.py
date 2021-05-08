import torch
import models.base


class GRU_Wrapper(models.base.BaseModel):
    """ Wrapper class for GRU model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)

    def train(self, data, target):
        """
        Trains the model after feeding in the batch

        Input shape:  (n_frames, n_features)
        Target shape: (n_frames, )
        """
        data = data.unsqueeze(0)                    # Insert batch dimension which is required
        output_probs = self.model(data).squeeze(0)  # Remove the batch dimension after getting the output

        loss = self.loss_fn(output_probs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

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
