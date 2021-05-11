import torch
import torch.nn.functional as F
import models.base


class ANN_Wrapper(models.base.BaseModel):
    """ Wrapper class for ANN model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)
        self.loss_fn = F.binary_cross_entropy

    def train(self, data, target, snr_diff=None):
        """
        Trains the model after feeding in the batch

        Input shape:  (n_frames, n_features)
        Target shape: (n_frames, )
        """
        data = data.to(self.device)
        target = target.to(self.device)

        output_scores = self.model(data).squeeze(-1)  # Assumes data is I.I.D
        output_probs = torch.sigmoid(output_scores)

        loss = self.loss_fn(output_probs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def log(self, log_dir):
        """ Saves the training logs if tensorboard monitoring is enabled """
        pass

    def predict(self, data):
        """
        Predicts the frames of every audio sample as 0 or 1
            0: Frame doesn't require enhancement
            1: Frame requires enhancement
        """
        data = data.to(self.device)
        with torch.no_grad():
            output_scores = self.model(data).squeeze(-1)  # Assumes data is I.I.D
            output_probs = torch.sigmoid(output_scores)
            pred_labels = torch.where(output_probs >= 0.5, 1, 0)

        return pred_labels.cpu()
