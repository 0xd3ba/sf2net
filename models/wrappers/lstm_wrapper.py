import torch
import torch.nn.functional as F
import models.base


class LSTM_Wrapper(models.base.BaseModel):
    """ Wrapper class for LSTM model """

    def __init__(self, model, optimizer, lr_scheduler, device):
        super().__init__(model, optimizer, lr_scheduler, device)
        self.loss_fn = F.binary_cross_entropy

    def train(self, data, target, frame_diff=None):
        """
        Trains the model after feeding in the batch

        Input shape:  (n_frames, n_features)
        Target shape: (n_frames, )
        """

        self.model.train()

        # Split into batches for efficiency, without loosing much information
        # We can exploit the monotonic property of frames, i.e. they only depend
        # on only the preceding frame
        data_new = self._prepare_batch(data)

        # Now bring them to the GPU, if present
        target = target.to(self.device)
        data = data_new.to(self.device)

        output_probs = self.model(data)

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
        self.model.eval()
        with torch.no_grad():
            data = self._prepare_batch(data).to(self.device)
            output_probs = self.model(data)
            pred_labels = torch.where(output_probs >= 0.5, 1, 0)

        return pred_labels.cpu()

    def _prepare_batch(self, data):
        """
        Prepares the data by breaking into batches. Needed because frames are too much
        which slows down training tremendously
        """
        # Break the data into fixed length sequences of length T, such that
        # (T-1) frames will serve as the context to last frame (which is what we need)
        # NOTE: It means that first T-1 frames in the data will need to be padded
        seq_length = self.model.sequence_length
        data_new = data.unfold(0, seq_length, 1).permute([0, 2, 1])

        # Now build the padding required for the first few samples (T-1 of them)
        for i in reversed(range(seq_length - 1)):
            n_pads = seq_length - i - 1
            zeros = torch.zeros(n_pads, data.shape[-1]).to(self.device)
            seq_i = torch.vstack([zeros, data[:i + 1]]).to(self.device)

            # Need to add a dimension because data_new is of shape (batch, seq_len, n_features)
            # and seq_i is of shape (seq_len, n_features). Also need to add at the top because
            # these are the first samples
            data_new = torch.vstack([seq_i.unsqueeze(0), data_new]).to(self.device)

        return data_new
