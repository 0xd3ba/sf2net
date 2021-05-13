import os
import pickle
import torch
from tqdm import tqdm
from utils import preprocess
from utils import snr


class Trainer:
    """ Trainer class responsible for training the model """
    def __init__(self,
                 train_dataset,             # An instance of torch's Dataset class
                 validation_dataset,        # An instance of torch's Dataset class
                 model,                     # The model to use for training
                 transform,                 # The transformation function to apply on inputs
                 threshold,                 # The threshold for setting up target labels
                 window_size,               # The size of the frame window
                 window_stride,             # The amount of strides
                 epochs,                    # Number of epochs to train the model
                 validation_interval,       # Validation interval
                 save_dir,                  # Where to save the models
                 save_period,               # Save interval
                 tensorboard,               # Use Tensorboard to monitor the training ?
                 log_dir                    # The log directory for tensorboard logs
                 ):

        self.model = model
        self.train_dataset = train_dataset
        self.transform = transform
        self.threshold = threshold
        self.validation_dataset = validation_dataset
        self.epochs = epochs
        self.validation_interval = validation_interval
        self.model_chkpt_dir = save_dir
        self.model_chkpt_interval = save_period
        self.use_tensorboard = tensorboard
        self.tensorboard_log_dir = log_dir

        self.prepare_data = preprocess.PreprocessAudio(window_len=window_size,
                                                       stride_len=window_stride,
                                                       threshold=threshold,
                                                       transform_func=transform,
                                                       device=self.model.device)

    def start(self):
        """ Performs the training on the model for given number of epochs """
        train_data = self.train_dataset

        for epoch in range(1, self.epochs+1):

            train_loss = 0

            for (clean_t, _), (noisy_t, _), _ in tqdm(train_data, desc=f'[{epoch}/{self.epochs}][TRAIN] Samples processed'):

                # Sometimes, what it happens is that the wav file turns out to be corrupt and could not be read
                # So in such cases, we need to skip it. The dataset class takes care of printing an error message
                if clean_t is None or noisy_t is None:
                    continue

                # We'll be training our model for every sample, not in a batch
                # because of varying number of frames in different files -- introduces complications
                # but this doesn't cause much of an issue because:
                #
                # train_x is of shape: (n_features, n_frames)    (PS: Need to transpose)
                # train_y is of shape: (n_frames)
                #
                # Depending on the model, train_x's n_frames can be treated as a batch (IID assumption)
                # or an episode (for RL models) or (very long) sequence for RNNs
                train_x, train_y, frame_diffs = self.prepare_data.preprocess(clean_tensor=clean_t, noisy_tensor=noisy_t)
                train_loss += self.model.train(train_x, train_y, frame_diffs)

            # We need to do a validation now
            if epoch % self.validation_interval == 0:
                self._validate()

            # Check if we need to log, if yes, then write the logs to tensorboard
            # TODO: Implement the methods to log the statistics
            if self.use_tensorboard:
                self.model.log(self.tensorboard_log_dir)

            # Check if are ready to checkpoint the model
            if epoch % self.model_chkpt_interval == 0:
                self._checkpoint_model(epoch)

            print(f'Training Loss: {train_loss}')
            print()

    def _validate(self):
        """ Performs a validation on the validation data """
        val_data = self.validation_dataset
        val_acc = 0
        n_samples = 0

        for (clean_t, _), (noisy_t, _), _ in tqdm(val_data, desc='[VALIDATION] Samples processed'):
            val_x, val_y, frame_diffs = self.prepare_data.preprocess(clean_tensor=clean_t, noisy_tensor=noisy_t)
            val_acc += self.model.evaluate(val_x, val_y, frame_diffs)
            n_samples += 1

        print()
        print(f'Validation Accuracy: {val_acc / n_samples}')
        print()

    def _checkpoint_model(self, epoch):
        """ Checkpoints the model into the save directory """
        file_name = f'model_epoch_{epoch}.pkl'
        file_path = os.path.join(self.model_chkpt_dir, file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
