import numpy as np
import torch


class PreprocessAudio:
    """ Preprocessing class for preparing the data for training """

    def __init__(self, window_len, stride_len, threshold, transform_func, snr_func):
        self.window_len = window_len
        self.stride_len = stride_len
        self.threshold = threshold
        self.transform_func = transform_func
        self.snr_func = snr_func

    def preprocess(self, clean_tensor, noisy_tensor):
        """ Splits the tensors into frames and computes the targets using SNR estimator """

        # Note that we might miss some values because of stride_len
        # But that will not matter much, as the maximum we can lose is (stride_len-1) samples
        clean_unfolded = self.chunk_it(clean_tensor)
        noisy_unfolded = self.chunk_it(noisy_tensor)

        clean_snrs = self.snr_func(clean_unfolded)
        noisy_snrs = self.snr_func(noisy_unfolded)
        snr_diff = torch.abs(clean_snrs - noisy_snrs)

        # Now prepare the targets
        # A frame needs enhancement if
        #       |SNR(clean_frame) - SNR(noise_frame)| > threshold
        targets = torch.where(snr_diff > self.threshold, 1.0, 0.0)

        # Now transform the inputs, i.e. the noisy tensors
        noisy_transformed = self.transform_func(noisy_unfolded).squeeze(-1)

        return noisy_transformed, targets, snr_diff

    def chunk_it(self, wav_tensor):
        """ Chunks the audio into frames of fixed length with fixed stride """
        return wav_tensor.unfold(0, self.window_len, self.stride_len)

    def inverse_chunk_it(self, chunked_wav_tensor):
        """ Undoes the chunking process to a single wav tensor """
        # TODO: Write the method later
        pass
