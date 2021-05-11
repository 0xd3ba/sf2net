import numpy as np
import torch


class PreprocessAudio:
    """ Preprocessing class for preparing the data for training """

    def __init__(self, window_len, stride_len, threshold, transform_func):
        self.window_len = window_len
        self.stride_len = stride_len
        self.threshold = threshold
        self.transform_func = transform_func

    def preprocess(self, clean_tensor, noisy_tensor):
        """ Splits the tensors into frames and computes the targets using SNR estimator """

        # Note that we might miss some values because of stride_len
        # But that will not matter much, as the maximum we can lose is (stride_len-1) samples
        clean_unfolded = self.chunk_it(clean_tensor)
        noisy_unfolded = self.chunk_it(noisy_tensor)

        # Now transform the tensors
        noisy_transformed = self.transform_func(noisy_unfolded).squeeze(-1)
        clean_transformed = self.transform_func(clean_unfolded).squeeze(-1)

        # Compute the distance between the transformed tensors
        transform_diff = (clean_transformed - noisy_transformed).norm(p=2, dim=-1)

        # The frames need enhancement if
        #   Distance(transformed_clean_frame, transformed_noise_frame) > threshold
        targets = torch.where(transform_diff < self.threshold, 0.0, 1.0)

        return noisy_transformed, targets, transform_diff

    def chunk_it(self, wav_tensor):
        """ Chunks the audio into frames of fixed length with fixed stride """
        return wav_tensor.unfold(0, self.window_len, self.stride_len)

    def inverse_chunk_it(self, chunked_wav_tensor):
        """ Undoes the chunking process to a single wav tensor """
        # TODO: Write the method later
        pass
