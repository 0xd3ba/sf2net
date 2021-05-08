import os
import pathlib
from collections import namedtuple
from torch.utils.data import Dataset
import torchaudio


# A convenient named-tuple to pack items together
dataset_item = namedtuple("dataset_item", ['clean_file', 'noisy_file', 'enhanced_file'])


def _build_walker(clean_path, noisy_path, enhanced_path):
    """
    Prepares a list of form: [(clean_sample, noisy_sample, enhanced_sample), ...]
    This method might need changes depending on how the enhancer names the files.

    We're dealing with
        - Clean/Noisy samples produced by MS-SNSD script (duh)
        - Enhanced outputs produced by Facebook-Research's denoiser (https://github.com/facebookresearch/denoiser)
    """

    return _denoiser_walker(clean_path, noisy_path, enhanced_path)


def _denoiser_walker(clean_path, noisy_path, enhanced_path):
    """
    File name format of
        - Clean sample:     clnsp<id>.wav
        - Noisy sample:     noisy<id>_SNRdb_<snr_level>_clnsp<id>.wav
        - Enhanced sample:  noisy<id>_SNRdb_<snr_level>_clnsp<id>_enhanced.wav

    There is a one-to-one mapping between a noisy sample and an enhanced sample (should be obvious why)
    But there can be an one-to-many mapping between a clean sample and noisy samples (various degrees of SNR levels for
    the same clean sample and/or with various noises)
    """

    walker = []

    clean_files = sorted(clean_path.glob('*.wav'))
    for cf in clean_files:
        cf_name = (str(cf).split(os.path.sep)[-1]).split('.')[0]          # File name without .wav extension
        noisy_files = noisy_path.glob(f'*_{cf_name}.wav')                 # All noisy files for this clean file
        enhanced_files = enhanced_path.glob(f'*_{cf_name}_enhanced.wav')  # All enhanced files for this clean file

        # We can be sure that the ordering is maintained if we sort the files
        noisy_files = sorted(noisy_files)
        enhanced_files = sorted(enhanced_files)

        for nf, ef in zip(noisy_files, enhanced_files):
            entry = dataset_item(clean_file=cf,
                                 noisy_file=nf,
                                 enhanced_file=ef)
            walker.append(entry)

    return walker


class MS_SNSD(Dataset):
    """ Data-loader class for MS-SNSD dataset """

    dataset_train = 'mssnd_train'
    dataset_val = 'mssnd_validation'
    dataset_test = 'mssnd_testing'

    def __init__(self,
                 root_dir,      # The root directory storing the dataset
                 train_dir,     # The directory name of the training data (root_dir/train_dir)
                 val_dir,       # The directory name of the validation data (root_dir/val_dir)
                 test_dir,      # The directory name of the testing data (root_dir/testing_dir)
                 clean_dir,     # The directory name of the clean audio samples (root_dir/*/clean_dir)
                 noisy_dir,     # The directory name of the noisy audio samples (root_dir/*/noisy_dir)
                 enhanced_dir,  # The directory name of the enhanced audio samples (root_dir/*/enhanced_dir)
                 dataset_type,  # Data-loader for what ? training/validation/testing
                 ):

        assert (dataset_type == self.dataset_train or
                dataset_type == self.dataset_val or
                dataset_type == self.dataset_test), f'Dataset type "{dataset_type}" is not valid'

        self.loader_type = dataset_type
        self._root_path = pathlib.Path(root_dir)

        if dataset_type == self.dataset_train:
            self._dataset_dir = self._root_path / train_dir
        elif dataset_type == self.dataset_val:
            self._dataset_dir = self._root_path / val_dir
        else:
            self._dataset_dir = self._root_path / test_dir

        # Now set the paths for the clean, noisy and enhanced files
        self._clean_path = self._dataset_dir / clean_dir
        self._noisy_path = self._dataset_dir / noisy_dir
        self.enhanced_path = self._dataset_dir / enhanced_dir

        # Build the mapping between the audio files
        self._walker = _build_walker(self._clean_path, self._noisy_path, self.enhanced_path)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, index):
        """
        Returns the samples at the specified index

        During training and validation, we have no need for enhanced file. Don't load it (Saves some computation)
        During testing, we need it however
        """
        sample = self._walker[index]
        clean_wav, _ = torchaudio.load(sample.clean_file)
        noisy_wav, _ = torchaudio.load(sample.noisy_file)
        enhanced_wav = None

        if self.loader_type == self.dataset_test:
            enhanced_wav, _ = torchaudio.load(sample.enhanced_file)
            enhanced_wav = enhanced_wav[0]

        # The tensors are of shape (channel_num, samples)
        # We are dealing with single channel wav files, so the first dimension is of no use
        clean_wav = clean_wav[0]
        noisy_wav = noisy_wav[0]

        return clean_wav, noisy_wav, enhanced_wav

