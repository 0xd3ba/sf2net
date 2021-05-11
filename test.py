import pathlib
import pickle
import torch
from tqdm import tqdm
from utils import preprocess
from utils import snr


class Tester:
    """ Class responsible for testing the performance of the model """

    def __init__(self,
                 test_dataset,      # The testing dataset
                 model_path,        # The path to the pretrained model
                 transform,         # The transformation function to apply on input data
                 threshold,         # The SNR threshold
                 output_dir         # The directory where to store the results
                 ):
        self.dataset = test_dataset
        self.transform = transform
        self.threshold = threshold
        self.output_dir = pathlib.Path(output_dir)
        self.model_path = pathlib.Path(model_path)
        self.model = None
        self.snr_func = snr.wada_snr
        self.prepare_data = preprocess.PreprocessAudio(window_len=transform.MelSpectrogram.n_fft,
                                                       stride_len=transform.MelSpectrogram.hop_length,
                                                       threshold=threshold,
                                                       transform_func=transform)

    def start(self):
        """
        Starts the testing process.
        Produces the post-processed enhanced files into provided output directory
        """
        self._load()     # Loads the model
        for (clean_t, _), (noisy_t, _), (enhanced_t, enhanced_file) in tqdm(self.dataset, desc='Samples Processed'):
            # Sometimes, what it happens is that the wav file turns out to be corrupt and could not be read
            # So in such cases, we need to skip it. The dataset class takes care of printing an error message
            if clean_t is None or noisy_t is None or enhanced_t is None:
                continue

            # We have no need for targets or absolute SNR differences as we are not training
            # noisy_t_transf, _, _ = self.prepare_data.preprocess(clean_tensor=clean_t, noisy_tensor=noisy_t)

    def _load(self):
        """ Load the model into memory """
        # Now try loading the model -- It might throw an exception
        # So need to consider this case as well
        try:
            with open(self.model_path, 'rb') as m:
                self.model = pickle.load(m)
        except FileNotFoundError:
            print(f'ERROR: Could not load model from "{self.model_path}"')
            exit(0)

        # NOTE: For now assume that it is indeed an object of our model wrappers
        #       Might need to get rid of this assumption later
