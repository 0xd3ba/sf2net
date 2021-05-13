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
                 window_size,       # The size of the window
                 window_stride,     # The window stride amount
                 output_dir         # The directory where to store the results
                 ):
        self.dataset = test_dataset
        self.transform = transform
        self.threshold = threshold
        self.window_size = window_size
        self.window_stride = window_stride
        self.output_dir = pathlib.Path(output_dir)
        self.model_path = pathlib.Path(model_path)
        self.model = None
        self.snr_func = snr.wada_snr
        self.prepare_data = None
        self.overall_correct = 0    # How many frames did we correctly classify
        self.overall_nframes = 0    # Out of how many frames in total

    def start(self):
        """
        Starts the testing process.
        Produces the post-processed enhanced files into provided output directory
        """

        self._load()     # Loads the model
        for (clean_t, _), (noisy_t, n_file), (enhanced_t, e_file) in tqdm(self.dataset, desc='Samples Processed'):
            # Sometimes, what it happens is that the wav file turns out to be corrupt and could not be read
            # So in such cases, we need to skip it. The dataset class takes care of printing an error message
            if clean_t is None or noisy_t is None or enhanced_t is None:
                continue

            # We have no need for targets or absolute frame-wise differences as we are not training
            noisy_t_transf, targets, frame_diffs = self.prepare_data.preprocess(clean_tensor=clean_t, noisy_tensor=noisy_t)
            pred_targets, pred_probs = self.model.predict(noisy_t_transf)

            self._print_prediction_stats(pred_targets, targets, n_file)
            mod_enhanced_t = self._replace_frames(enhanced_t, noisy_t, pred_targets, pred_probs)
            self._produce_wav(mod_enhanced_t, e_file)

        # Now print the overall accuracy
        print()
        overall_frac_corr = round(self.overall_correct / self.overall_nframes, 2)
        print(f'Overall stats:    {self.overall_correct}/{self.overall_nframes} ({overall_frac_corr})')
        print()

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

        self.prepare_data = preprocess.PreprocessAudio(window_len=self.window_size,
                                                       stride_len=self.window_stride,
                                                       threshold=self.threshold,
                                                       transform_func=self.transform,
                                                       device=self.model.device)

        # NOTE: For now assume that it is indeed an object of our model wrappers
        #       Might need to get rid of this assumption later

    def _print_prediction_stats(self, pred_targets, true_targets, file_name):
        """ Prints the prediction statistics for the given file and predictions """
        n_correct = (pred_targets == true_targets).sum().cpu().item()
        n_total = pred_targets.shape[0]
        frac_corr = round(n_correct / n_total, 2)

        # Update the overall count
        self.overall_correct += n_correct
        self.overall_nframes += n_total

        print(f'{file_name}:    {n_correct}/{n_total} ({frac_corr})')

    def _replace_frames(self, enhanced_t, noisy_t, pred_targets, pred_probs):
        """ Replaces the frames of enhanced tensor with noisy tensor """
        # TODO: Write the code to replace with smoothing applied
        pass

    def _produce_wav(self, modded_enhanced_t, enhanced_file_name):
        """ Produces the .wav file from the given tensor """
        # TODO: Write the code to generate the wav file from the given tensor
        pass