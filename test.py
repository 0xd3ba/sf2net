import pathlib
import pickle
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

    def start(self):
        """
        Starts the testing process.
        Produces the post-processed enhanced files into provided output directory
        """
        self.load()     # Loads the model
        for (clean_t, _), (noisy_t, _), (enhanced_t, enhanced_file) in tqdm(self.dataset, desc='Samples Processed'):
            print(clean_t, noisy_t, enhanced_t)

    def load(self):
        """ Load the model into memory """
        # Now try loading the model -- It might throw an exception
        # So need to consider this case as well
        try:
            with open(self.model_path, 'rb') as m:
                self.model = pickle.load(m)
                print(self.model)
        except FileNotFoundError:
            print(f'ERROR: Could not load model from "{self.model_path}"')
            exit(0)

        # NOTE: For now assume that it is indeed an object of our model wrappers
        #       Might need to get rid of this assumption later
