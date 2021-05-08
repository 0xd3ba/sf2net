import argparse
import torch
import torchaudio.transforms
from data_loader.ms_snsd import MS_SNSD
from utils import parse_config
from utils import factory
from train import Trainer

ARGS_MODE_TRAIN = 'train'
ARGS_MODE_TEST = 'test'
SNR_DIFF_THRESHOLD = 1.0

def validate_args(args):
    """ Validates the user supplied arguments """
    assert (args.mode == ARGS_MODE_TRAIN or args.mode == ARGS_MODE_TEST), f'Invalid mode: "{args.mode}"'
    # Checks for entered model was done in the config parsing stage -- Can assume it is correct


def start(config, model, start_mode, device):
    """
    By now, the arguments and the configuration have been parsed
    Ready to start the training/testing process
    """
    dataset_params = config.get_dataset_params()
    transform = getattr(torchaudio.transforms, config.get_data_transform_type())
    transform = transform(**config.get_data_transform_args())

    # *******************************************************************************
    # Input features to the model is the following (from torchaudio's documentation)
    # This may need changes depending on the transformation being applied
    # Basically, the output of the transform is of shape (..., features_dim, time)
    # We need the features_dim to be our input shape.
    # Could not find a nice and a clean way to generalize this, so need to manually
    # change depending on the transform
    ip_dim = transform.n_mels
    # *******************************************************************************

    # Prepare the dataset accordingly
    # NOTE: Because of our approach, PyTorch's data-loaders introduces complications. So not using it
    if start_mode == ARGS_MODE_TEST:
        test_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_test)

    # ================================ TRAINING MODE ================================
    else:
        train_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_train)
        val_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_val)

        model = factory.build(ip_dim, config, device)
        trainer = Trainer(train_dataset=train_dataset,
                          validation_dataset=val_dataset,
                          model=model,
                          transform=transform,
                          threshold=SNR_DIFF_THRESHOLD,
                          **config.get_trainer_params())
        trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='The configuration JSON file to use (default is "config.json")')
    parser.add_argument('--model', type=str, required=True,
                        help=f'The model to use (see config.json)')
    parser.add_argument('--mode', default=ARGS_MODE_TRAIN, type=str,
                        help=f'Train/Test the model: [{ARGS_MODE_TRAIN} | {ARGS_MODE_TEST}]')

    args = parser.parse_args()
    validate_args(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = parse_config.ConfigParser(args.config, args.model)
    start(config, args.model, args.mode, device)
