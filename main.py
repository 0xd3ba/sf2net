import argparse
import torch
import torchaudio.transforms
from data_loader.ms_snsd import MS_SNSD
from utils import parse_config

ARGS_MODE_TRAIN = 'train'
ARGS_MODE_TEST  = 'test'
ARGS_MODEL_LIST = ['ann', 'lstm', 'gru', 'reinforce', 'dqn']

def validate_args(args):
    """ Validates the user supplied arguments """
    assert args.model in ARGS_MODEL_LIST, f'Invalid model: "{args.model}"'
    assert (args.mode == ARGS_MODE_TRAIN or args.mode == ARGS_MODE_TEST), f'Invalid mode: "{args.mode}"'


def start(config, mode, device):
    """
    By now, the arguments and the configuration have been parsed
    Ready to start the training/testing process
    """
    dataset_params = config.get_dataset_params()
    transform = getattr(torchaudio.transforms, config.get_data_transform_type())
    transform = transform(**config.get_data_transform_args())

    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Prepare the dataset accordingly
    # NOTE: Because of our approach, PyTorch's dataloaders introduces complications. So not using it
    if mode == ARGS_MODE_TEST:
        test_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_test)
    else:
        train_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_train)
        val_dataset = MS_SNSD(**dataset_params, dataset_type=MS_SNSD.dataset_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='The configuration JSON file to use (default is "config.json")')
    parser.add_argument('--model', type=str, required=True,
                        help=f'The model to use: {ARGS_MODEL_LIST}')
    parser.add_argument('--mode', default=ARGS_MODE_TRAIN, type=str,
                        help=f'Train/Test the model: [{ARGS_MODE_TRAIN} | {ARGS_MODE_TEST}]')

    args = parser.parse_args()
    validate_args(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = parse_config.ConfigParser(args.config, args.model)
    start(config, args.mode, device)
