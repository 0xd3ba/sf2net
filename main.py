import argparse
from utils import parse_config

def start(config, mode):
    """
    By now, the arguments and the configuration have been parsed
    Ready to starts the training/testing process
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='The configuration JSON file to use (default is "config.json")')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use: [ann | lstm | gru | reinforce | ddqn]')
    parser.add_argument('--mode', default="train", type=str,
                        help='Train/Test the model: [train | test] (default is "train")')

    args = parser.parse_args()
    config = parse_config.ConfigParser.parse(args.config, args.model)
    start(config, args.mode)
