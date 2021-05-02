from utils import json_utils


class ConfigParser:
    """
    Class to parse the configuration JSON file
    """
    def __init__(self):
        pass

    @classmethod
    def parse(cls, config_path, model):
        """
        Parses the configuration JSON file and returns an instantiated
        ConfigParser object
        """
        json_dict = json_utils.read_json(config_path)

        # TODO: Parse the contents appropriately