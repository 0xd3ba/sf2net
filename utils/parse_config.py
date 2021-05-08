from utils import json_utils


class ConfigParser:
    """
    Class to parse the configuration JSON file
    """
    def __init__(self, config_path, model):

        json_dict = json_utils.read_json(config_path)

        self.dataset_params = json_dict['dataset']              # Save the information about the data
        self.data_transform = json_dict['transform']            # Save the information on audio transformation to apply
        self.supported_models = json_dict['models'].keys()      # List of supported models

        assert model in self.supported_models, f"Model: {model} is not supported"

        self.model = json_dict['models'][model]['class']        # Save the class name of the model
        self.model_params = json_dict['models'][model]['args']  # Save the model's parameters
        self.ann_params = json_dict['models']['ann']['args']    # Save the feed-forward network parameters
        self.optimizer_params = json_dict['optimizer']          # Save the optimizer information
        self.lr_scheduler_params = json_dict['lr_scheduler']    # Save the learning rate scheduler parameters
        self.trainer_params = json_dict['trainer']              # Save the information about the trainer

    def get_dataset_params(self):
        return self.dataset_params

    def get_data_transform_type(self):
        return self.data_transform['class']

    def get_data_transform_args(self):
        return self.data_transform['args']

    def get_model_key(self):
        return self.model

    def get_model_params(self):
        return self.model_params

    def get_ann_params(self):
        return self.ann_params

    def get_optimizer_params(self):
        return self.optimizer_params

    def get_lr_scheduler_params(self):
        return self.lr_scheduler_params

    def get_trainer_params(self):
        return self.trainer_params
