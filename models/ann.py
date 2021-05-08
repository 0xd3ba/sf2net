import torch
import torch.nn as nn


class ANN(nn.Module):
    """ Class for feed-forward ANN """

    def __init__(self, ip_dim, op_dim, n_hidden, units_list):
        super().__init__()

        assert n_hidden == len(units_list), "Number of units doesn't match the number of hidden layers"

        self.ip_dim = ip_dim
        self.op_dim = op_dim
        self.activation = nn.ReLU   # Change this depending on needs

        layers_list = []
        curr_dim = ip_dim

        # Now prepare the layers
        for next_dim in units_list:
            layer_i = nn.Linear(curr_dim, next_dim)     # Simple feed-forward layer
            activ_i = self.activation()                 # Use the default activation function set earlier

            layers_list.append(layer_i)
            layers_list.append(activ_i)

            curr_dim = next_dim

        # Now we need to prepare the output layer
        # No need to set any activation for it, the caller will do it
        layer_i = nn.Linear(curr_dim, op_dim)
        layers_list.append(layer_i)

        # Now build the model and save it
        self.ann = nn.Sequential(*layers_list)

    def forward(self, X):
        """ Input is of shape (batch, ip_dim). Just do a forward pass and return """
        return self.ann(X)
