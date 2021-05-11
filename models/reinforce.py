import torch.nn as nn
import models.ann


class REINFORCE(nn.Module):
    """ Class for REINFORCE RL algorithm """

    def __init__(self,
                 input_dim,         # Number of input features
                 ann_params         # The parameters to the feed-forward network
                 ):
        super().__init__()

        # One difference here is that the number of actions is 2
        # Because we need to predict a probability for each action
        # One might think, it is enough to have one value, as the other can be found by
        # subtracting from 1. But, this is required for back-propagating the gradient
        # for both actions, instead of just single action.
        ann_params['output_dim'] = 2

        # Policy Network is same as a traditional ANN
        self.policy_net = models.ann.ANN(input_dim=input_dim, **ann_params)

    def forward(self, X):
        """ Input is of shape (batch, input_dim). Just do a forward pass and return """
        action_scores = self.policy_net(X)   # output shape: (batch, n_actions)
        return action_scores
