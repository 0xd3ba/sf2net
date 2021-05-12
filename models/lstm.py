import torch
import torch.nn as nn
import models.ann


class LSTM(nn.Module):
    """ Class for (Bi)LSTM RNN """

    def __init__(self,
                 input_dim,         # Number of input features
                 sequence_length,   # The length of the input sequence
                 n_recurrent,       # Number of recurrent layers
                 hidden_size,       # Number of units in hidden state
                 dropout,           # Dropout probability
                 bidirectional,     # Use bidirectional model ?
                 ann_params         # The parameters to the feed-forward network
                 ):
        super().__init__()
        self.n_recurrent = n_recurrent
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=n_recurrent,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

        # Now build the ANN by forwarding the parameters
        ann_ip_dims = hidden_size if not bidirectional else hidden_size*2
        self.ann = models.ann.ANN(input_dim=ann_ip_dims, **ann_params)

    def forward(self, X):
        # input shape: (batch, seq_len, n_features)
        lstm_output, _ = self.lstm(X)            # Shape: (batch, seq_len, hidden_size(*2 if bidirectional))

        # All batches, all features, but only of the last time-step
        # Shape: (batch, hidden_dims)
        ann_input = lstm_output[:, -1, :]             # Shape: (batch, hidden_dims)
        ann_output = self.ann(ann_input).squeeze(-1)  # Shape: (batch, )
        frame_probs = torch.sigmoid(ann_output)       # Shape: (batch, )

        # The outputs indicate the probability of each frame requiring enhancement
        # for each audio file in the batch
        return frame_probs
