"""Core model extracted from archived research code; see project.json and CHANGELOG.md."""

import numpy as np
import torch
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class myLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, layers, n_layers_LSTM):

        # The caller owns the RNG seed; do not reset global state here.

        super(myLSTM, self).__init__()

        # add LSTM layers
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers_LSTM
        self.lstm = torch.nn.LSTM(
            input_size, hidden_dim, self.n_layers, batch_first=True
        )

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()

        # add full-connectted layers
        layer_list.append(("activation_%d" % 0, self.activation()))
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % (i + 1), self.activation()))

        layer_list.append(("layer_%d" % (self.depth - 1), torch.nn.Dropout(0.1)))

        layer_list.append(
            ("layer_%d" % (self.depth), torch.nn.Linear(layers[-2], layers[-1]))
        )

        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x, t, len_seq):

        packed_input = pack_padded_sequence(
            input=x, lengths=len_seq, batch_first=True, enforce_sorted=False
        )

        out_lstm, h_c_n = self.lstm(packed_input)

        out_lstm, _ = pad_packed_sequence(out_lstm, batch_first=True)

        t_adapt = t[:, : out_lstm.shape[1], :]

        out_lstm_t = torch.cat([out_lstm, t_adapt], dim=2)
        # out = self.layers(h_c_n[0][-1,:,:])
        out = self.layers(out_lstm_t)

        return out
