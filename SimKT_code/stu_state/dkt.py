from abc import ABC

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class DKT(nn.Module, ABC):
    def __init__(self, rnn_type, input_dim, hidden_dim,num_layer):
        super(DKT, self).__init__()
        assert rnn_type in ['rnn', 'lstm', 'gru']
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=False)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=False)

    def forward(self, input_emb):
        ks_emb, _ = self.rnn(input_emb)  # [seq_len, batch_size, hidden_dim]
        return ks_emb
