
import torch
import torch.nn as nn
import math
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
# 定义TTT配置
# 定义TTT配置
# 定义TTT配置
class TTTConfig(PretrainedConfig):
    def __init__(self, input_dim=256, hidden_size=128, num_hidden_layers=3, intermediate_size=256, num_attention_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads

# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

# 定义TTT模型
class TTTModel(nn.Module):
    def __init__(self, config):
        super(TTTModel, self).__init__()
        self.attention = SelfAttention(config.input_dim, config.num_attention_heads)
        self.layers = nn.ModuleList([nn.Linear(config.input_dim if i == 0 else config.hidden_size, config.hidden_size) for i in range(config.num_hidden_layers)])
        self.activation = nn.ReLU()

    def forward(self, input_emb):
        # 通过注意力层，输出维度为input_dim
        hidden_states = self.attention(input_emb)
        # 通过线性层，第一层输入维度为input_dim
        for i, layer in enumerate(self.layers):
            if i == 0:
                hidden_states = self.activation(layer(hidden_states))
            else:
                hidden_states = self.activation(layer(hidden_states))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

# 修改DKT模型以使用TTT
class DKT(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_layers, use_ttt=False):
        super(DKT, self).__init__()
        self.use_ttt = use_ttt
        if use_ttt:
            config = TTTConfig(input_dim=input_dim, hidden_size=hidden_dim, num_hidden_layers=num_layers)
            self.rnn = TTTModel(config)
        else:
            assert rnn_type in ['rnn', 'lstm', 'gru']
            if rnn_type == 'rnn':
                self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)
            elif rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)
            else:
                self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

    def forward(self, input_emb):
        if self.use_ttt:
            ks_emb = self.rnn(input_emb).last_hidden_state
        else:
            ks_emb, _ = self.rnn(input_emb)
        return ks_emb

# 创建并初始化模型
# input_dim = 100
# hidden_dim = 50
# num_layers = 2
# use_ttt = False  # 设置为True以使用TTT-Linear
# model = DKT(rnn_type='rnn', input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, use_ttt=use_ttt)
#
# # 创建一些示例输入
# input_emb = torch.randn(10, 32, input_dim)  # [seq_len, batch_size, input_dim]
# output = model(input_emb)
# print(output.shape)
