import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAKT(nn.Module):
    def __init__(self, args, data):
        """
        num_skills (int): number of skills
        embed_dim (int): input embedding and attention dot-product dimension
        num_attn_layers (int): number of attention layers
        num_heads (int): number of parallel attention heads
        encode_pos (bool): if True, use relative position embeddings
        max_pos (int): number of position embeddings to use
        drop_prob (float): dropout probability
        """

        super(SAKT, self).__init__()
        self.embed_dim = args.emb_dim
        self.num_attn_layers = args.num_attn_layer
        self.num_heads = args.num_heads
        self.encode_pos = args.encode_pos
        self.max_pos = args.max_pos
        self.drop_prob = args.drop_prob
        num_skills = args.num_ques

        # self.item_embeds = nn.Embedding(num_items + 1, self.embed_dim // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, self.embed_dim, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(self.max_pos, self.embed_dim // self.num_heads)
        self.pos_value_embeds = nn.Embedding(self.max_pos, self.embed_dim // self.num_heads)

        self.pos_emb = CosinePositionalEmbedding(self.embed_dim, self.max_pos)

        self.lin_in = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.attn_layers = clone(MultiHeadedAttention(self.embed_dim, self.num_heads, self.drop_prob), self.num_attn_layers)

        self.dropout = nn.Dropout(p=self.drop_prob)
        self.lin_out = nn.Linear(self.embed_dim, 1)

        self.attn_weight = None

    def get_inputs(self, ques_emb, answer):
        answer = answer.unsqueeze(-1).float()
        inputs = torch.cat([ques_emb, ques_emb], dim=-1)
        inputs[..., :self.embed_dim] *= answer
        inputs[..., self.embed_dim:] *= 1 - answer
        return inputs

    def get_query(self, ques_emb):
        query = torch.cat([ques_emb], dim=-1)
        return query

    def forward(self, curr_ques_emb, interact_emb, next_ques_emb):
        # inputs = self.get_inputs(curr_ques_emb, answer)
        inputs = F.relu(self.lin_in(interact_emb))

        query = self.get_query(next_ques_emb)
        t = self.pos_emb(inputs)
        inputs = inputs + t

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        outputs = self.attn_layers[0](query, inputs, inputs, self.encode_pos,
                                      self.pos_key_embeds, self.pos_value_embeds, mask)

        for l in self.attn_layers[1:]:
            residual = l(query, outputs, outputs, self.encode_pos, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        # return self.lin_out(outputs)
        return outputs  # 输出知识状态向量


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

        self.layer_norm = nn.LayerNorm(total_size)
        self.ffn = feedforward(total_size, drop_prob)
        self.use_ffn = True
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.prob_attn = None

    def forward(self, query, key, value, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]
        input = query
        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        out = self.layer_norm(self.dropout1(out) + input)
        # out = self.dropout1(out) + input
        if self.use_ffn:
            out = self.ffn(out)

        return out


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size()[-1])
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # (1, seq, Feature)


class feedforward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, inp):
        out = self.dropout1(self.activation(self.linear1(inp)))
        out = self.dropout2(self.linear2(out)) + inp
        return self.layer_norm(out)
        # return out


def future_mask(seq_length):
    mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])
