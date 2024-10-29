from abc import ABC
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import dgl
from dgl.nn.pytorch import GATConv
import numpy as np


# ----------------- for multiple meta-path fusion ------------------------
class attnVec_dot(nn.Module, ABC):  # train attention vector + dot
    def __init__(self, args, num_mataPath, device):
        super(attnVec_dot, self).__init__()
        self.num_path = num_mataPath
        self.path_weight = None
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        # [num_ques, num_metaPath, 1]
        self.path_weight = F.softmax(torch.matmul(semantic_embeddings, self.attnVec), dim=1)
        # [num_ques, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * self.path_weight, dim=1, keepdim=False)
        return ques_embedding


class attnVec_dot_fc(nn.Module, ABC):  # train attention vector + dot
    def __init__(self, args, num_mataPath, device):
        super(attnVec_dot_fc, self).__init__()
        self.num_path = num_mataPath
        self.path_weight = None
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        # [num_ques, num_metaPath, 1]
        self.path_weight = F.softmax(torch.matmul(semantic_embeddings, self.attnVec), dim=1)
        # [num_ques, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * self.path_weight, dim=1, keepdim=False)
        ques_embedding = torch.tanh(self.fc(ques_embedding))
        return ques_embedding


class attnVec_nonLinear(nn.Module, ABC):  # train attention vector + nonLinear
    #
    def __init__(self, args, num_metaPath, device):
        super(attnVec_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.path_weight = None
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        #  可以不用激活函数
        # ques_embeddings = torch.tanh(self.fc(ques_embeddings))  # 这么写会覆盖原本的题目嵌入

        # [num_ques, num_metaPath, 1]
        trans_embeddings = torch.tanh(self.fc(semantic_embeddings))
        self.path_weight = F.softmax(torch.matmul(trans_embeddings, self.attnVec), dim=1)

        # self.path_weight = torch.matmul(ques_embeddings, self.attnVec)
        # self.path_weight = F.softmax(torch.sigmoid(self.path_weight), dim=2)  # 不用sigmoid

        # [seq_len, batch_size, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * self.path_weight, dim=1, keepdim=False)
        return ques_embedding


class attnMat_nonLinear(nn.Module, ABC):  # train attention vector + nonLinear
    def __init__(self, args, num_metaPath, device):
        super(attnMat_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.attnLinear_list = [nn.Linear(args.emb_dim, args.emb_dim, bias=True).to(device) for i in range(self.num_path)]
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, semantic_embeddings):
        transEmb_list = [F.relu(self.attnLinear_list[i](semantic_embeddings[i])) for i in range(self.num_path)]
        stack_embedding = torch.stack(transEmb_list, dim=1)  # (N, M, D)

        path_weight = F.softmax(torch.matmul(stack_embedding, self.attnVec), dim=1)
        # [num_ques, emb_dim]
        ques_embedding = torch.sum(stack_embedding * path_weight, dim=1, keepdim=False)
        ques_embedding = F.relu(ques_embedding)

        # fused_embedding = torch.sum(ques_embedding, dim=1, keepdim=False)  # (N, D)
        # fused_embedding = F.relu(fused_embedding)

        return ques_embedding


class attnVec_nonLinear_fc(nn.Module, ABC):  # train attention vector + nonLinear
    def __init__(self, args, num_metaPath, device):
        super(attnVec_nonLinear_fc, self).__init__()
        self.num_path = num_metaPath
        self.path_weight = None
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim)
        self.fc2 = nn.Linear(args.emb_dim, args.emb_dim)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        #  可以不用激活函数
        # ques_embeddings = torch.tanh(self.fc(ques_embeddings))  # 这么写会覆盖原本的题目嵌入

        # [num_ques, num_metaPath, 1]
        trans_embeddings = torch.tanh(self.fc1(semantic_embeddings))
        self.path_weight = F.softmax(torch.matmul(trans_embeddings, self.attnVec), dim=1)

        # self.path_weight = torch.matmul(ques_embeddings, self.attnVec)
        # self.path_weight = F.softmax(torch.sigmoid(self.path_weight), dim=2)  # 不用sigmoid

        # [seq_len, batch_size, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * self.path_weight, dim=1, keepdim=False)
        ques_embedding = F.relu(self.fc2(ques_embedding))
        return ques_embedding


class attnVec_topK(nn.Module, ABC):
    def __init__(self, args, num_metaPath, device):
        super(attnVec_topK, self).__init__()
        self.num_path = num_metaPath
        self.path_weight = None
        self.top_k = args.top_k
        self.emb_dim = args.emb_dim
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        # [num_ques, num_metaPath, 1]
        self.path_weight = F.softmax(torch.matmul(torch.tanh(self.fc(semantic_embeddings)), self.attnVec), dim=1)

        # topK selection
        select_weight = torch.topk(self.path_weight, k=self.top_k, dim=1, sorted=False)
        self.path_weight = select_weight.values
        index = select_weight.indices.repeat(1, 1, self.emb_dim)
        ques_embeddings = torch.gather(semantic_embeddings, dim=1, index=index)

        # [num_ques, emb_dim]
        ques_embedding = torch.sum(ques_embeddings * self.path_weight, dim=1, keepdim=False)
        return ques_embedding


class concat_nonLinear(nn.Module, ABC):  # concat + nonLinear
    def __init__(self, args, num_metaPath):
        super(concat_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.fc = nn.Linear(args.emb_dim * num_metaPath, args.emb_dim)
        # print(111)
        # self.fc.weight.requires_grad = False

    def forward(self, ques_embeddings):
        ques_embeddings = torch.cat(ques_embeddings, dim=-1)
        ques_embedding = F.relu(self.fc(ques_embeddings))
        return ques_embedding


class sa_concat_nonLinear(nn.Module, ABC):  # concat + nonLinear
    def __init__(self, args, num_metaPath):
        super(sa_concat_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.query_lin = nn.Linear(args.emb_dim, args.emb_dim, bias=False)
        self.key_lin = nn.Linear(args.emb_dim, args.emb_dim, bias=False)
        self.value_lin = nn.Linear(args.emb_dim, args.emb_dim, bias=False)
        self.fc = nn.Linear(args.emb_dim * num_metaPath, args.emb_dim)

    def forward(self, ques_embeddings):  # list of [B, D]
        start, step = 0, 32
        num_ques = ques_embeddings[0].size()[0]
        ques_embedding_list = []
        while start < num_ques:
            temp_ques_embeddings = [e[start: start+step, :] for e in ques_embeddings]
            ques_embedding = torch.stack(temp_ques_embeddings, dim=1)  # [B, L, D]
            print(ques_embedding.size())
            query, key, value = self.query_lin(ques_embedding), self.key_lin(ques_embedding), self.value_lin(ques_embedding)
            ques_embedding = self.self_attention(query, key, value)
            tuple_ques_emb = torch.split(ques_embedding, split_size_or_sections=1, dim=1)
            ques_embedding = torch.cat(tuple_ques_emb, dim=-1)
            ques_embedding = F.relu(self.fc(ques_embedding))
            ques_embedding_list.append(ques_embedding)
            start += step
        print("get embedding")
        return torch.cat(ques_embedding_list, dim=0).transpose(0, 1)

    @staticmethod
    def self_attention(query, key, value, dropout=None):
        # Compute scaled dot product attention
        # query: [B, L, D]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(query.size(-1))
        prob_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            prob_attn = dropout(prob_attn)
        return torch.matmul(prob_attn, value)
