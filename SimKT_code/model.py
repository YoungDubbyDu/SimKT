from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from SimKT_code.ques_emb.PreEmb import PreEmb
from SimKT_code.stu_state.dkt import DKT
from SimKT_code.stu_state.dkvmn import DKVMN_MODEL
from SimKT_code.stu_state.sakt import SAKT
from SimKT_code.predict.Predict import Predict


class KT_Model(nn.Module, ABC):
    def __init__(self, args, device, data):
        super(KT_Model, self).__init__()
        assert args.ks_mode in ['dkt', 'dkvmn', 'sakt']
        if args.ks_mode == 'dkt':
            assert args.embed_mode in ['pre_emb', 'random']
            # 题目嵌入可以预训练好的 可以随机初始化
            if args.embed_mode == 'pre_emb':
                self.QuesEmb_Layer = PreEmb(args, device)
            else:
                print("using random")
                self.QuesEmb_Layer = nn.Embedding(args.num_ques, args.emb_dim)
            # 融合题目的embedding和答题情况的0/1 交互情况
            self.Fusion_Layer = Fusion_Module(args.emb_dim, device)
            # 知识状态模型
            print('using dkt')
            self.KS_Layer = DKT(args.rnn_mode, args.emb_dim * 2, args.hidden_dim, args.rnn_num_layer)
            # 做预测
            self.Predict_Layer = Predict(args.hidden_dim, args.emb_dim, args.exercise_dim, args.predict_type)
        elif args.ks_mode == 'dkvmn':
            assert args.embed_mode in ['pre_emb', 'random']
            # 题目嵌入可以预训练好的 可以随机初始化
            if args.embed_mode == 'pre_emb':
                self.QuesEmb_Layer = PreEmb(args, device)
            else:
                print("using random")
                self.QuesEmb_Layer = nn.Embedding(args.num_ques, args.q_embed_dim)
            # 融合题目的embedding和答题情况的0/1 交互情况
            self.Fusion_Layer = Fusion_Module(args.emb_dim, device)
            print("using dkvmn")
            self.KS_Layer = DKVMN_MODEL(args, data)
            # 做预测
            self.Predict_Layer = Predict(args.qa_embed_dim, args.q_embed_dim, args.exercise_dim, args.predict_type)

        elif args.ks_mode == 'sakt':
            assert args.embed_mode in ['pre_emb', 'random']
            # 题目嵌入可以预训练好的 可以随机初始化
            if args.embed_mode == 'pre_emb':
                self.QuesEmb_Layer = PreEmb(args, device)
            else:
                print("using random")
                self.QuesEmb_Layer = nn.Embedding(args.num_ques, args.emb_dim)
            # 融合题目的embedding和答题情况的0 / 1交互情况
            self.Fusion_Layer = Fusion_Module(args.emb_dim, device)
            print("using sakt")
            self.KS_Layer = SAKT(args, data)
            # 做预测
            self.Predict_Layer = Predict(args.emb_dim, args.emb_dim, args.exercise_dim, args.predict_type)

        self.ques_emb = None
        self.ks_emb = None

    def forward(self, seq_lens, pad_ques, pad_answer, pad_next, args):
        # 输入题号 获得每个学生每个时间步的嵌入

        if args.ks_mode == 'dkt':
            self.ques_emb = self.QuesEmb_Layer(pad_ques)
            # 下一个要预测对的题目嵌入
            next_emb = self.QuesEmb_Layer(pad_next)
            # 将题目嵌入和答题情况融合得到输入向量
            input_emb = self.Fusion_Layer(self.ques_emb, pad_answer)
            # 得到知识状态向量
            # print(self.ques_emb.shape, next_emb.shape, input_emb.shape)
            self.ks_emb = self.KS_Layer(input_emb)
            # 得到预测结果

            pad_predict = self.Predict_Layer(self.ks_emb, next_emb)
        # self.ques_emb = self.QuesEmb_Layer(pad_ques)
        # # 下一个要预测对的题目嵌入
        # next_emb = self.QuesEmb_Layer(pad_next)
        # # 将题目嵌入和答题情况融合得到输入向量
        # input_emb = self.Fusion_Layer(self.ques_emb, pad_answer)
        # 得到知识状态向量
        # if args.ks_mode == 'dkt':
        #     self.ks_emb = self.KS_Layer(input_emb)
        elif args.ks_mode == 'dkvmn':
            pad_ques = pad_ques.transpose(0, 1)
            pad_answer = pad_answer.transpose(0, 1)
            pad_next = pad_next.transpose(0, 1)

            self.ques_emb = self.QuesEmb_Layer(pad_ques)
            input_emb = self.Fusion_Layer(self.ques_emb, pad_answer)
            self.ks_emb = self.KS_Layer(pad_ques, self.ques_emb, input_emb)
            # 得到预测结果
            pad_predict = self.Predict_Layer(self.ks_emb, self.ques_emb)

        elif args.ks_mode == 'sakt':
            pad_ques = pad_ques.transpose(0, 1)
            pad_answer = pad_answer.transpose(0, 1)
            pad_next = pad_next.transpose(0, 1)
            self.ques_emb = self.QuesEmb_Layer(pad_ques)

            next_emb = self.QuesEmb_Layer(pad_next)
            input_emb = self.Fusion_Layer(self.ques_emb, pad_answer)
            self.ks_emb = self.KS_Layer(self.ques_emb, input_emb,next_emb)
            # print(self.ques_emb.shape,next_emb.shape,input_emb.shape)
            # 得到预测结果
            # print(self.ks_emb.shape,next_emb.shape) # 都反了
            pad_predict = self.Predict_Layer(self.ks_emb, next_emb)
            # print(pad_predict.shape)
        else:
            pad_predict = None

        if args.ks_mode == 'dkt':
            pack_predict = pack_padded_sequence(pad_predict, seq_lens.cpu(), batch_first=False, enforce_sorted=True)
        else:
            pack_predict = pack_padded_sequence(pad_predict, seq_lens.cpu(), batch_first=True, enforce_sorted=True)
        return pack_predict


# 融合交互
class Fusion_Module(nn.Module, ABC):
    def __init__(self, emb_dim, device):
        super(Fusion_Module, self).__init__()
        self.transform_matrix = torch.zeros(2, emb_dim * 2, device=device)
        self.transform_matrix[0][emb_dim:] = 1.0
        self.transform_matrix[1][:emb_dim] = 1.0

    def forward(self, ques_emb, pad_answer):
        ques_emb = torch.cat((ques_emb, ques_emb), -1)
        answer_emb = F.embedding(pad_answer, self.transform_matrix)
        # print(ques_emb.shape,answer_emb.shape)
        input_emb = ques_emb * answer_emb
        return input_emb

