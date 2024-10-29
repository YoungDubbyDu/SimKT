
import os
print(os.getcwd())
import torch,sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from SimKT_code.train import train, test
from SimKT_code.loader import load_data
# from SimKT_code.loader_qa import load_data


# print(torch.__version__)
# 默认参数设置
def parse_args():
    parser = argparse.ArgumentParser()
    # for loading data
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--data_path", type=str, default="E:/study/KT/DataProcess2022")
    parser.add_argument("--data_set", type=str, default='ASSIST09')
    parser.add_argument("--num_ques", type=int, default=15680)
    parser.add_argument("--input", type=str, default='question')
    parser.add_argument("--root", type=str, default="E:/study/SimKT/Supplement-Code")


    # for embedding and input layer
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--embed_mode", type=str, default="pre")  # pre_emb, random
    # dkvmn
    parser.add_argument('--q_embed_dim', type=int, default=128, help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256, help='answer and question embedding dimensions')

    # for knowledge state layer
    parser.add_argument("--hidden_dim", type=int, default=128)
    # 1.rnn
    parser.add_argument("--rnn_mode", type=str, default='lstm')  # lstm rnn gru
    parser.add_argument("--rnn_num_layer", type=int, default=1)
    # 2. dkvmn
    parser.add_argument('--memory_size', type=int, default=100, help='memory size')
    parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')
    # 3. sakt
    parser.add_argument('--num_attn_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=512)
    parser.add_argument('--drop_prob', type=float, default=0.05)

    # for predict layer
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--predict_type", type=str, default="mlp")  # dot, mlp

    # for training
    parser.add_argument("--model_type", type=str, default='SimQE_DKT')  # dkt, dkvmn, sakt
    parser.add_argument("--ks_mode", type=str, default='dkt')  # dkt, dkvmn, sakt
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--l2_weight", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda:0")

    # for saving
    parser.add_argument('--save_dir', type=str, default='./result/ASSIST09', help='the dir which save results')
    parser.add_argument('--log_file', type=str, default='logs.txt', help='the name of logs file')
    parser.add_argument('--result_file', type=str, default='tunings.txt', help='the name of results file')
    parser.add_argument('--remark', type=str, default='', help='remark the experiment')

    # for SimKT
    parser.add_argument("--meta_paths", type=list,
                        default=['quq', 'qkq'])
    parser.add_argument("--top_k", type=int,
                        default=3)
    parser.add_argument("--fusion", type=str,
                        default='concat_nonLinear')
    # 'concat_nonLinear', 'attnVec_dot', 'attnVec_nonLinear', 'attnVec_topK'

    return parser.parse_args()


# ASSIST09: ['qkq_contDiff1.0', 'qtq_noWgt']
# ASSIST12: ['quq_cw', 'qkq_contDiff', 'qtq_noWgt', 'qkckq_contDiff_64']
# EdNet: ['quq_cw', 'qkq_contDiff', 'qucuq_cw_120', 'qkckq_contDiff_64']
# A17: ['quq_wgt1.0',"qkq_contDiff_pos",'qucuq_cw_120', 'qkckq_contDiff_60']
# All: ['quq_cw', 'qkq_contDiff', 'qucuq_cw_120', 'qkckq_contDiff_64', 'qtq_noWgt']
args = parse_args()
args.data_set = "ASSIST09"
args.save_dir = "../result/%s" % args.data_set
args.embed_mode = "pre_emb"
data_loader = load_data(args)


for args.meta_paths in [['qkq_contDiff1.0', 'qtq_noWgt']]:
    # for args.lr in [0.01, 0.0001]:
    for args.top_k in [2]:  # valid if fusionnix == "attnVec_topK"
        if args.ks_mode == "sakt":
            for args.l2_weight in [1e-4]:  # sakt选1e-4
                train(data_loader, args)
        elif args.ks_mode == "dkvmn":
            for i in range(1):
                for args.l2_weight in [1e-8]:  # dkvmn选1e-8
                    train(data_loader, args)
        else:
            for i in range(1):
                train(data_loader, args)
