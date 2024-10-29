import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from train_utils import Logger
from model import KT_Model
from sklearn import metrics
from evaluation import question2auc
import os


def train(loader, args):

    logger = Logger(args)
    device = torch.device(args.device)
    model = KT_Model(args, device, loader).to(device)

    # model.load_state_dict(torch.load("../param/embedding_ASSIST09_128_0.790.pkl"), strict=False)
    # for para in model.QuesEmb_Layer.parameters():
    #     para.requires_grad = False

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_weight)
    criterion = nn.BCELoss(reduction='mean')

    for epoch in range(1, args.max_epoch + 1):
        logger.epoch_increase()
        # num_batch = len(loader['train'])
        # 5个tensor，分别对应一个batch学生的各种信息
        for i, (seq_lens, pad_data, pad_answer, pad_index, pack_label) in enumerate(loader['train']):
            # 得到预测值
            pack_predict = model(seq_lens, pad_data, pad_answer, pad_index, args)  # 运行模型
            # 由于pytorch版本原因，修改length为cpu形式
            # 对填充过的预测，拿掉填充并组合在一起
            # print(pad_predict.shape)
            loss = criterion(pack_predict.data, pack_label.data)
            # print(i, num_batch, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 分别算测试集和训练集的auc
        train_metrics_dict = evaluate(model, loader['train'], args)
        test_metrics_dict = evaluate(model, loader['test'], args)

        logger.one_epoch(epoch, train_metrics_dict, test_metrics_dict, model)

        if logger.is_stop():
            break
        # end of epoch
    logger.one_run(args)
    # end of run
    

def evaluate(model, data, args):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pack_label in data:
        pack_predict = model(seq_lens, pad_data, pad_answer, pad_index, args)  # 运行模型

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)

    all_pred = torch.cat(pred_list, 0)
    all_target = torch.cat(true_list, 0)
    auc = metrics.roc_auc_score(all_target, all_pred)

    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    acc = metrics.accuracy_score(all_target, all_pred)

    model.train()
    return {'auc': auc, 'acc': acc}


def test(loader, args):
    device = torch.device(args.device)
    # if args.ks_model == 'dkt':
    model = KT_Model(args, device).to(device)
    # elif args.model == 'model_cwx':
    #     model = GraphKT_cwx(args, device).to(device)
    # elif args.model == 'model_cw_pk':
    #     model = GraphKT_cw_pk(args, device).to(device)
    # else:
    #     model = None
    # 调用pkl模型
    param_path = '../param/params_%s_%d_%d.pkl' % (
        (args.data_set, args.emb_dim, args.hidden_dim))
    if os.path.isfile(param_path):
        model.load_state_dict(torch.load(param_path))
        print("load model done")

    train_or_test = 'test'
    # 核心 获取每个题目对应的auc
    question2auc(model, loader[train_or_test], args.data_set, train_or_test)

