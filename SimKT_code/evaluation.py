from sklearn import metrics
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import os
PATH = "C://Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess/data_GIKT"


def evaluate(model, data):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pack_label in data:
        pack_pred = model(seq_lens, pad_data, pad_answer, pad_index)

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_pred.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)
    auc = metrics.roc_auc_score(np.concatenate(true_list, 0), np.concatenate(pred_list, 0))
    model.train()
    return {'auc': auc}


def question2auc(model, data, dataset, train_or_test):
    # with open(os.path.join(PATH, dataset, "ques2skill.pkl"), 'rb') as f:
    #     ques2skill_dict = pickle.load(f)

    model.eval()
    ques2pair_dict, skill2pair_dict = {}, {}
    for seq_lens, pad_data, pad_answer, pad_index, pad_label in data:
        pad_predict = model(pad_data, pad_answer, pad_index)  # 运行模型
        pack_predict = pack_padded_sequence(pad_predict, seq_lens.cpu(), enforce_sorted=True)
        pack_label = pack_padded_sequence(pad_label, seq_lens.cpu(), enforce_sorted=True)

        y_true = pack_label.data.cpu().contiguous().view(-1).detach().numpy()
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach().numpy()

        next_questions = pack_padded_sequence(pad_index, seq_lens.cpu(), enforce_sorted=True)
        next_questions = next_questions.data.cpu().contiguous().view(-1).detach().numpy()

        for ques, score, label in zip(next_questions, y_pred, y_true):
            if ques not in ques2pair_dict.keys():
                ques2pair_dict[ques] = []
            ques2pair_dict[ques].append((score, label))

            # # for skill2auc
            # skill = ques2skill_dict[ques]
            # if skill not in skill2pair_dict.keys():
            #     skill2pair_dict[skill] = []
            # skill2pair_dict[skill].append((score, label))

    ques2pair_list = sorted(ques2pair_dict.items(), key=lambda x: x[0])
    print("num of questions: ", len(ques2pair_list))
    # skill2pair_list = sorted(skill2pair_dict.items(), key=lambda x: x[0])
    # print("num of skills: ", len(skill2pair_list))

    # que2auc
    ques2auc_dict = {}
    skillNum_delete, quesNum_delete = 0, 0
    for ques, score_labels in ques2pair_list:
        score_list, label_list = zip(*score_labels)
        if np.sum(label_list) == 0.0 or np.prod(label_list) == 1.0:
            quesNum_delete += 1
            continue
        auc = metrics.roc_auc_score(y_true=label_list, y_score=score_list)
        ques2auc_dict[ques] = auc

    # # for skill2auc
    # skill2auc_dict = {}
    # for skill, score_labels in skill2pair_list:
    #     score_list, label_list = zip(*score_labels)
    #     if np.sum(label_list) == 0.0 or np.prod(label_list) == 1.0:
    #         skillNum_delete += 1
    #         continue
    #     auc = metrics.roc_auc_score(y_true=label_list, y_score=score_list)
    #     skill2auc_dict[skill] = auc

    model.train()

    print("num of deleted questions = %d" % quesNum_delete)
    print("num of deleted skills = %d" % skillNum_delete)

    # for ques2auc
    with open("../result/%s/SimQE+DKT_ques2%sAUC.pkl" % (dataset, train_or_test), 'wb') as f:
        pickle.dump(ques2auc_dict, f)
    print(ques2auc_dict)

    # # for skill2auc
    # with open("../result/%s/SimQE+DKT_skill2%sAUC.pkl" % (dataset, train_or_test), 'wb') as f:
    #     pickle.dump(skill2auc_dict, f)
    # print(skill2auc_dict)