import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from scipy import interpolate
import matplotlib
# PATH = "D:/ZJP/DataProcess1.0/DataProcess1.0/data_GIKT"


def draw(X,Y, factor_mode, auc_mode, draw_mode):
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = ['Times New Roman']
    # plt.style.use('default')
    # way 1:
    # plt.plot(x, y)
    # x1,x2,x3,x4,x5,x6 = X
    # y1,y2,y3,y4,y5,y6 = Y
    x1,x4,x6 = X
    y1,y4,y6 = Y
    # way 2:
    if draw_mode == 'scatter':
        plt.scatter(x1, y1, label='SimQE+DKT')
        # plt.scatter(x2, y2, label=MODEL2)
        # plt.scatter(x3, y3, label=MODEL3)
        plt.scatter(x4, y4, label='Q-E+DKT')
        # plt.scatter(x5, y5, label=MODEL5)
        plt.scatter(x6, y6, label=MODEL6)

    # way 3:
    # f = interpolate.interp1d(x, y)
    # xnew = np.linspace(np.min(x), np.max(x), 10)
    # ynew = f(xnew)
    # plt.plot(x, y, 'o', xnew, ynew, '-')

    # way 4:
    elif draw_mode == 'errorbar':
        x_mean, y_mean, y_sem = bins_mean_sem(x1, y1, BIN)
        plt.figure(figsize=(10,6),dpi=100)
        # eb1 = plt.errorbar(x_mean, y_mean, y_sem, label='MSQE+DKT',  linestyle='-',color='#32AEEC')
        eb1 = plt.errorbar(x_mean, y_mean, y_sem, label='SimQE+DKT', linestyle='-', color=(0.121, 0.466, 0.705))
        eb1[-1][0].set_linestyle('-.')
        # x_mean, y_mean, y_sem = bins_mean_sem(x2, y2, BIN)
        # # plt.errorbar(x_mean, y_mean, y_sem, label=MODEL2)
        # eb2 = plt.errorbar([x + 0 for x in x_mean], y_mean, y_sem, label=MODEL2, linestyle='-')
        # eb2[-1][0].set_linestyle('-')
        # x_mean, y_mean, y_sem = bins_mean_sem(x3, y3, BIN)
        # # plt.errorbar(x_mean, y_mean, y_sem, label=MODEL2)
        # eb3 = plt.errorbar([x + 0 for x in x_mean], y_mean, y_sem, label=MODEL3, linestyle='-.')
        # eb3[-1][0].set_linestyle('-.')
        x_mean, y_mean, y_sem = bins_mean_sem(x4, y4, BIN)
        # plt.errorbar(x_mean, y_mean, y_sem, label=MODEL2)
        # eb4 = plt.errorbar([x + 0.7 for x in x_mean], y_mean, y_sem,color='#E3863C', label='Q-E+DKT', linestyle='-')
        eb4 = plt.errorbar([x + 0.7 for x in x_mean], y_mean, y_sem, color='orange', label='Q-E+DKT', linestyle='-')
        eb4[-1][0].set_linestyle(':')
        # x_mean, y_mean, y_sem = bins_mean_sem(x5, y5, BIN)
        # eb5 = plt.errorbar([x + 0 for x in x_mean], y_mean, y_sem, label=MODEL5, linestyle='solid')
        # eb5[-1][0].set_linestyle(':')
        x_mean, y_mean, y_sem = bins_mean_sem(x6, y6, BIN)
        # eb6 = plt.errorbar([x + 1.5 for x in x_mean], y_mean, y_sem, color="#E9262E",label=MODEL6, linestyle='-')
        eb6 = plt.errorbar([x + 1.5 for x in x_mean], y_mean, y_sem, color="green", label=MODEL6, linestyle='-')
        eb6[-1][0].set_linestyle(':')

    # plt.xlabel(factor_mode)
    plt.xlabel('Attempt number of question',fontdict={'fontsize':14,'family':'Times New Roman'})
    # plt.xlabel('',fontsize=14)
    plt.ylabel('AUC',fontdict={'fontsize':14,'family':'Times New Roman'})
    # plt.title('%s2%sAUC_compare' % (factor_mode, auc_mode))
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.savefig("E:\Study\SimKT/Fig5.png",dpi=600)
    # plt.savefig("%s/%s2%sAUC_%s_bin%d(%s-%s-%s)_bw.png" % (DATASET, factor_mode, auc_mode, draw_mode, BIN, MODEL1,MODEL4,MODEL6))
    plt.show()
    plt.close()


def bins_mean_sem(x, y, n_bins=10):
    min_x = np.min(x)
    max_x = np.max(x)
    step = (max_x - min_x) / n_bins
    x_bins, y_bins = [], []

    last_k = 0
    x_one_bin, y_one_bin = [], []
    for i in range(len(x)):
        k = int((x[i] - min_x) / step)
        if k != last_k:
            x_bins.append(x_one_bin)
            y_bins.append(y_one_bin)
            last_k = k
            x_one_bin = []
            y_one_bin = []
        else:
            x_one_bin.append(x[i])
            y_one_bin.append(y[i])

    x_mean, y_mean, y_sem = [], [], []
    for x_bin in x_bins:
        x_mean.append(np.mean(x_bin))
    for y_bin in y_bins:
        y_mean.append(np.mean(y_bin))
        # y_sem.append(np.std(y_bin) / np.sqrt(len(y_bin)))  # 标准误差
        y_sem.append(np.std(y_bin))  # 标准差

    return x_mean, y_mean, y_sem


def compare_skill_x2auc(factor_mode, auc_mode, draw_mode):
    # x2auc_list = MergeSameCount(x2auc_list)
    x2auc_list1 = get_skill_x2auc(MODEL1, factor_mode, auc_mode)
    x2auc_list2 = get_skill_x2auc(MODEL2, factor_mode, auc_mode)

    x2auc_list1 = sorted(x2auc_list1, key=lambda t: t[0])
    x2auc_list2 = sorted(x2auc_list2, key=lambda t: t[0])
    x_list1, auc_list1 = zip(*x2auc_list1)
    x_list2, auc_list2 = zip(*x2auc_list2)
    draw(x_list1, auc_list1, x_list2, auc_list2, factor_mode, auc_mode, draw_mode)


def get_skill_x2auc(modelName, factor_mode, auc_mode):
    with open(os.path.join(PATH, DATASET, "skill2%s.pkl" % factor_mode), 'rb') as f:
        skill2x_dict = pickle.load(f)
    with open(os.path.join(DATASET, "%s_skill2%sAUC.pkl" % (modelName, auc_mode)), 'rb') as f:
        skill2auc_dict = pickle.load(f)

    x2auc_list = []
    for skill, auc in skill2auc_dict.items():
        if skill not in skill2auc_dict.keys():  # for x != 'skillCount'
            continue
        x = skill2x_dict[skill]
        # x = skill2x_dict.get(skill, 0)
        # if x > 2:
        #     continue
        x2auc_list.append((x, auc))
    return x2auc_list


def compare_question_x2auc(factor_mode, auc_mode, draw_mode):
    # x2auc_list = MergeSameCount(x2auc_list)
    x2auc_list1 = get_question_x2auc(MODEL1, factor_mode, auc_mode)
    # x2auc_list2 = get_question_x2auc(MODEL2, factor_mode, auc_mode)
    # x2auc_list3 = get_question_x2auc(MODEL3, factor_mode, auc_mode)
    x2auc_list4 = get_question_x2auc(MODEL4, factor_mode, auc_mode)
    # x2auc_list5 = get_question_x2auc(MODEL5, factor_mode, auc_mode)
    x2auc_list6 = get_question_x2auc(MODEL6, factor_mode, auc_mode)

    x2auc_list1 = sorted(x2auc_list1, key=lambda t: t[0])
    # x2auc_list2 = sorted(x2auc_list2, key=lambda t: t[0])
    # x2auc_list3 = sorted(x2auc_list3, key=lambda t: t[0])
    x2auc_list4 = sorted(x2auc_list4, key=lambda t: t[0])
    # x2auc_list5 = sorted(x2auc_list5, key=lambda t: t[0])
    x2auc_list6 = sorted(x2auc_list6, key=lambda t: t[0])

    x_list1, auc_list1 = zip(*x2auc_list1)
    # x_list2, auc_list2 = zip(*x2auc_list2)
    # x_list3, auc_list3 = zip(*x2auc_list3)
    x_list4, auc_list4 = zip(*x2auc_list4)
    # x_list5, auc_list5 = zip(*x2auc_list5)
    x_list6, auc_list6 = zip(*x2auc_list6)
    # print(x_list6,auc_list6)
    # X = (x_list1,x_list2,x_list3,x_list4,x_list5,x_list6)
    # Y = (auc_list1,auc_list2,auc_list3,auc_list4,auc_list5,auc_list6)
    X = (x_list1,  x_list4,  x_list6)
    Y = (auc_list1, auc_list4,auc_list6)
    draw(X,Y,factor_mode, auc_mode, draw_mode)


def get_question_x2auc(modelName, factor_mode, auc_mode):
    with open(os.path.join(DATASET, "%s_ques2%s.pkl" % (modelName,factor_mode)), 'rb') as f:
        ques2x_dict = pickle.load(f)
    with open(os.path.join(DATASET, "%s_ques2%sAUC.pkl" % (modelName, auc_mode)), 'rb') as f:
        ques2auc_dict = pickle.load(f)

    x2auc_list = []
    for ques, auc in ques2auc_dict.items():
        x = ques2x_dict.get(ques, 0)
        # if x > 800:
        #     continue
        x2auc_list.append((x, auc))
    return x2auc_list


if __name__ == '__main__':
    BIN = 20
    MODEL1 = 'SimQE+DKT'
    # MODEL2 = 'DHKT'
    # MODEL3 = 'COKT'
    MODEL4 = 'DKT_Q'
    # MODEL5 = 'AKT'
    MODEL6 = 'PEBG+DKT'


    DATASET = "../result/ASSIST09"
    # trainQuesCount, testQuesCount, testSkillCount, testDiffStd, testDiffVar, Gini, testReverseRatio, quesCount
    # compare_skill_x2auc('testDiffStd', 'test', 'errorbar')
    compare_question_x2auc('trainQuesCount', 'test', 'errorbar')
