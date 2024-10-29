import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from scipy import interpolate
import matplotlib


def file_to_list(filename, min_seq_len=3, max_seq_len=200, truncate=False):

    def split_func(_seq_len):
        _split_list = []
        while _seq_len > 0:
            if _seq_len >= max_seq_len:
                _split_list.append(max_seq_len)
                _seq_len -= max_seq_len
            elif _seq_len >= min_seq_len:
                _split_list.append(_seq_len)
                _seq_len -= _seq_len
            else:
                _seq_len -= min_seq_len
        return len(_split_list), _split_list

    seq_lens, ques_ids, answers = [], [], []
    k_split = -1
    with open(filename) as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if i % 3 == 0:
            seq_len = int(line)
            if seq_len < min_seq_len:
                i += 3
                continue
            else:
                k_split, split_list = split_func(seq_len)
                if truncate:
                    k_split = 1
                    seq_lens.append(split_list[0])
                else:
                    seq_lens += split_list
        else:
            line = line.split(',')
            array = [int(eval(e)) for e in line]
            if i % 3 == 1:
                for j in range(k_split):
                    ques_ids.append(array[max_seq_len * j: max_seq_len * (j + 1)])

            else:
                for j in range(k_split):
                    answers.append(array[max_seq_len * j: max_seq_len * (j + 1)])
        i += 1
    assert len(seq_lens) == len(ques_ids) == len(answers)
    stu_num = len(seq_lens)
    attempt_num = sum(seq_lens)
    print('stu_num：',stu_num,end='  ')
    print('attempt_num：',attempt_num,end='  ')
    # print(type(ques_ids))
    ques_seq = []
    for i in ques_ids:
        ques_seq+=i
    # print(set(ques_seq))
    ques_num = len(set(ques_seq))
    print('ques_num：',ques_num)
    aver_response = attempt_num/ques_num
    print('每题平均作答：',aver_response)
    aver_answer = attempt_num / stu_num
    print('学生平均答题：', aver_answer)
    sparsity = 1 - attempt_num / (stu_num * ques_num)

    print('稀疏度：',sparsity)
    return sparsity

    # return seq_lens, ques_ids, answers

def calculate_sparsity_sort(data_set):
    # for num in range(1,6):
    num=1

    with open(DATAPATH + data_set + "/train_test/train_question.txt", 'r') as f:
        attempt_count = 0
        ques_seq = []
        lines = f.readlines()

        stu_num=int(len(lines)/3)
        print('stu_num',stu_num)
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            # print('line',line)
            if i % 3 == 0:
                seq_len = int(line) # 读取了三行中第一列
                # print(seq_len)
                attempt_count+=seq_len


            if i % 3 == 1:
                ques_list = line  # 读取了三行中第二列

                ques_seq.append(ques_list)


            i+=1
        print(ques_seq)
                # print(seq_len)
                # if seq_len < min_seq_len:



        print('attempt_count',attempt_count)


def calculate_sparsity(data_set):
    inte_num = pd.read_csv(os.path.join(DATAPATH, data_set, "graph", "stu_ques.csv")).shape[0]
    print('inte_num:', inte_num)
    stu_num = np.load(os.path.join(DATAPATH, data_set, "graph", "stu_skill_mat.npy")).shape[0]
    print('stu_num:',stu_num)
    ques_num = np.load(os.path.join(DATAPATH, data_set, "graph", "ques_skill_mat.npy")).shape[0]
    print('ques_num',ques_num)
    sparsity = 1-inte_num/(stu_num*ques_num)
    print(sparsity)
    return sparsity

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

def draw(X,Y,model,draw_type):

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    x = X
    y1,y2,y3,y4,y5,y6,y7,y8 = Y
    m1,m2,m3,m4,m5,m6,m7,m8 = model
    if draw_type == 'plot':
        plt.plot(x, y1, label=m1)
        plt.plot(x, y2, label=m2)
        plt.plot(x, y3, label=m3)
        plt.plot(x, y4, label=m4)
        plt.plot(x, y5, label=m5)
        plt.plot(x, y6, label=m6)
        plt.plot(x, y7, label=m7)
        plt.plot(x, y8, label=m8)
    # plt.scatter(x, y1, label=m1)
    # plt.scatter(x, y2, label=m2)
    # plt.scatter(x, y3, label=m3)

    # x_mean, y_mean, y_sem = bins_mean_sem(x, y1, BIN)
    # eb1 = plt.errorbar(x, y_mean, y_sem, label=m1, linestyle='--')
    # eb1[-1][0].set_linestyle('--')
    # x_mean, y_mean, y_sem = bins_mean_sem(x, y2, BIN)
    # # plt.errorbar(x_mean, y_mean, y_sem, label=MODEL2)
    # eb2 = plt.errorbar([x + 0 for x in x_mean], y_mean, y_sem, label=m2, linestyle='-')
    # eb2[-1][0].set_linestyle('-')
    # x_mean, y_mean, y_sem = bins_mean_sem(x, y3, BIN)
    # # plt.errorbar(x_mean, y_mean, y_sem, label=MODEL2)
    # eb3 = plt.errorbar([x + 0 for x in x_mean], y_mean, y_sem, label=m3, linestyle='-.')
    # eb3[-1][0].set_linestyle('-.')
        plt.xlabel('稀疏度')
        plt.ylabel('AUC')
        # plt.title('%s2%sAUC_compare' % (factor_mode, auc_mode))
        plt.grid(True)
        plt.legend()
        # plt.savefig("%s/%s2%sAUC_%s_bin%d(%s)_bw.png" % (DATASET, factor_mode, auc_mode, draw_mode, BIN, MODEL1))
        plt.show()
        plt.close()

    if draw_type=="bar":
        fb1 = np.array([0.8853,0.7541, 0.7367,0.7402, 0.7318, 0.7050, ])# DKT_Q
        fb2 = np.array([0.5, 0.5,0.7605,0.5,0.7731, 0.7543]) # PEBG
        fb3 = np.array([0.8923,0.7723, 0.7663, 0.7916, 0.7939, 0.7812]) #SimQE
        # fb4 = [0.850, 0.877, 0.865]
        # fb5 = [0.77, 0.760, 0.806]
        # fb6 = [0.847, 0.873, 0.877]
        fig, ax = plt.subplots()

        index_fb1 = [1, 2, 3,4,5,6]
        index_fb2 = [1.1, 2.1, 3.1,4.1,5.1,6.1]
        index_fb3 = [1.2, 2.2, 3.2,4.2,5.2,6.2]

        plt.bar(index_fb1, fb1, width=0.1, label='DKT_QE', zorder=1)
        plt.bar(index_fb2, fb2, width=0.1, label='Pebg_QE+DKT',  zorder=1)
        plt.bar(index_fb3, fb3, width=0.1, label='Sim_QE+DKT',  zorder=1)

        plt.legend(frameon=False)
        # for i in range(3):
        #     plt.scatter(index_fb3[i], fb5[i], s=400, marker="*", color='black', zorder=2)

        ax.set_ylim(0.65, 0.90)
        # plt.xticks([1.15, 2.15, 3.15], ['Statics2011', 'ASSIST17', 'EdNet'], fontsize=16)
        ax.set_xticks([1.15, 2.15, 3.15,4.15,5.15,6.15], ['Statics2011(0.758)','ASSIST17(0.900)','EdNet(0.988)','Eedi(0.996)','ASSIST09(0.996)', 'ASSIST12(0.998)',])
        ax.set_yticks(np.arange(0.65, 0.90, 0.05))

        # plt.scatter(2.2, 1.01, s=400, marker="*", color='black', zorder=2)  # 120
        # plt.text(2.25, 1, 'MVP')
        # plt.text(0.8, 1.32, '(b)')
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('AUC', fontweight='bold')

        ax2 = ax.twinx()
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        # plt.savefig('./bar_example.png', bbox_inches='tight', dpi=1200)
        plt.show()


if __name__ =="__main__":
    DATAPATH = 'E:\Study\KT\DataProcess2022/'
    DATASET = ['ASSIST09','ASSIST12','ASSIST17','EdNet','Statics2011','Eedi']
    BIN=20
    sp_list = []
    for dataset in DATASET:
        print(dataset)
        spar = file_to_list(DATAPATH + dataset + "/train_test/train_question.txt")
        sp_list.append(spar)
        # for num in range(1,6):
            # file_to_list(DATAPATH+dataset+"/train_test/train_question_%s.txt"%num )
    #     calculate_sparsity_sort(dataset)
    #     sp_list.append(calculate_sparsity(dataset))
    print(sp_list)
    X = sorted(sp_list)
    print(X)
    index = np.array(sp_list).argsort()
    print(index)
    # print(X)
    auc_list1 = np.array([0.7671,0.7297,0.7186,0.7006,0.8403,0.7624])[index] #dkt
    auc_list2 = np.array([0.7618,0.7219,0.7159,0.6869,0.8350,0.7466])[index] # dkvmn
    auc_list3 = np.array([0.7939,0.7812,0.7723,0.7663,0.8923,0.7916])[index]# simkt
    auc_list4 = np.array([0.7542, 0.7541, 0.7661, 0.7649, 0.8738, 0.7911])[index]  # dhkt
    auc_list5 = np.array([0.7859, 0.7594, 0.7723, 0.7641, 0.8907, 0.7881])[index]  # GIKT
    auc_list6 = np.array([0.7899, 0.7665, 0.7620, 0.7647, 0.8718, 0.7890])[index]  # AKT
    auc_list7 = np.array([0.7781, 0.7691, 0.7636, 0.7604, 0.8786, 0.7768])[index]  # GASKT
    auc_list8 = np.array([0.7625, 0.7298, 0.7077, 0.6986, 0.8252, 0.7486])[index]  # SAKT
    model=['DKT','DKVMN','SimKT','DHKT','GIKT','AKT','GASKT','SAKT']
    #
    #
    # # auc_list1=auc_list1[X]
    print(auc_list1)
    Y = (auc_list1, auc_list2, auc_list3,auc_list4, auc_list5, auc_list6,auc_list7,auc_list8)
    draw(X, Y,model,draw_type='bar')

    # DATASET='ASSIST09/'