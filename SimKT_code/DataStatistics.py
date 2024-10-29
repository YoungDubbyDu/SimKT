import pickle
import numpy as np
import pandas as pd
import linecache
import csv
k_split = None


def file_to_list(filename, min_seq_len=3, max_seq_len=200):
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

    seq_lens = []
    ques_ids = []
    answers = []
    global k_split
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
                seq_lens += split_list
                print(split_list) # 表示每个数字
        else:
            line = line.split(',')
            array = [int(e) for e in line]
            if i % 3 == 1:
                for j in range(k_split):
                    ques_ids.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            else:
                for j in range(k_split):
                    answers.append(array[max_seq_len * j: max_seq_len * (j + 1)])
        i += 1
    # for integrity, check the lengths
    assert len(seq_lens) == len(ques_ids) == len(answers)
    return seq_lens, ques_ids, answers


def get_ques2count(train_or_test):
    seq_lens, ques_ids, answers = file_to_list(DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test)
    ques2count_dict = {}
    for ques_list in ques_ids:
        for ques_id in ques_list:
            ques2count_dict[ques_id] = ques2count_dict.get(ques_id, 0) + 1
    with open(SAVEPATH+DATASET + "ques2%sQuesCount.pkl" % train_or_test, 'wb') as f:
        pickle.dump(ques2count_dict, f)
    print(ques2count_dict)


def get_skill2count(train_or_test):
    seq_lens, skill_ids, answers = file_to_list(DATASET + "%s_skill.csv" % train_or_test)
    skill2count_dict = {}
    for skill_list in skill_ids:
        for skill_id in skill_list:
            skill2count_dict[skill_id] = skill2count_dict.get(skill_id, 0) + 1
    with open(DATASET + "skill2%sSkillCount.pkl" % train_or_test, 'wb') as f:
        pickle.dump(skill2count_dict, f)
    print(skill2count_dict)


def test_skill_count():
    seq_lens, skill_ids, answers = file_to_list(DATASET + "test_skill.csv")
    skill2count_dict = {}
    for skill_list in skill_ids:
        for skill_id in skill_list:
            skill2count_dict[skill_id] = skill2count_dict.get(skill_id, 0) + 1
    with open(DATASET + "test_skill2count.pkl", 'wb') as f:
        pickle.dump(skill2count_dict, f)
    print(skill2count_dict)


def get_skill_answers(skill_list, answer_list):
    skill2answers_dict = {}
    for s, a in zip(skill_list, answer_list):
        if s not in skill2answers_dict.keys():
            skill2answers_dict[s] = []
        skill2answers_dict[s].append(a)
    return skill2answers_dict


def calculate_Gini(answer_list):
    p = sum(answer_list) * 1.0 / len(answer_list)
    Gini = 2 * p * (1 - p)
    return Gini


def get_skill_Gini():
    seq_lens, skill_ids, answers = file_to_list(DATASET + "test_skill.csv")
    skill2Ginis_dict = {}
    for skill_list, answer_list in zip(skill_ids, answers):
        skill2answers_dict = get_skill_answers(skill_list, answer_list)
        for _skill, _answer_list in skill2answers_dict.items():
            if _skill not in skill2Ginis_dict.keys():
                skill2Ginis_dict[_skill] = []
            skill2Ginis_dict[_skill].append(calculate_Gini(_answer_list))

    skill2MeanGini_dict = {}
    for skill, Gini_list in skill2Ginis_dict.items():
        skill2MeanGini_dict[skill] = np.mean(Gini_list)
    with open(DATASET + "test_skill2Gini.pkl", 'wb') as f:
        pickle.dump(skill2MeanGini_dict, f)
    print(skill2MeanGini_dict)


def get_skill2DiffVar(train_or_test):
    ques_diff = pd.read_csv(DATASET + "question_diff10.csv", names=['ques', 'diff'], sep=',')
    ques2diff_dict = dict(zip(ques_diff['ques'], ques_diff['diff']))
    seq_lens, skill_ids, answers = file_to_list(DATASET + "%s_skill.csv" % train_or_test)
    seq_lens, ques_ids, answers = file_to_list(DATASET + "%s_ques.csv" % train_or_test)
    skill2vars_dict = {}
    for skill_list, ques_list in zip(skill_ids, ques_ids):
        temp_skill2diffs_dict = {}
        for skill, ques in zip(skill_list, ques_list):
            if skill not in temp_skill2diffs_dict.keys():
                temp_skill2diffs_dict[skill] = []
            temp_skill2diffs_dict[skill].append(ques2diff_dict[ques])
        for skill, diff_list in temp_skill2diffs_dict.items():
            if skill not in skill2vars_dict.keys():
                skill2vars_dict[skill] = []
            skill2vars_dict[skill].append(np.var(diff_list))

    skill2DiffVar_dict = {}
    for skill, var_list in skill2vars_dict.items():
        skill2DiffVar_dict[skill] = np.mean(var_list)
    with open(DATASET + "skill2%sDiffVar.pkl" % train_or_test, 'wb') as f:
        pickle.dump(skill2DiffVar_dict, f)
    print(skill2DiffVar_dict)


def get_skill2DiffStd(train_or_test):
    ques_diff = pd.read_csv(DATASET + "question_diff10.csv", names=['ques', 'diff'], sep=',')
    ques2diff_dict = dict(zip(ques_diff['ques'], ques_diff['diff']))
    seq_lens, skill_ids, answers = file_to_list(DATASET + "%s_skill.csv" % train_or_test)
    seq_lens, ques_ids, answers = file_to_list(DATASET + "%s_ques.csv" % train_or_test)
    skill2stds_dict = {}
    for skill_list, ques_list in zip(skill_ids, ques_ids):
        temp_skill2diffs_dict = {}
        for skill, ques in zip(skill_list, ques_list):
            if skill not in temp_skill2diffs_dict.keys():
                temp_skill2diffs_dict[skill] = []
            temp_skill2diffs_dict[skill].append(ques2diff_dict[ques])
        for skill, diff_list in temp_skill2diffs_dict.items():
            if skill not in skill2stds_dict.keys():
                skill2stds_dict[skill] = []
            skill2stds_dict[skill].append(np.std(diff_list))

    skill2DiffStd_dict = {}
    for skill, std_list in skill2stds_dict.items():
        skill2DiffStd_dict[skill] = np.mean(std_list)
    with open(DATASET + "skill2%sDiffStd.pkl" % train_or_test, 'wb') as f:
        pickle.dump(skill2DiffStd_dict, f)
    print(skill2DiffStd_dict)


def get_reverseCount(answer_list):
    reverseCount = 0
    for i in range(1, len(answer_list)):
        if answer_list[i-1] != answer_list[i]:
            reverseCount += 1
    return reverseCount


def get_skill2reverseRatio(train_or_test):
    seq_lens, skill_ids, answers = file_to_list(DATASET + "%s_skill.csv" % train_or_test)
    skill2reverseRatios_dict = {}
    for skill_list, answer_list in zip(skill_ids, answers):  # for one user
        skill2answers_dict = get_skill_answers(skill_list, answer_list)
        for sk, ans_list in skill2answers_dict.items():
            if sk not in skill2reverseRatios_dict.keys():
                skill2reverseRatios_dict[sk] = []
            skill2reverseRatios_dict[sk].append(get_reverseCount(ans_list) * 1.0 / len(ans_list))

    skill2MeanReverseRatio_dict = {}
    for skill, reverseRatio_list in skill2reverseRatios_dict.items():
        skill2MeanReverseRatio_dict[skill] = np.mean(reverseRatio_list)
    with open(DATASET + "skill2%sReverseRatio.pkl" % train_or_test, 'wb') as f:
        pickle.dump(skill2MeanReverseRatio_dict, f)
    print(skill2MeanReverseRatio_dict)



def train_question_count(train_or_test):
    filename = DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test
    attempt_count=[]
    # seq_lens, ques_ids, answers = file_to_list(DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test)
    with open(filename) as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # print('line',line)

        if i % 3 == 0:
            seq_len = int(line)
            attempt_count.append(seq_len)
            # print(seq_len)
            # if seq_len < min_seq_len:
            i += 3
            continue
        # else:
        #     # k_split, split_list = split_func(seq_len)
        #     seq_lens += split_list
        #     print(split_list) # 表示每个数字
    # else:
    print(attempt_count,len(attempt_count)) #学生答题数的列表
    sort_count = sorted(attempt_count)
    print('sort_count',sort_count) # 答题数排序
    sort_index = np.array(attempt_count).argsort()
    print('sort_index',sort_index)
    # sort1 = sort_count[0:666]
    index1 = sort_index[0:666]

    index2 = sort_index[666:2*666]
    index3 = sort_index[666*2:3*666]
    index4 = sort_index[666*3:4*666]
    index5 = sort_index[4*666:5*666]

    # sort2 = sort_count[666:666*2]
    # sort3 = sort_count[666*2:666*3]
    # sort4 = sort_count[666*3:666*4]
    # sort5 = sort_count[666*4:666*5]
    # print(sort1)
    # print(index1)
    # num = 1
    # for i in [index1,index2,index3,index4,index5]:
    #
    #     print(num,i)
    #     attempt_txt(i,filename,num)
    #     num+=1

def attempt_txt(index_list,filename,num):
    with open(DATAPATH + DATASET + "train_test/train_question_%s.txt"%num, 'w') as f:
        # print(index_list)
        for i in index_list:
            for k in range(3*i,3*i+3):
                text = linecache.getline(filename, k + 1)
                f.write(text)
    # # f = open(DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test, 'r', encoding='utf-8')
    # data = pd.read_csv(DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test)
    # print(data)
    # # for i in index1:
    # #     new = data[i:i+2]
    # #     new = pd.concat([new,new])
    # # new.to_csv(DATAPATH+DATASET + "train_test/%s_question_1.csv" % train_or_test)


def split_dataset(filename, min_seq_len=3, max_seq_len=200, truncate=False):
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
    # for integrity, check the lengths
    assert len(seq_lens) == len(ques_ids) == len(answers)
    return seq_lens, ques_ids, answers


def split_train_attempt(train_or_test):
    """
    按照训练集答题数等分数据集
    """
    filename = DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test
    attempt_count=[]
    # seq_lens, ques_ids, answers = file_to_list(DATAPATH+DATASET + "train_test/%s_question.txt" % train_or_test)
    with open(filename,'r') as file:
        lines = file.readlines()
    i = 0
    stu_index_list=[]
    stu_index = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # print('line',line)
        if i % 3 == 0:
            seq_len = int(line)
            if seq_len>=3:
                stu_index_list.append(stu_index)
                stu_index+=1
                # 学生答题大于3 才能用
                attempt_count.append(seq_len)
            else:
                stu_index+=1
            # print(seq_len)
            # if seq_len < min_seq_len:
            i += 3
            continue
        # else:
        #     # k_split, split_list = split_func(seq_len)
        #     seq_lens += split_list
        #     print(split_list) # 表示每个数字
    # else:
    print('筛选后的学生序列',stu_index_list,len(stu_index_list))
    print(attempt_count,len(attempt_count)) #学生答题数的列表
    sort_count = sorted(attempt_count)
    print('sort_count',sort_count,len(sort_count)) # 答题数排序
    attempt_sum = sum(sort_count)
    print('交互总次数',attempt_sum)
    attempt_each = attempt_sum/5
    print('每组交互数,',attempt_each)
    sort_index = np.array(attempt_count).argsort()
    print('sort_index',sort_index,len(sort_index)) # 一共3071个学生
    real_index_list=[]
    for each_index in sort_index:
        real_index = stu_index_list[each_index]
        real_index_list.append(real_index)
    print('对应原数据集中索引为',real_index_list)
    sum_at=0
    stunum=0
    stu_num_list=[]
    for t in sort_count:
        sum_at += t
        stunum += 1
        if (sum_at+100) >= attempt_each: # 每次超过一次就重新计数
            print('此时交互数',sum_at)
            print('此时学生数',stunum)
            sum_at = 0
            stu_num_list.append(stunum)
            stunum = 0

    stunum=len(sort_count)-sum(stu_num_list)
    stu_num_list.append(stunum)
    print('此时交互数', sum(sort_count[-stunum:]))
    print('此时学生数', stunum)
    print('每个数据集里对应学生数',stu_num_list)
    print(sum(stu_num_list))
    index1 = real_index_list[0:stu_num_list[0]]
    print(len(index1))
    index2 = real_index_list[stu_num_list[0]:stu_num_list[0]+stu_num_list[1]]
    print(len(index2))
    index3 = real_index_list[stu_num_list[0]+stu_num_list[1]:stu_num_list[2]+stu_num_list[1]+stu_num_list[0]]
    print(len(index3))
    index4 = real_index_list[stu_num_list[1]+stu_num_list[2]+stu_num_list[0]:stu_num_list[2]+stu_num_list[3]+stu_num_list[0]+stu_num_list[1]]
    print(len(index4))
    index5 = real_index_list[stu_num_list[2]+stu_num_list[3]+stu_num_list[0]+stu_num_list[1]:stu_num_list[3]+stu_num_list[4]+stu_num_list[1]+stu_num_list[0]+stu_num_list[2]]
    print(len(index5))
    if index5[stu_num_list[4]-1] ==real_index_list[sum(stu_num_list)-1]:
        print('Split successfully')
    num = 1
    for i in [index1,index2,index3,index4,index5]:
        print(num,i)
        attempt_txt(i,filename,num)
        num+=1

if __name__ == '__main__':
    DATAPATH = "../data/"
    SAVEPATH = "../result/"
    DATASET = "ASSIST09/"
    # split_dataset(DATAPATH+DATASET+"/train_test/train_question.txt")
    split_train_attempt('train')
    # train_question_count('train')
    # get_ques2count('train')

    # get_skill2count('test')

    # get_skill_Gini()

    # get_skill2DiffVar('test')
    # get_skill2DiffStd('test')

    # get_skill2reverseRatio('test')

