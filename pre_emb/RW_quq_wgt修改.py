import numpy as np
import pandas as pd
import time
import os
import random
import matplotlib.pyplot as plt
import torch


class MetaPathGenerator:
    def __init__(self):
        self.ques_stuAnsList = dict()
        self.stu_quesAnsList = dict()

    def read_data(self):
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))
        # stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph","stu_ability.csv")) #考虑离散能力值
        stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_abi_w.csv")) # 考虑连续能力值
        ques_disc = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_discvalue.csv"))
        for index, row in stu_ques_df.iterrows():
            stuID, quesID = int(row['stu']), int(row['ques'])
            # 读取当前答题正确情况
            answerState = row['correct']
            if quesID not in self.ques_stuAnsList:
                self.ques_stuAnsList[quesID] = []
                # 填入邻居的学生id和答题情况和对应能力  q1:(u1,1,6)
            self.ques_stuAnsList[quesID].append((stuID, answerState,stu_ability.iloc[stuID,1]))
            if stuID not in self.stu_quesAnsList:
                self.stu_quesAnsList[stuID] = []
                # 填入邻居的题目id和答题情况和区分度  u1:(q1,1,3)
            self.stu_quesAnsList[stuID].append((quesID, answerState, ques_disc.iloc[quesID,1]))
            # self.stu_quesAnsList[stuID].append((quesID, answerState))


    def generate_wgtRW_quq(self, numWalks, walkLength, D=1.0):
        # 带权了1.0的难度

        # 获取平均能力值
        stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph","stu_abi.csv"))
        average_abiscore = np.average(stu_ability['ability'])
        rank_list=[]
        rank_list0=[]
        # 遍历序号和题目
        for idx, ques in enumerate(self.ques_stuAnsList):
            if idx % 100 == 0:
                print(idx)
                # 选择游走次数
            for j in range(0, numWalks):
                # 初始化问题、学生、答题
                theQues, theStu, theAns = ques, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # plt_x = np.array([])
                    # plt_y = np.array([])
                    # pick the next student node
                    # 取出该问题所有邻居学生信息
                    stuAnsList = list(self.ques_stuAnsList[theQues])
                    # 若邻居很多就随机选20个
                    # if len(stuAnsList) > 10:
                    #     stuAnsList = random.sample(stuAnsList, 10)
                    indexList = list(range(0, len(stuAnsList)))
                    # 根据问题选择学生
                    if i == 0:
                        # 第一个随机选择一个学生进行（需要修改，优秀选择能力中等的）
                        stuTuple, ansTuple,abilityTuple= zip(*stuAnsList)
                        ability_list = list(abilityTuple)

                        # for id in stuTuple:
                        #     stu_list.append(stu_ability.iloc[id, 1])
                        # p = self.get_possbility(ability_list, 3.5)
                        p = self.get_possbility(ability_list, average_abiscore)

                        theIndex = np.random.choice(indexList, replace=False, p=None)
                        # 取得第一个学生和答案和能力
                        theStu, theAns , theAbi = stuAnsList[theIndex]
                    else: # 此时已经两个问题[1,32]
                        # 随机取0-1 若大于1/邻居长度就继续
                        if np.random.rand() > 1 / len(stuAnsList):
                            # 分别取得邻居学生元组和答案元组
                            stuTuple, ansTuple, abilityTuple = zip(*stuAnsList)
                            # stu_list = []
                            # for id in stuTuple:
                            #     stu_list.append(stu_ability.iloc[id, 1])

                            ability_list = list(abilityTuple)
                            # 转移概率列表 相同方式计算 相同答案 概率为1
                            probList = list(1 - self.getDist(ansTuple, theAns) / D)
                            # 若存在才删除
                            if theStu in stuTuple:
                                delIndex = stuTuple.index(theStu)
                                del indexList[delIndex] # 删除学生索引
                                del probList[delIndex]

                                del ability_list[delIndex]
                            # plt_x = np.append(plt_x, ability_list)
                            # print('ability_list',ability_list,len(ability_list))
                            # print('theANS',theAns,theAns.shape)
                            # ability_list = np.array(ability_list)
                            # p = ability_list * -int(theAns) + (1 - int(theAns)) * ability_list
                            # p = torch.sigmoid(torch.tensor(p))
                            # p = p.numpy()
                            p = np.array(self.get_possbility(ability_list,average_abiscore))
                            # p = np.array(self.get_possbility(ability_list, 3.5))
                            if np.sum(probList) != 0:
                                # 将几个1 全部变为1/4 平均 修改为1 直接乘转化概率
                                # probList = self.normalize(probList)

                                # probList *= p
                                probList = self.normalize(probList)
                                # plt_y = np.append(plt_y, probList)
                                # if len(plt_x) > 1 and len(plt_y) > 1:
                                #     plt.scatter(plt_x, plt_y)
                                #     plt.xlabel('ability')
                                #     plt.ylabel('probability')
                                #     plt.savefig('pic.png')
                                #     plt.show()
                                # 随机选择到下一个index
                                theIndex = np.random.choice(indexList, replace=False, p=probList)
                                # theIndex = torch.multinomial(torch.tensor(probList),num_samples=1)# 排第几个
                                # print('indexList长度',len(indexList),indexList)
                                # index=theIndex
                                # theIndex = indexList[index]
                                # print('theIndex',theIndex)
                                # print('prob_list',probList)
                                # index=theIndex
                                # # index = indexList.index(theIndex)
                                # # print('此时索引为,',index)
                                # # print('对应在prob里的谁',probList[index])
                                # sort_p = sorted(probList).index(probList[index])
                                # sort0 = sort_p/len(probList)
                                # rank =str(sort_p)+'/'+str(len(probList))
                                # # print(rank)
                                # rank_list0.append(sort0)
                                # rank_list.append(rank)


                                # 根据索引，选择到下一个学生
                                theStu, theAns, theAbi = stuAnsList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next question node
                    # 根据学生选择问题(使用题目区分度指标)
                    if np.random.rand() > 1 / len(self.stu_quesAnsList[theStu]):
                        # 该学生对应的（问题，答案）列表
                        quesAnsList = list(self.stu_quesAnsList[theStu])
                        if len(quesAnsList) > 10: # 若超过10个问题则随机选10个
                            quesAnsList = random.sample(quesAnsList, 10)
                        # 获取索引
                        indexList = list(range(0, len(quesAnsList)))
                        # 分别得到问题和答案，区分度元组
                        quesTuple, ansTuple, discTuple = zip(*quesAnsList)
                        # quesTuple, ansTuple,  = zip(*quesAnsList)
                        # 概率列表：概率公式为（1-第一个学生答案和答案元组中的距离）
                        # 思想： 答案相同为0，转移概率=1，答案不同为1，转移概率为0
                        disc_list = list(discTuple)
                        probList = list(1 - self.getDist(ansTuple, theAns) / D)
                        # 要删除的索引（上一个问题）
                        if theQues in quesTuple:
                            delIndex = quesTuple.index(theQues)
                            del indexList[delIndex] #删除索引列表里的上一个题目索引
                            del probList[delIndex] # 删除它的概率
                            del disc_list[delIndex]
                        # 对区分度进行概率化
                        # plt_x = np.append(plt_x,disc_list)
                        p = np.array(self.softMax(np.array(disc_list)))
                        if np.sum(probList) != 0: # 如果不是全部为0，即答案全不同
                            # 分配概率
                            probList *= p
                            probList = self.normalize(probList)
                            # plt_y = np.append(plt_y, probList)
                            # if len(plt_x) > 1 and len(plt_y) > 1:
                            #     plt.scatter(plt_x, plt_y)
                            #     plt.xlabel('disc')
                            #     plt.ylabel('probability')
                            #     plt.savefig('pic.png')
                            #     plt.show()
                            # replace是否可以取相同元素，p规定每个元素的概率
                            theIndex = np.random.choice(indexList, replace=False, p=probList)
                            # 根据索引取得问题和答案
                            theQues, theAns,theDisc= quesAnsList[theIndex]
                            # theQues, theAns = quesAnsList[theIndex]
                    # 添加到问题序列
                    one_walk.append(theQues)




                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))

        # print('ranklist:', rank_list)
        # plt.hist(rank_list0[:500000], bins=20)
        # plt.savefig('1.jpg')
        # plt.show()

    @staticmethod
    def softMax(vec):
        exp = np.exp(vec)
        sumExp = sum(exp)
        result = exp * 1.0 / sumExp
        return result

    @staticmethod
    def normalize(vec):
        # 给概率为1 的平均分配自己的概率
        sumValue = np.sum(vec)
        result = np.array(vec) * 1.0 / sumValue
        return result

    @staticmethod
    def getDist(diffTuple, theDiff):
        return np.abs(np.array(diffTuple) - theDiff)


    @staticmethod
    def get_possbility(ability_list, avg):
        # 将能力转化为概率，中间大两头小
        # print(ability_list)
        def softMax(vec):
            exp = np.exp(vec)
            sumExp = sum(exp)
            result = exp * 1.0 / sumExp
            return result

        if len(ability_list) == 1:
            return [1]
        else:
            diff = np.abs(np.array(ability_list) - avg)
            # p = 1 - np.array(diff / np.sum(diff))
            # return p / np.sum(p)
            return softMax(- diff)

if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    data_set = "EdNet"
    # for data_set in ["ASSIST09", "EdNet"]:
    save_folder = "../pre_emb/%s/walks" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for num_walks in [10]:
        for walk_length in [80]:
            out_file_name = os.path.join(save_folder, "walks_quq_wgt6.2(10)_%d_%d.txt") % (num_walks, walk_length)
            outfile = open(out_file_name, 'w')

            t = time.time()
            mpg = MetaPathGenerator()
            # 读取数据生成了两组字典
            mpg.read_data()
            mpg.generate_wgtRW_quq(numWalks=num_walks, walkLength=walk_length)
            print("time consuming: %d seconds" % (time.time() - t))
            outfile.close()
