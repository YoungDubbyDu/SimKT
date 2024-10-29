import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os
from pre_emb.Wgt_Walker import WeightedWalker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaPathGenerator:
    def __init__(self):
        self.ques_skillDiffList = dict()
        self.skill_quesDiffList = dict()

    def read_data(self):
        # 加载题目和技能对应图
        ques_skill_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_skill.csv"))
        ques_skill_var = pd.read_csv(os.path.join(data_path, data_set, "graph", "skill_var.csv"))
        with open(os.path.join(data_path, data_set, "attribute", "quesID2diffValue_dict.txt")) as f:
            ques2diff_dict = eval(f.read())
        # 转化为字典形式的邻接表：{q1:[(k1,w11),(k3,w13)]}
        for index, row in ques_skill_df.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            # 读取当前的难度
            diffValue = ques2diff_dict[quesID]
            if quesID not in self.ques_skillDiffList:
                self.ques_skillDiffList[quesID] = []
            self.ques_skillDiffList[quesID].append((skillID, diffValue, ques_skill_var.iloc[skillID,1]))
            if skillID not in self.skill_quesDiffList:
                self.skill_quesDiffList[skillID] = []
            self.skill_quesDiffList[skillID].append((quesID, diffValue))

    def generate_wgtRW_qkq(self, numWalks, walkLength, D):  # D notates the biggest distance between diff
        # 遍历所有题目，字典中的所有键

        for idx, ques in enumerate(self.ques_skillDiffList):
            if idx % 1000 == 0:
                print(idx)
                # 每一个节点可能会随机采样很多次

            for j in range(0, numWalks):  # new sequence
                # 初始化问题-技能-权重
                theQues, theSkill, theDiff = ques, None, None
                one_walk = [theQues] # 初始化列表
                # 固定化随机游走序列长度

                for i in range(0, walkLength):
                    # plt_x = np.array([])
                    # plt_y = np.array([])
                    # pick the next skill node
                # 从题目到技能
                    skillDiffList = list(self.ques_skillDiffList[theQues])
                    indexList = list(range(0, len(skillDiffList)))
                    # print(theQues)
                    # print(skillDiffList)
                    if i == 0:
                        # 修改 概率 选择时优先选择粒度更低的题目
                        skillTuple, diffTuple, varTuple = zip(*skillDiffList)
                        var_list = list(varTuple)
                        # plt_x.append(var_list)
                        # plt_x = np.append(plt_x,var_list)
                        # p = np.array(var_list)

                        p = np.array(var_list)
                        # for id in stuTuple:
                        #     stu_list.append(stu_ability.iloc[id, 1])
                        # -p是选择细粒度
                        # p = self.softMax(p)
                        p = self.softMax(-p)
                        # plt_y = np.append(plt_y,p)
                        theIndex = np.random.choice(indexList, replace=False, p=p)
                        # 取得第一个技能和难度和粒度
                        theSkill, theDiff ,theVar= skillDiffList[theIndex]
                    else:
                        if np.random.rand() > 1 / len(skillDiffList):
                            skillTuple, diffTuple ,varTuple= zip(*skillDiffList)
                            var_list = list(varTuple)
                            # 删除上一个skill
                            delIndex = skillTuple.index(theSkill)

                            del indexList[delIndex]  # 删除学生索引
                            del var_list[delIndex]

                            p = np.array(var_list)
                            # plt_x = np.append(plt_x,var_list)


                            # for id in stuTuple:
                            #     stu_list.append(stu_ability.iloc[id, 1])
                            # -p是细粒度
                            p = self.softMax(-p)
                            # plt_y = np.append(plt_y,p)

                            # print(indexList)
                            theIndex = np.random.choice(indexList, replace=False, p=p)
                            theSkill, theDiff ,theVar= skillDiffList[theIndex]

                    # if i == 0:  # 由于一道题目与多个技能连接时权重都是一样的，所以没有必要采用带权采样
                    #     theIndex = np.random.choice(indexList, replace=False, p=None)
                    #     theSkill, theDiff = skillDiffList[theIndex]
                    # else:
                    #     skillTuple, diffTuple = zip(*skillDiffList)
                    #     probList = list(1 - self.getDist(diffTuple, theDiff) / D)
                    #     probList[skillTuple.index(theSkill)] *= decay_rate
                    #     probList = self.softMax(probList)
                    #     theIndex = np.random.choice(indexList, replace=False, p=probList)
                    #     theSkill, theDiff = skillDiffList[theIndex]
                    # one_walk.append(theSkill)

                    # pick the next question node
                # 从技能到题目
                    if np.random.rand() > 1 / len(self.skill_quesDiffList[theSkill]):
                        quesDiffList = list(self.skill_quesDiffList[theSkill])
                        indexList = list(range(0, len(quesDiffList)))
                        quesTuple, diffTuple = zip(*quesDiffList)
                        probList = list(1 - self.getDist(diffTuple, theDiff) / D)
                        delIndex = quesTuple.index(theQues)
                        del indexList[delIndex]
                        del probList[delIndex]
                        # probList[quesTuple.index(theQues)] *= decay_rate
                        probList = self.softMax(probList)
                        theIndex = np.random.choice(indexList, replace=False, p=probList)
                        theQues, theDiff = quesDiffList[theIndex]
                    one_walk.append(theQues)
                    # if len(plt_x)> 1 and len(plt_y)>1:
                    #     plt.scatter(plt_x, plt_y)
                    #     plt.xlabel('var')
                    #     plt.ylabel('probability')
                    #     plt.savefig('pic.png')
                    #     plt.show()
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))



    @staticmethod
    def softMax(vec):
        exp = np.exp(vec)
        sumExp = sum(exp)
        result = exp * 1.0 / sumExp
        return result

    @staticmethod
    def getDist(diffTuple, theDiff):
        return np.abs(np.array(diffTuple) - theDiff)

# 运行后可以得到游走序列
if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    data_set = "ASSIST09"
    # for data_set in ["ASSIST09"]:
    save_folder = "../pre_emb/%s/walks" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    D_dict = {'qk': 1}
    # decay_rate = 0.5
    mpg = MetaPathGenerator()
    mpg.read_data()
    for num_walks in [10]:
        for walk_length in [80]:
            t = time.time()
            out_file_name = os.path.join(save_folder, "walks_qkq_contDiff1.0_%d_%d.txt") % (num_walks, walk_length)
            outfile = open(out_file_name, 'w')
            # 得到游走序列
            mpg.generate_wgtRW_qkq(numWalks=num_walks, walkLength=walk_length, D=D_dict['qk'])
            outfile.close()
            print("time consuming: %d seconds" % (time.time() - t))


