import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        self.ques_stuAnsList = dict()
        self.stu_quesAnsList = dict()

    def read_data(self):
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))
        for index, row in stu_ques_df.iterrows():
            stuID, quesID = int(row['stu']), int(row['ques'])
            # 读取当前答题正确情况
            answerState = row['correct']
            if quesID not in self.ques_stuAnsList:
                self.ques_stuAnsList[quesID] = []
                # 填入邻居的学生id和答题情况
            self.ques_stuAnsList[quesID].append((stuID, answerState))
            if stuID not in self.stu_quesAnsList:
                self.stu_quesAnsList[stuID] = []
            self.stu_quesAnsList[stuID].append((quesID, answerState))

    def generate_wgtRW_quq(self, numWalks, walkLength, D=1.0):
        # 带权了1.0的难度
        # 遍历序号和题目
        for idx, ques in enumerate(self.ques_stuAnsList):
            if idx % 1000 == 0:
                print(idx)
                # 选择游走次数
            for j in range(0, numWalks):
                # 初始化问题、学生、答题
                theQues, theStu, theAns = ques, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the next student node
                    # 取出该问题所有邻居学生信息
                    stuAnsList = list(self.ques_stuAnsList[theQues])
                    indexList = list(range(0, len(stuAnsList)))
                    # 根据问题选择学生
                    if i == 0:
                        # 第一个随机选择一个学生进行（需要修改，优秀选择能力中等的）
                        theIndex = np.random.choice(indexList, replace=False, p=None)
                        # 取得第一个学生和答案
                        theStu, theAns = stuAnsList[theIndex]
                    else: # 此时已经两个问题[1,32]
                        # 随机取0-1 若大于1/邻居长度就继续
                        if np.random.rand() > 1 / len(stuAnsList):
                            # 分别取得学生元组和答案元组
                            stuTuple, ansTuple = zip(*stuAnsList)
                            # 转移概率列表 相同方式计算 相同答案 概率为1
                            probList = list(1 - self.getDist(ansTuple, theAns) / D)
                            delIndex = stuTuple.index(theStu)
                            del indexList[delIndex] # 删除学生索引
                            del probList[delIndex]
                            if np.sum(probList) != 0:
                                probList = self.normalize(probList)
                                # 随机选择到下一个index
                                theIndex = np.random.choice(indexList, replace=False, p=probList)
                                # 根据索引，选择到下一个学生
                                theStu, theAns = stuAnsList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next question node
                    # 根据学生选择问题
                    if np.random.rand() > 1 / len(self.stu_quesAnsList[theStu]):
                        # 该学生对应的（问题，答案）列表
                        quesAnsList = list(self.stu_quesAnsList[theStu])
                        # 获取索引
                        indexList = list(range(0, len(quesAnsList)))
                        # 分别得到问题和答案元组
                        quesTuple, ansTuple = zip(*quesAnsList)
                        # 概率列表：概率公式为（1-第一个学生答案和答案元组中的距离）
                        # 思想： 答案相同为0，转移概率=1，答案不同为1，转移概率为0
                        probList = list(1 - self.getDist(ansTuple, theAns) / D)
                        # 要删除的索引（上一个问题）
                        delIndex = quesTuple.index(theQues)
                        del indexList[delIndex] #删除索引列表里的上一个题目索引
                        del probList[delIndex] # 删除它的概率
                        if np.sum(probList) != 0: # 如果不是全部为0，即答案全不同
                            # 分配概率
                            probList = self.normalize(probList)
                            # replace是否可以取相同元素，p规定每个元素的概率
                            theIndex = np.random.choice(indexList, replace=False, p=probList)
                            # 根据索引取得问题和答案
                            theQues, theAns = quesAnsList[theIndex]
                    # 添加到问题序列
                    one_walk.append(theQues)

                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))

    @staticmethod
    def normalize(vec):
        # 给概率为1 的平均分配自己的概率
        sumValue = np.sum(vec)
        result = np.array(vec) * 1.0 / sumValue
        return result

    @staticmethod
    def getDist(diffTuple, theDiff):
        return np.abs(np.array(diffTuple) - theDiff)


if __name__ == "__main__":
    # data_path = "C:/Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess2.0"
    data_path = "E:/Study/SimKT/SimKT/data"
    data_set = "ASSIST09"
    # for data_set in ["ASSIST09", "EdNet"]:
    save_folder = "%s/walks" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for num_walks in [10]:
        for walk_length in [80]:
            out_file_name = os.path.join(save_folder, "walks_quq_wgt_%d_%d.txt") % (num_walks, walk_length)
            outfile = open(out_file_name, 'w')
            t = time.time()
            mpg = MetaPathGenerator()
            # 读取数据生成了两组字典
            mpg.read_data()
            mpg.generate_wgtRW_quq(numWalks=num_walks, walkLength=walk_length)
            print("time consuming: %d seconds" % (time.time() - t))
            outfile.close()
