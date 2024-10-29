import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        # 初始化问题-学生的邻接表
        self.ques_stuAnsList = dict()
        self.stu_quesAnsList = dict()

    def read_data(self, theAns):
        """
        the Ans [0,1]
        if ans=0时
        """
        # 读取学生-问题.csv的图数据（stu、ques_correct、time、attempt、first）
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))
        # 获取数据集中所有做错的题目
        stu_ques_df = stu_ques_df[stu_ques_df['correct'] == theAns]
        # index为行的序号 row是横变竖（7行2列）
        for index, row in stu_ques_df.iterrows():
            # 取得 学生和题目id
            stuID, quesID = int(row['stu']), int(row['ques'])
            # 取得 正确率 0
            answerState = row['correct']
            if quesID not in self.ques_stuAnsList:
                # 创建键值对 问题为key，值是[]
                self.ques_stuAnsList[quesID] = []
            # 将（学生，做题情况）添加到[]
            self.ques_stuAnsList[quesID].append((stuID, answerState))
            # 对学生同样操作
            if stuID not in self.stu_quesAnsList:
                self.stu_quesAnsList[stuID] = []
            self.stu_quesAnsList[stuID].append((quesID, answerState))

    def generate_wgtRW_quq(self, numWalks, walkLength):
        # 遍历学生-题目邻接表: 序号和题目
        for idx, ques in enumerate(self.ques_stuAnsList):
            if idx % 1000 == 0:
                # 每1000次的时候打印
                print(idx)
                # 选择游走次数
            for j in range(0, numWalks):
                # 初始化 题目、学生、答案
                theQues, theStu, theAns = ques, None, None
                # 一次游走 开头为题目
                one_walk = [theQues]
                # 按照游走长度开始游走
                for i in range(0, walkLength):
                    # 问题选学生
                    # pick the next student node
                    # 将邻接表中的一个问题[] 转为列表
                    stuAnsList = list(self.ques_stuAnsList[theQues])
                    # 邻居的长度序列
                    indexList = list(range(0, len(stuAnsList)))
                    # 随机选择index，无p
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    # 将选择的index对应学生和答案放入
                    theStu, theAns = stuAnsList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next question node
                    # 学生再选问题
                    # 将问题对学生的邻接表转化出来
                    quesAnsList = list(self.stu_quesAnsList[theStu])
                    # 邻居长度索引
                    indexList = list(range(0, len(quesAnsList)))
                    # 随机选择下一个index
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    # 根据索引选择 问题和 答案
                    theQues, theAns = quesAnsList[theIndex]
                    # 将该问题添加入 问题游走序列（全是问题）
                    one_walk.append(theQues)
                    # 写入文件
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


if __name__ == "__main__":
    # data_path = "C:/Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess3.0"
    # data_set = "ASSIST09"
    data_path = "E:/Study/SimKT/SimKT"
    # 设置数据集和保存的文件位置
    for data_set in ["ASSIST09"]:
        save_folder = "./%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        for num_walks in [10]:
            # 游走次数设置为10
            for walk_length in [80]:
                # 游走长度设置为80
                out_file_name = os.path.join(save_folder, "walks_quq_cw_%d_%d.txt") % (num_walks, walk_length)
                # 创建好文件后打开
                outfile = open(out_file_name, 'w')
                # ans先0后1
                for ans in [0, 1]:
                    # 记录时间
                    t = time.time()
                    # 初始化问题-学生和学生-问题的邻接表
                    mpg = MetaPathGenerator()
                    # 读取数据，生成两组字典{qi:[(stu,correct)]}
                    mpg.read_data(ans)
                    # 根据游走步数和长度写好文件
                    mpg.generate_wgtRW_quq(numWalks=num_walks, walkLength=walk_length)
                    print("time consuming: %d seconds" % (time.time() - t))
                outfile.close()
