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
            answerState = 1
            if quesID not in self.ques_stuAnsList:
                self.ques_stuAnsList[quesID] = []
            self.ques_stuAnsList[quesID].append((stuID, answerState))
            if stuID not in self.stu_quesAnsList:
                self.stu_quesAnsList[stuID] = []
            self.stu_quesAnsList[stuID].append((quesID, answerState))

    def generate_wgtRW_quq(self, numWalks, walkLength):
        for idx, ques in enumerate(self.ques_stuAnsList):
            if idx % 100 == 0:
                print(idx)
            for j in range(0, numWalks):
                theQues, theStu, theAns = ques, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the next student node
                    stuAnsList = list(self.ques_stuAnsList[theQues])
                    indexList = list(range(0, len(stuAnsList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theStu, theAns = stuAnsList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next question node
                    quesAnsList = list(self.stu_quesAnsList[theStu])
                    indexList = list(range(0, len(quesAnsList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theQues, theAns = quesAnsList[theIndex]
                    one_walk.append(theQues)
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    # data_path = "E:/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess2.0"
    # data_set = "ASSIST09"
    for data_set in ["EdNet"]:
        save_folder = "./%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        for num_walks in [10]:
            for walk_length in [80]:
                out_file_name = os.path.join(save_folder, "walks_quq_noWgt_%d_%d.txt") % (num_walks, walk_length)
                outfile = open(out_file_name, 'w')
                t = time.time()
                mpg = MetaPathGenerator()
                mpg.read_data()
                mpg.generate_wgtRW_quq(numWalks=num_walks, walkLength=walk_length)
                print("time consuming: %d seconds" % (time.time() - t))
                outfile.close()
