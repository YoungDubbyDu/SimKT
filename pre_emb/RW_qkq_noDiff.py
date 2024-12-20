import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        self.ques_skillDiffList = dict()
        self.skill_quesDiffList = dict()

    def read_data(self):
        ques_skill_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_skill.csv"))
        for index, row in ques_skill_df.iterrows():
            quesID, skillID, diffValue = int(row['ques']), int(row['skill']), 1
            if quesID not in self.ques_skillDiffList:
                self.ques_skillDiffList[quesID] = []
            self.ques_skillDiffList[quesID].append((skillID, diffValue))
            if skillID not in self.skill_quesDiffList:
                self.skill_quesDiffList[skillID] = []
            self.skill_quesDiffList[skillID].append((quesID, diffValue))

    def generate_wgtRW_qkq(self, numWalks, walkLength):
        for idx, ques in enumerate(self.ques_skillDiffList):
            if idx % 1000 == 0:
                print(idx)
            for j in range(0, numWalks):  # new sequence
                theQues, theSkill, theDiff = ques, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the next skill node
                    skillDiffList = list(self.ques_skillDiffList[theQues])
                    indexList = list(range(0, len(skillDiffList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theSkill, theDiff = skillDiffList[theIndex]
                    # one_walk.append(theSkill)

                    # pick the next question node
                    quesDiffList = list(self.skill_quesDiffList[theSkill])
                    indexList = list(range(0, len(quesDiffList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theQues, theDiff = quesDiffList[theIndex]
                    one_walk.append(theQues)
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


if __name__ == "__main__":
    t = time.time()
    data_path = "C:/Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess3.0"
    # data_set = "ASSIST09"
    for data_set in ["ASSIST12"]:
        save_folder = "./%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        mpg = MetaPathGenerator()
        mpg.read_data()
        for num_walks in [10]:
            for walk_length in [80]:
                outfile = open(os.path.join(save_folder, "walks_qkq_noDiff_%d_%d.txt") % (num_walks, walk_length), 'w')
                mpg.generate_wgtRW_qkq(numWalks=num_walks, walkLength=walk_length)
                outfile.close()
        print("time consuming: %d seconds" % (time.time() - t))
