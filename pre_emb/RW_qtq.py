import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        self.ques_tmpDiffList = dict()
        self.tmp_quesDiffList = dict()

    def read_data(self):
        df = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_template.csv"))
        for index, row in df.iterrows():
            quesID, tmpID, diffValue = int(row['ques']), int(row['template']), 1
            if quesID not in self.ques_tmpDiffList:
                self.ques_tmpDiffList[quesID] = []
            self.ques_tmpDiffList[quesID].append((tmpID, diffValue))
            if tmpID not in self.tmp_quesDiffList:
                self.tmp_quesDiffList[tmpID] = []
            self.tmp_quesDiffList[tmpID].append((quesID, diffValue))

    def generate_wgtRW_qtq(self, numWalks, walkLength):
        for idx, ques in enumerate(self.ques_tmpDiffList):
            if idx % 1000 == 0:
                print(idx)
            for j in range(0, numWalks):  # new sequence
                theQues, theTmp, theDiff = ques, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the next tmp node
                    tmpDiffList = list(self.ques_tmpDiffList[theQues])
                    indexList = list(range(0, len(tmpDiffList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theTmp, theDiff = tmpDiffList[theIndex]
                    # one_walk.append(theTmp)

                    # pick the next question node
                    quesDiffList = list(self.tmp_quesDiffList[theTmp])
                    indexList = list(range(0, len(quesDiffList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theQues, theDiff = quesDiffList[theIndex]
                    one_walk.append(theQues)
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


if __name__ == "__main__":
    t = time.time()
    data_path = "E:/Study/SimKT/SimKT/data"
    data_set = "ASSIST09"
    save_folder = "./%s/walks" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    mpg = MetaPathGenerator()
    mpg.read_data()
    for num_walks in [10]:
        for walk_length in [80]:
            outfile = open(os.path.join(save_folder, 'walks_qtq_noWgt_%d_%d.txt' % (num_walks, walk_length)), 'w')
            mpg.generate_wgtRW_qtq(numWalks=num_walks, walkLength=walk_length)
            outfile.close()
    print("time consuming: %d seconds" % (time.time() - t))


