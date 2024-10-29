import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        self.ques_stuAnsList = dict()
        self.stu_quesAnsList = dict()
        self.stu_clusterList = dict()
        self.cluster_stuList = dict()

    def read_data(self, numCluster):
        df_stu_cluster = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_cluster_%d.csv" % numCluster))
        df_stu_ques = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))

        stu_set = set(df_stu_ques['stu']) & set(df_stu_cluster['stu'])
        df_stu_ques = df_stu_ques[df_stu_ques['stu'].isin(stu_set)]
        df_stu_cluster = df_stu_cluster[df_stu_cluster['stu'].isin(stu_set)]

        for index, row in df_stu_cluster.iterrows():
            stuID, clusterID, weight = int(row['stu']), int(row['cluster']), 1
            if clusterID not in self.cluster_stuList:
                self.cluster_stuList[clusterID] = []
            self.cluster_stuList[clusterID].append((stuID, weight))
            if stuID not in self.stu_clusterList:
                self.stu_clusterList[stuID] = []
            self.stu_clusterList[stuID].append((clusterID, weight))

        for index, row in df_stu_ques.iterrows():
            stuID, quesID, answerState = int(row['stu']), int(row['ques']), row['correct']
            if quesID not in self.ques_stuAnsList:
                self.ques_stuAnsList[quesID] = []
            self.ques_stuAnsList[quesID].append((stuID, answerState))
            if stuID not in self.stu_quesAnsList:
                self.stu_quesAnsList[stuID] = []
            self.stu_quesAnsList[stuID].append((quesID, answerState))

    def generate_wgtRW_qucuq(self, numWalks, walkLength):
        for idx, ques in enumerate(self.ques_stuAnsList):
            if idx % 1000 == 0:
                print(idx)
            for j in range(0, numWalks):
                theQues, theStu, theCluster, theAns, theWgt = ques, None, None, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the first next student node
                    stuAnsList = list(self.ques_stuAnsList[theQues])
                    indexList = list(range(0, len(stuAnsList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theStu, theAns = stuAnsList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next cluster node
                    clusterList = list(self.stu_clusterList[theStu])
                    indexList = list(range(0, len(clusterList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theCluster, theWgt = clusterList[theIndex]
                    # one_walk.append(theCluster)

                    # pick the second next student node
                    stuList = list(self.cluster_stuList[theCluster])
                    indexList = list(range(0, len(stuList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theStu, theWgt = stuList[theIndex]
                    # one_walk.append(theStu)

                    # pick the next question node
                    quesAnsList = list(self.stu_quesAnsList[theStu])
                    indexList = list(range(0, len(quesAnsList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theQues, theAns = quesAnsList[theIndex]
                    one_walk.append(theQues)
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


def softMax(vec):
    exp = np.exp(vec)
    sumExp = sum(exp)
    result = exp * 1.0 / sumExp
    return result


if __name__ == "__main__":
    data_path = "C:/Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess2.0"
    # data_set = "ASSIST09"
    for data_set in ["EdNet"]:
        save_folder = "./%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        for num_walks in [10]:
            for walk_length in [80]:
                for num_cluster in [120]:
                    out_file_name = os.path.join(save_folder, 'walks_qucuq_noWgt_%d_%d_%d.txt' % (
                        num_cluster, num_walks, walk_length))
                    outfile = open(out_file_name, 'w')
                    t = time.time()
                    mpg = MetaPathGenerator()
                    mpg.read_data(num_cluster)
                    mpg.generate_wgtRW_qucuq(numWalks=num_walks, walkLength=walk_length)
                    print("time consuming: %d seconds" % (time.time() - t))
                    outfile.close()
