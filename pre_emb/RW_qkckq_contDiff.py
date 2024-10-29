import numpy as np
import pandas as pd
import time
import os


class MetaPathGenerator:
    def __init__(self):
        self.skill_clusterRatioList = dict()
        self.cluster_skillRatioList = dict()
        self.ques_skillDiffList = dict()
        self.skill_quesDiffList = dict()

    def read_data(self, num_cluster):
        df_skill_cluster = pd.read_csv(os.path.join(data_path, data_set, "graph", "skill_cluster_%d.csv" % num_cluster))
        df_ques_skill = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_skill.csv"))
        skill_set = set(df_ques_skill['skill']) & set(df_skill_cluster['skill'])
        df_ques_skill = df_ques_skill[df_ques_skill['skill'].isin(skill_set)]
        df_skill_cluster = df_skill_cluster[df_skill_cluster['skill'].isin(skill_set)]

        with open(os.path.join(data_path, data_set, "attribute", "quesID2diffValue_dict.txt")) as f:
            ques2diff_dict = eval(f.read())

        for index, row in df_skill_cluster.iterrows():
            skillID, clusterID, ratio = int(row['skill']), int(row['cluster']), 1
            if skillID not in self.skill_clusterRatioList:
                self.skill_clusterRatioList[skillID] = []
            self.skill_clusterRatioList[skillID].append((clusterID, ratio))
            if clusterID not in self.cluster_skillRatioList:
                self.cluster_skillRatioList[clusterID] = []
            self.cluster_skillRatioList[clusterID].append((skillID, ratio))

        for index, row in df_ques_skill.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            diffValue = ques2diff_dict[quesID]
            if quesID not in self.ques_skillDiffList:
                self.ques_skillDiffList[quesID] = []
            self.ques_skillDiffList[quesID].append((skillID, diffValue))
            if skillID not in self.skill_quesDiffList:
                self.skill_quesDiffList[skillID] = []
            self.skill_quesDiffList[skillID].append((quesID, diffValue))

    def generate_wgtRW_qkckq(self, numWalks, walkLength, D_dict):
        for idx, ques in enumerate(self.ques_skillDiffList):
            if idx % 1000 == 0:
                print(idx)
            for j in range(0, numWalks):
                theQues, theSkill, theCluster, theDiff, theRatio = ques, None, None, None, None
                one_walk = [theQues]
                for i in range(0, walkLength):
                    # pick the first next skill node
                    skillDiffList = list(self.ques_skillDiffList[theQues])
                    indexList = list(range(0, len(skillDiffList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theSkill, theDiff = skillDiffList[theIndex]
                    # if i == 0:  # 题目与其相邻技能之间的权重都一样，没有必要使用权重采样
                    #     theIndex = np.random.choice(indexList, replace=False, p=None)
                    #     theSkill, theDiff = skillDiffList[theIndex]
                    # else:
                    #     skillTuple, diffTuple = zip(*skillDiffList)
                    #     probList = list(1 - np.abs(np.array(diffTuple) - theDiff) / D_dict['qk'])
                    #     # probList[skillTuple.index(theSkill)] *= decay_rate
                    #     probList = softMax(probList)
                    #     theIndex = np.random.choice(indexList, replace=False, p=probList)
                    #     theSkill, theDiff = skillDiffList[theIndex]
                    # one_walk.append(theSkill)

                    # pick the next cluster node
                    clusterRatioList = list(self.skill_clusterRatioList[theSkill])
                    indexList = list(range(0, len(clusterRatioList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theCluster, theRatio = clusterRatioList[theIndex]
                    # one_walk.append(theCluster)

                    # pick the second next skill node
                    skillRatioList = list(self.cluster_skillRatioList[theCluster])
                    indexList = list(range(0, len(skillRatioList)))
                    theIndex = np.random.choice(indexList, replace=False, p=None)
                    theSkill, theRatio = skillRatioList[theIndex]
                    # one_walk.append(theSkill)

                    # pick the next question node
                    quesDiffList = list(self.skill_quesDiffList[theSkill])
                    indexList = list(range(0, len(quesDiffList)))
                    quesTuple, diffTuple = zip(*quesDiffList)
                    probList = list(1 - np.abs(np.array(diffTuple) - theDiff) / D_dict['qk'])
                    probList = softMax(probList)
                    theIndex = np.random.choice(indexList, replace=False, p=probList)
                    theQues, theDiff = quesDiffList[theIndex]
                    one_walk.append(theQues)
                outfile.write("%s\n" % ','.join([str(node) for node in one_walk]))


def softMax(vec):
    exp = np.exp(vec)
    sumExp = sum(exp)
    result = exp * 1.0 / sumExp
    return result


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    # data_set = "ASSIST09"
    for data_set in ["ASSIST09"]:
        save_folder = "./%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        Dw_dict = {'qk': 1, 'kc': 1}
        # decay_rate = 0.5
        for numCluster in [64]:
            for num_walks in [10]:
                for walk_length in [80]:
                    t = time.time()
                    out_file_name = os.path.join(save_folder, 'walks_qkckq_contDiff_%d_%d_%d.txt' % (
                        numCluster, num_walks, walk_length))
                    outfile = open(out_file_name, 'w')
                    mpg = MetaPathGenerator()
                    mpg.read_data(numCluster)
                    mpg.generate_wgtRW_qkckq(numWalks=num_walks, walkLength=walk_length, D_dict=Dw_dict)
                    print("time consuming: %d seconds" % (time.time() - t))
                    outfile.close()
