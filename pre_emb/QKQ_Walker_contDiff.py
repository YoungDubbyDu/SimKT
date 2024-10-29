import numpy as np
import pandas as pd
import time
import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from pre_emb.Wgt_Walker import WeightedWalker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def to_tensor(x):
    return torch.tensor(x).to(device)


class WeightedWalker:
    # 根据边权重，动态地确定节点的采样概率
    # 每调用一次get_next，完成从A类节点（头节点）到B类节点（尾节点）的一步游走，v_num是A类节点的总数，edges是A类节点指向B类节点的边的列表
    def __init__(self, v_num, edges):
        self.v_num = v_num
        ngbrs = [[] for i in range(v_num)]
        weights = [[] for i in range(v_num)] # 问题难度v
        for u, v, w in edges:
            u = int(u)
            v = int(v)
            ngbrs[u].append(v)
            weights[u].append(w)

        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)  # 填充后里邻居
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True) # 填充后问题难度
        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]).unsqueeze(1) # 邻居个数
        self.max_ngbr_num = torch.max(self.ngbr_nums).item() # 最大邻居数

        skill_var = pd.read_csv(os.path.join(data_path, data_set, "graph", "skill_var.csv"))
        var = skill_var.iloc[:, 1]
        var = to_tensor(var).reshape(-1, 1)
        self.node_attr_mat = 1-var
        self.D = 1
        self.the_nodes = None


    def get_prob(self, weights, ngbrs, nums, the_weights,flag):
        if the_weights is not None:
            # 难度之差越小,则概率越大
            if flag ==1:  #K-Q
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights-the_weights), to_tensor([self.D]))
                # prob = torch.ones(weights.size()).to(device)
            else:  #Q-K
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))
                # prob = torch.ones(weights.size()).to(device)

                prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # print(prob2,prob2.shape)
                prob *= prob2
        else:
            prob = torch.ones(weights.size()).to(device)
            # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
            # prob*=prob2

        # 防止回溯，即防止重复采样同一节点
        if self.the_nodes is not None:
            mask1 = (ngbrs == self.the_nodes)
            mask2 = torch.rand(nums.size()).to(device) > torch.true_divide(to_tensor([1]), nums)
            prob = prob.masked_fill(mask1 & mask2, -1e32)

        # 令填充的部分采样概率为0
        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0) # [[ 0,1,2,3]]

        mask = x >= nums

        prob = prob.masked_fill(mask, -1e32)

        prob = F.softmax(prob, dim=1)
        print(prob,prob.shape)
        return prob  # [v_num * walk_num, max_ngbr_num]


    def get_next(self, v, the_weights,flag):
        # v是[0,0,0,01,1,1,1]
        # 根据节点得到 对应的权重、邻居数、邻居
        expand_pad_weights = self.weights[v]
        expand_ngbr_nums = self.ngbr_nums[v]
        expand_pad_ngbrs = self.ngbrs[v]

        # print('expand_pad_weights:',expand_pad_weights)
        # print('expand_ngbr_nums:',expand_ngbr_nums)
        # print('expand_pad_ngbrs:',expand_pad_ngbrs)
        #  返回每个邻居之间的转移概率
        expand_pad_prob = self.get_prob(expand_pad_weights, expand_pad_ngbrs, expand_ngbr_nums, the_weights,flag)
        # print('expand_pad_prob：',expand_pad_prob )
        next_index = torch.multinomial(expand_pad_prob, num_samples=1)  # 按概率采样每行的索引
        # print('next_index',next_index)
        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index) # 根据每行的索引对应到节点
        # print('nodes:',self.the_nodes)
        next_v = self.the_nodes.flatten() # 平铺为新的一排 k1,k1,k1
        # print('next_v',next_v)
        the_weights = torch.gather(expand_pad_weights, 1, next_index) # 根据采样索引读取的对应的题目难度
        # print('the_weights:',the_weights)
        return next_v, the_weights

class QKQ_Walker:
    def __init__(self):
        # self.D = 1  # D notates the biggest distance between diff
        self.qk_edge_list = []
        self.kq_edge_list = []
        self.num_ques = None
        self.num_skill = None

        self.read_data()

    def read_data(self):
        ques_skill_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_skill.csv"))
        with open(os.path.join(data_path, data_set, "attribute", "quesID2diffValue_dict.txt")) as f:
            ques2diff_dict = eval(f.read())

        self.num_ques = len(set(ques_skill_df['ques']))
        self.num_skill = len(set(ques_skill_df['skill']))

        for index, row in ques_skill_df.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            diffValue = ques2diff_dict[quesID]

            self.qk_edge_list.append((quesID, skillID, diffValue))
            self.kq_edge_list.append((skillID, quesID, diffValue))


    def create_paths(self, walk_num=10, walk_len=80):
        # 构建QK_Walker和KQ_walker
        QK_Walker = WeightedWalker(v_num=self.num_ques, edges=self.qk_edge_list)
        KQ_Walker = WeightedWalker(v_num=self.num_skill, edges=self.kq_edge_list)
        # 0-10 来10行在转置降维 一行num_ques*walk_num列
        next_q = torch.arange(self.num_ques).to(device).repeat(walk_num, 1).T.flatten()
        print(('next_q:',next_q))
        paths = [next_q]
        KQ_Walker.the_nodes = next_q.unsqueeze(1)
        the_weights = None
        for i in range(1, walk_len):
            print("%dth hop" % i)
            next_k, the_weights = QK_Walker.get_next(next_q, the_weights,flag=0)
            next_q, the_weights = KQ_Walker.get_next(next_k, the_weights,flag=1)
            paths.append(next_q)

        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1)
        paths = paths.view(-1, walk_len)
        return paths


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    for data_set in ["ASSIST09"]:
        save_folder = "../pre_emb/%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        D_dict = {'qk': 1}
        # decay_rate = 0.5
        walker = QKQ_Walker()
        for num_walks in [10]:
            for walk_length in [80]:
                t = time.time()
                save_file = "walks_qkq_contDiff_pos_%d_%d.txt" % (num_walks, walk_length)
                with open(os.path.join(save_folder, save_file), 'w') as f:
                    paths1 = walker.create_paths(num_walks, walk_length) # 游走路径
                    for path1 in paths1.cpu().detach().tolist():
                        f.write(','.join([str(e) for e in path1]) + '\n')
                print("time consuming: %d seconds" % (time.time() - t))
