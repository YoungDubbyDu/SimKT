import numpy as np
import pandas as pd
import time
import os
import torch
# from pre_emb.Wgt_Walker import WeightedWalker
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    return torch.tensor(x).to(device)


class WeightedWalker:
    # 根据边权重，动态地确定节点的采样概率
    # 每调用一次get_next，完成从A类节点（头节点）到B类节点（尾节点）的一步游走，v_num是A类节点的总数，edges是A类节点指向B类节点的边的列表
    def __init__(self, v_num, edges):

        self.v_num = v_num

        ngbrs = [[] for i in range(v_num)]
        weights = [[] for i in range(v_num)]
        for u, v, w in edges:
            u = int(u)
            v = int(v)
            ngbrs[u].append(v)
            weights[u].append(w)


        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True)
        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]).unsqueeze(1)
        self.max_ngbr_num = torch.max(self.ngbr_nums).item()

        # stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ability.csv"))  # 考虑离散能力值
        stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_abi_w.csv"))  # 考虑连续能力值
        ability = stu_ability.iloc[:, 1]
        abi_average = np.average(ability)
        ability = torch.tensor(ability).reshape(-1, 1)

        self.node_attr_mat = torch.sigmoid(-torch.abs(ability - abi_average)).to(device)  # 正向的
        # self.node_attr_mat = torch.sigmoid(torch.abs(ability - abi_average)).to(device)  # 反向验证

        ques_disc = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_discvalue.csv")) # 考虑题目区分度
        disc = ques_disc.iloc[:,1]
        disc = torch.tensor(disc).reshape(-1, 1)
        # self.node_attr_mat1 = torch.sigmoid(disc).to(device)    # 正向的
        # self.node_attr_mat1 = torch.sigmoid(-disc).to(device) # 反例证明
        self.D = 1
        self.the_nodes = None

    def get_prob(self, weights, ngbrs, nums, the_weights,flag):
        if flag == 0:
            # Q-U,考虑学生能力值
            if the_weights is not None:
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights-the_weights), to_tensor([self.D]))
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # prob *= prob2
            else:
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # prob *= prob2
        else:
            # U-Q 考虑题目区分度
            if the_weights is not None:
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))
                # 考虑题目区分度
                prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                prob *= prob2

            else:
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # 考虑题目区分度
                prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                prob *= prob2

        # 防止回溯，即防止重复采样同一节点
        if self.the_nodes is not None:
            mask1 = (ngbrs == self.the_nodes)
            mask2 = torch.rand(nums.size()).to(device) > torch.true_divide(to_tensor([1]), nums)
            prob = prob.masked_fill(mask1 & mask2, 1e-32)  # 赋予一个接近于0的非零值，避免特殊情况下计算prob时分母为零

        # 令填充的部分采样概率为0
        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0)
        mask = x >= nums
        prob = prob.masked_fill(mask, 0)

        prob = torch.true_divide(prob, torch.sum(prob, dim=1, keepdim=True))
        return prob  # [v_num * walk_num, max_ngbr_num]



    def get_next(self, v, the_weights,flag):
        expand_pad_weights = self.weights[v]
        expand_ngbr_nums = self.ngbr_nums[v]
        expand_pad_ngbrs = self.ngbrs[v]

        expand_pad_prob = self.get_prob(expand_pad_weights, expand_pad_ngbrs, expand_ngbr_nums, the_weights,flag)

        next_index = torch.multinomial(expand_pad_prob, num_samples=1)  # 按概率采样

        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index)
        next_v = self.the_nodes.squeeze()

        the_weights = torch.gather(expand_pad_weights, 1, next_index)
        return next_v, the_weights


class QUQ_Walker:
    def __init__(self):
        # self.D = 1  # D notates the biggest distance between answer
        self.qu_edge_list = []
        self.uq_edge_list = []
        self.num_stu = None
        self.num_ques = None

        self.read_data()

    def read_data(self):
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))
        self.num_stu = len(set(stu_ques_df['stu']))
        self.num_ques = len(set(stu_ques_df['ques']))
        print("num of stu: %d" % self.num_stu)
        print("num of ques: %d" % self.num_ques)

        for index, row in stu_ques_df.iterrows():
            stuID, quesID = int(row['stu']), int(row['ques'])
            answerState = row['correct']
            self.qu_edge_list.append((quesID, stuID, answerState))
            self.uq_edge_list.append((stuID, quesID, answerState))

    def create_paths(self, walk_num=10, walk_len=80):
        # 构建QU_Walker和UQ_walker
        QU_Walker = WeightedWalker(v_num=self.num_ques, edges=self.qu_edge_list)
        UQ_Walker = WeightedWalker(v_num=self.num_stu, edges=self.uq_edge_list)

        next_q = torch.arange(self.num_ques).to(device).repeat(walk_num, 1).T.flatten()
        paths = [next_q]
        UQ_Walker.the_nodes = next_q.unsqueeze(1)
        the_weights = None
        for i in range(1, walk_len):
            print("%dth hop" % i)
            next_u, the_weights = QU_Walker.get_next(next_q, the_weights,0)
            next_q, the_weights = UQ_Walker.get_next(next_u, the_weights,1)
            paths.append(next_q)

        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1)
        paths = paths.view(-1, walk_len)
        return paths


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    for data_set in ["EdNet"]:
        save_folder ="../pre_emb/%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        D_dict = {'qu': 1}
        # decay_rate = 0.5
        walker = QUQ_Walker()
        for num_walks in [10]:
            for walk_length in [80]:
                t = time.time()
                save_file = "walks_quq_wgt_pos6.2_%d_%d.txt" % (num_walks, walk_length)
                with open(os.path.join(save_folder, save_file), 'w') as f:
                    paths1 = walker.create_paths(num_walks, walk_length)
                    for path1 in paths1.cpu().detach().tolist():
                        f.write(','.join([str(e) for e in path1]) + '\n')
                print("time consuming: %d seconds" % (time.time() - t))
