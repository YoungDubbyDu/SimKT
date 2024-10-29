import numpy as np
import pandas as pd
import time
import os
import torch
# from pre_emb.Wgt_Walker import WeightedWalker
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

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
        for u,v,w in edges:

            u = int(u)
            v = int(v)
            ngbrs[u].append(v)
            weights[u].append(w)
        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True)

        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]).unsqueeze(1)
        self.max_ngbr_num = torch.max(self.ngbr_nums).item()
        print("max neighbor num", self.max_ngbr_num)

        stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_abi_w.csv"))  # 考虑连续带权能力值
        ability = stu_ability.iloc[:, 1]
        abi_average = np.average(ability)
        ability = torch.tensor(ability).reshape(-1, 1).to(device)
        self.node_attr_mat = torch.sigmoid(-torch.abs(ability - abi_average)).to(device) # 正向的
        # self.node_attr_mat = ability
        # self.node_attr_mat = torch.sigmoid(torch.abs(ability - abi_average)).to(device)  # 反向验证


        ques_disc = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_discvalue.csv"))  # 考虑题目区分度
        disc = ques_disc.iloc[:, 1]
        disc = torch.tensor(disc).reshape(-1, 1)
        self.node_attr_mat1 = torch.sigmoid(disc).to(device)    # 正向的
        # self.node_attr_mat1 = torch.sigmoid(-disc).to(device) # 反例证明
        self.D = 1
        self.the_nodes = None

    def get_prob(self, weights, ngbrs, nums, the_weights,flag):
        # print('ngbrs:',ngbrs,'ngbrsshape:',ngbrs.shape)
        # print('nums:',nums,'numshape:',nums.shape)
        if flag == 0:
            # Q-U,考虑学生能力值
            if the_weights is not None:
                # print('上一个权重',the_weights)
                # print('这一次权重',weights)
                # prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))+1e-32
                # print('prob0',prob)
                prob = torch.ones(weights.size()).to(device)
                # ability = self.node_attr_mat # 此时为(3300,1)的能力值
                # # print('ability',ability,ability.shape)
                # ability = F.embedding(ngbrs, ability).squeeze(2)
                # # print('ability', ability, ability.shape)
                # w = -the_weights*ability+(1-the_weights)*ability
                # # print('w',w,w.shape)
                # prob2 = torch.sigmoid(10*w)
                # print('prob2',prob2,prob2.shape)
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # prob *= prob2
                # prob=prob2
            else:
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # print('prob0', prob)
                #
                # print(self.node_attr_mat1)
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # print(ngbrs)
                # print('prob2:', prob2, 'probshape', prob2.shape)
                # prob *= prob2
        else:
            # U-Q 考虑题目区分度
            if the_weights is not None:
                # prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))+1e-32
                # 考虑题目区分度
                prob = torch.ones(weights.size()).to(device)
                # prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                # # print(prob2,prob2.shape)
                # # prob *= prob2
                # # print(prob)
                # prob = prob2

            else:
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # 考虑题目区分度
                # prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                # prob *= prob2

        # 防止回溯，即防止重复采样同一节点
        if self.the_nodes is not None:
            mask1 = (ngbrs == self.the_nodes)
            mask2 = torch.rand(nums.size()).to(device) > torch.true_divide(to_tensor([1]), nums)
            prob = prob.masked_fill(mask1 & mask2, 1e-32)  # 赋予一个接近于0的非零值，避免特殊情况下计算prob时分母为零

        # 令填充的部分采样概率为0
        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0)
        mask = x >= nums
        prob = prob.masked_fill(mask, 0)

        prob = torch.true_divide(prob, torch.sum(prob, dim=1, keepdim=True)) # 归一化可以去掉
        # print('prob',prob,'prob的shape:',prob.shape)
        # print(prob[prob<0])
        return prob  # [v_num * walk_num, max_ngbr_num]

    def get_next(self, v, the_weights,flag):
        expand_pad_weights = self.weights[v]
        expand_ngbr_nums = self.ngbr_nums[v]
        expand_pad_ngbrs = self.ngbrs[v]


        expand_pad_prob = self.get_prob(expand_pad_weights, expand_pad_ngbrs, expand_ngbr_nums, the_weights,flag)
        # print('expand_pad_prob',expand_pad_prob,expand_pad_prob.shape)
        # # print('expand_pad_ngbrs',expand_pad_ngbrs,expand_pad_ngbrs.shape)
        #
        # _, sort_index = expand_pad_prob.sort(descending=True) # 获取排序素索引
        # rank0 = (1 + sort_index) / (torch.count_nonzero(expand_pad_prob > 0.001, dim=1).reshape(-1, 1))
        # rank0=rank0.cpu().reshape(-1)
        # rank0 = rank0[rank0<=1]
        # rank0 = rank0[rank0>0.001]
        # print('rank0',rank0)
        # print('所有邻居的权重',expand_pad_prob,expand_pad_prob.shape)
        next_index = torch.multinomial(expand_pad_prob, num_samples=1,replacement=True)  # 按概率采样
        # print('权重采样索引结果',next_index,next_index.shape)
        # _, sort_index = expand_pad_prob.sort(descending=True) # 获取排序素索引
    # print(sort_index)
    #     _,col_rank = torch.where(sort_index==next_index)
    #     col_rank=col_rank.reshape(-1,1)
    #     rank0 = (1+col_rank)/(torch.count_nonzero(expand_pad_prob > 0.001,dim=1).reshape(-1, 1))
        # rank = str(1 + rank) + '/' + str(len(expand_pad_prob[0][expand_pad_prob[0]>0.01]))
        # rank0=rank0.cpu().reshape(-1)
        # rank0 = rank0[rank0<=1]
        # rank0 = rank0[rank0>0.001]
        # print('rank0',rank0)
        # if flag == 0:
        #     global rank_list
        #     rank_list =torch.cat((rank_list,rank0),0)

            # rank_list1.append(rank)
        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index)
        # print('根据索引取值',self.the_nodes,self.the_nodes.shape)
        # print(self.the_nodes in expand_pad_ngbrs)
        # plt.scatter(expand_pad_ngbrs.cpu()[0], expand_pad_prob.cpu()[0])
        # plt.vlines(self.the_nodes.cpu()[0], ymin=0, ymax=1, colors='r')
        # plt.show()
        next_v = self.the_nodes.squeeze()
        # print('next_v',next_v,next_v.shape)

        the_weights = torch.gather(expand_pad_weights, 1, next_index)
        return next_v, the_weights


class QUQ_Walker:
    def __init__(self):
        def sample(df, col, max_num=600): # 变小
            ui_df = df.groupby([col], as_index=False)
            df_list = []
            for ui in ui_df:
                tmp_user, tmp_inter = ui[0], ui[1]
                if len(tmp_inter) > max_num:
                    tmp_inter = tmp_inter.sample(n=max_num)
                df_list.append(tmp_inter)
            return pd.concat(df_list)

        # self.D = 1  # D notates the biggest distance between answer
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"),
                                  usecols=['stu', 'ques', 'correct'])

        self.num_stu = max(set(stu_ques_df['stu'])) + 1
        self.num_ques = max(set(stu_ques_df['ques'])) + 1
        print("num of stu: %d" % self.num_stu)
        print("num of ques: %d" % self.num_ques)

        # 采样邻居，避免显存不足
        uq_df = sample(stu_ques_df[['stu', 'ques', 'correct']], 'stu')
        qu_df = sample(stu_ques_df[['ques', 'stu', 'correct']], 'ques')

        self.qu_edge_list = qu_df.values.tolist()
        self.uq_edge_list = uq_df.values.tolist()

        print('read data done')

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
    rank_list=torch.tensor([])
    # rank_list1=[]
    for data_set in ["ASSIST12"]:
        save_folder = "../pre_emb/%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        D_dict = {'qu': 1}
        # decay_rate = 0.5
        walker = QUQ_Walker()

        for num_walks in [10]:
            for walk_length in [80]:
                t = time.time()
                save_file = "walks_quq_nowgt_pos_%d_%d.txt" % (num_walks, walk_length)

                print("start saving")
                with open(os.path.join(save_folder, save_file), 'w') as f:
                    for i in range(1):  # 循环10次叠加
                        ps = walker.create_paths(num_walks, walk_length)
                        # print('ps', ps, 'shape', ps.shape)
                        for p in ps.cpu().detach().tolist():
                            f.write(','.join([str(e) for e in p]) + '\n')
                print("time consuming: %d seconds" % (time.time() - t))
    # print(rank_list,rank_list.shape)
    #
    # # # plt.yticks(np.linspace(0,1,10))
    # plt.hist(rank_list[:100000].tolist(),bins=20)
    # plt.savefig('1.jpg')
    # plt.show()

