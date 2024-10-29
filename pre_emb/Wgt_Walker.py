import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
import time
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
    return torch.tensor(x).to(device)


class WeightedWalker:
    # 根据边权重，动态地确定节点的采样概率
    # 其完成从A类节点（头节点）到B类节点（尾节点）的一步游走，v_num是A类节点的总数，edges是A类节点指向B类节点的边的列表
    def __init__(self, v_num, edges):
        self.v_num = v_num

        ngbrs = [[] for i in range(v_num)]
        weights = [[] for i in range(v_num)]
        for u, v, w in edges:
            u = int(u)
            v = int(v)
            # 将问题对应的技能全部放入
            ngbrs[u].append(v)
            # 将对应权重放入
            weights[u].append(w)
        # 处理为等长的tensor
        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True)
        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]) # 获取邻居数
        self.max_ngbr_num = torch.max(self.ngbr_nums).item()  # 获取最大邻居数
        # self.D = 1

    def get_prob(self, weights, nums, the_weights):
        # prob = to_tensor([1]) - torch.abs(weights-the_weights) / to_tensor([self.D])
        prob = - torch.abs(weights - the_weights)

        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0)
        nums = torch.unsqueeze(nums, 1)
        mask = x >= nums
        prob = prob.masked_fill(mask, -1e32)
        prob = F.softmax(prob, dim=1)
        return prob  # [v_num * walk_num, max_ngbr_num]

    def get_next(self, v, the_weights):
        expand_pad_weights = self.weights[v]
        expand_ngbr_nums = self.ngbr_nums[v]
        expand_pad_prob = self.get_prob(expand_pad_weights, expand_ngbr_nums, the_weights)
        next_index = torch.multinomial(expand_pad_prob, num_samples=1)

        expand_pad_ngbrs = self.ngbrs[v]
        next_v = torch.gather(expand_pad_ngbrs, 1, next_index).flatten()

        the_weights = torch.gather(expand_pad_weights, 1, next_index)
        return next_v, the_weights

    def create_paths(self, walk_num, walk_len):
        next_v = torch.arange(self.v_num).to(device).repeat(walk_num, 1).T.flatten()
        paths = [next_v]
        the_weights = to_tensor([1e32]).unsqueeze(1)
        for i in range(walk_len - 1):
            print("%dth hop" % i)
            next_v, the_wgt = self.get_next(next_v, the_weights)
            paths.append(next_v)

        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1)
        paths = paths.view(-1, walk_len)

        return paths


def create_qkq_paths():
    # 虚构一个二部图
    num_ques, num_skill, num_edge = 5000, 12000, 500000
    head_nodes = list(np.random.choice(num_ques, num_edge))
    tail_nodes = list(np.random.choice(num_skill, num_edge))
    weight_list = list(np.random.random(num_edge))
    qk_edge_list = list(zip(head_nodes, tail_nodes, weight_list))
    kq_edge_list = list(zip(tail_nodes, head_nodes, weight_list))

    # 构建QK_Walker和KQ_walker
    QK_Walker = WeightedWalker(v_num=num_ques, edges=qk_edge_list)
    KQ_Walker = WeightedWalker(v_num=num_skill, edges=kq_edge_list)

    # 执行带权随机游走，获取节点序列
    walk_num, walk_len = 10, 80
    next_q = torch.arange(num_ques).to(device).repeat(walk_num, 1).T.flatten()
    paths = [next_q]
    the_weights = to_tensor([1e32]).unsqueeze(1)
    for i in range(walk_len - 1):
        print("%dth hop" % i)
        next_k, the_weights = QK_Walker.get_next(next_q, the_weights)
        next_q, the_weights = KQ_Walker.get_next(next_k, the_weights)
        paths.append(next_q)

    paths = [path.unsqueeze(-1) for path in paths]
    paths = torch.cat(paths, dim=-1)
    paths = paths.view(-1, walk_len)
    return paths


if __name__ == '__main__':
    t = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_path = create_qkq_paths()
    print(all_path)
    print("spend %d seconds: " % (time.time()-t))
