import torch
import pandas as pd
import numpy as np
import time
import os


def qu_AdjMat():  # get adjacency matrix
    df = pd.read_csv(os.path.join(dir_path, data_set, "graph", "stu_ques.csv"))
    num_stu = int(df['stu'].max()) + 1

    df_correct = df[df['correct'] == 1]
    AdjMat_c = np.zeros(shape=(num_ques, num_stu), dtype=np.float32)
    for index, row in df_correct.iterrows():  # 矩阵里面是1，表示对应学生i做对了题目j，否则可能没做或者做错了
        stuID, quesID, answer = int(row['stu']), int(row['ques']), 1
        AdjMat_c[quesID][stuID] = answer

    df_wrong = df[df['correct'] == 0]
    AdjMat_w = np.zeros(shape=(num_ques, num_stu), dtype=np.float32)
    for index, row in df_wrong.iterrows():
        stuID, quesID, answer = int(row['stu']), int(row['ques']), 1
        AdjMat_w[quesID][stuID] = answer
    return torch.from_numpy(AdjMat_c), torch.from_numpy(AdjMat_w)


def get_mat(A_qu, B_qu):  # get abcd matrix
    B_uq = torch.transpose(B_qu, dim0=0, dim1=1)
    return torch.matmul(A_qu, B_uq)


if __name__ == "__main__":
    dir_path = "E:/Study/SimKT/SimKT/data"
    data_set = "ASSIST12/"
    save_folder = "./%s/sim_mat" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(dir_path, data_set, "encode", "question_id_dict.txt")) as f:
        dict_to_read = eval(f.read())
    num_ques = len(dict_to_read)

    t = time.time()

    adj_qu_c, adj_qu_w = qu_AdjMat()
    a_mat = get_mat(adj_qu_w, adj_qu_w)  # i作错，j做错
    b_mat = get_mat(adj_qu_c, adj_qu_w)  # i作对，j做错
    c_mat = get_mat(adj_qu_w, adj_qu_c)  # i作错，j做对
    d_mat = get_mat(adj_qu_c, adj_qu_c)  # i作对，j做对

    np.save(os.path.join(save_folder, "a_mat.npy"), a_mat)
    np.save(os.path.join(save_folder, "b_mat.npy"), b_mat)
    np.save(os.path.join(save_folder, "c_mat.npy"), c_mat)
    np.save(os.path.join(save_folder, "d_mat.npy"), d_mat)

    print("time consuming: %d seconds" % (time.time() - t))