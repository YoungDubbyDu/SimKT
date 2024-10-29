import argparse
import numpy as np
from gensim.models import Word2Vec
import time
from pre_emb.EmbTransfer import emb_transfer

import os

# 定义超参数
def parse_args():
    """
    Parses the SkipGram arguments.
    """
    parser = argparse.ArgumentParser(description="Run SkipGram.")

    parser.add_argument('--input', nargs='?', default='',
                        help='Input walks path')

    parser.add_argument('--output', nargs='?', default='',
                        help='Embeddings path')

    parser.add_argument('--dim', type=int, default=32,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--windowSize', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    return parser.parse_args()


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size=args.dim, window=args.windowSize, min_count=0, sg=1, workers=args.workers, iter=args.iter,negative=5)
    model.wv.save_word2vec_format(args.output)

    return


def main():
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    # 将得到的序列列表，放进函数中输出列表中每个元素的嵌入
    # get sentences
    print("reading walks")
    walks = []
    for line in open(args.input, 'r'):
        walk = line.rstrip().split(',')
        walks.append([eval(w) for w in walk])

    # # remove non-question nodes
    # print("removing non-question nodes")
    # processedWalks = []
    # for one_walk in walks:
    #     one_walk = np.array(one_walk)
    #     processedWalks.append(one_walk)

    # get question embeddings
    print("start to learn embedding")
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    data_path = "E:/Study/KT/DataProcess2022/"
    data_set = "Ednet"
    # n_question = len(eval(open(os.path.join(data_path, data_set, 'encode', 'question_id_dict.txt')).read()))
    n_question = np.load(os.path.join(data_path, data_set, "graph", "ques_skill_mat.npy")).shape[0]
    read_folder = "E:/Study/SimKT/SimKT/pre_emb/%s/walks" % data_set
    save_folder = "./pre_emb/%s/emb" % data_set
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    for MP in ['qkq']:
        for numWalks in [10]:
            for walkLength in [80]:
                for t in ["contDiff"]:
                    # parameters for learning node embeddings as below
                    for args.dim in [32]:
                        for args.windowSize in [3]:
                            for args.iter in [20]:
                                args.input = os.path.join(read_folder, "walks_%s_%s_%d_%d.txt" % (
                                    MP, t, numWalks, walkLength))
                                args.output = os.path.join(save_folder, "%s_%s_%d_%d_%d_%d.emb" % (
                                    MP, t, numWalks, walkLength, args.dim, args.windowSize))
                                t1 = time.time()
                                main()
                                emb_transfer(args.output, args.output + '.npy', t, n_question, args.dim)
                                print("用时%d秒" % (time.time() - t1))
