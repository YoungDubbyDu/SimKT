from gensim.models import KeyedVectors
import numpy as np
import os

# 将题目的.emb转化为矩阵 embedding是题目的
def emb_transfer(read_path, save_path, tp, emb_size, emb_dim):
    if tp == 'rand':
        vectors_np = np.random.normal(size=(emb_size, emb_dim)).astype(np.float32)
    else:
        # 处理有些题目不在训练集中的情况
        wv_from_text = KeyedVectors.load_word2vec_format(read_path, binary=False)
        num_nodes = len(wv_from_text.vocab)
        vectors_np = np.zeros(shape=(emb_size, emb_dim), dtype=np.float32)
        for vocab in wv_from_text.vocab.keys():
            vectors_np[eval(vocab)] = list(wv_from_text.get_vector(vocab))
        print("total question number：%d, actual question number：%d" % (emb_size, num_nodes))
    np.save(save_path, vectors_np)


if __name__ == '__main__':
    for data_set in ["assist09_hkt"]:
        for MP in ['qtq']:
            for t in ['noWgt']:
                for numWalks in [10, 15]:
                    for walkLength in [80, 100]:
                        for dim in [128]:
                            for window_size in [3, 5]:
                                wv_path = "../%s/emb/%s_%s_%d_%d_%d_%d.emb" \
                                          % (data_set, MP, t, numWalks, walkLength, dim, window_size)
                                emb_transfer(wv_path, wv_path + '.npy', t, 18209, dim)

