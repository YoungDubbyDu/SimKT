import torch
import numpy as np
device = torch.device('cuda:0')
mp_npy = np.load('E:/Study/SimKT/SimKT/pre_emb/ASSIST09/emb/qkckq_contDiff_64_10_80_128_3.emb.npy')
mp_dict=dict()
mp_dict['quq']= torch.from_numpy(mp_npy).to(device)
