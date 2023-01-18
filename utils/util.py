import numpy as np
import torch
import torch.nn as nn
import time, math, random
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Categorical
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from collections import deque
from pprint import pprint
import subprocess, sys, os
import itertools
from queue import Queue
from tqdm import trange, tqdm
from urllib import request
from matplotlib import pyplot as plt
import warnings
import wget

warnings.filterwarnings("ignore")

chmap = {'C': 0, 'G': 1, 'A': 2, 'T': 3}
idmap = {0: 'C', 1: 'G', 2: 'A', 3: 'T'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_DIR = '/home/qhoang/Code/masked-minimizer/seqdata/'

def seed(np_seed=19932010, torch_seed=26031993):
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)

def init_weights(m, method='kaiming'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if method == 'kaiming':
            torch.nn.init.kaiming_uniform_(m.weight)
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)

def cuda_memory(device):
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    return f'Total:{t}, Cached:{c}, Alloc:{a}, Free:{t-c-a}'

def print_cuda_memory(device, prefix=''):
    print(prefix + cuda_memory(device))

def sequence_mer_iterator(k, seq):
    slen = len(seq)
    mod_low = 4 ** (k-1)
    cur = 0
    for i in range(k-1):
        cur = cur * 4 + chmap[seq[i]]
    for i in range(k-1, slen):
        if i >= k:
            cur -= mod_low * chmap[seq[i-k]]
        cur = cur * 4 + chmap[seq[i]]
        yield cur

def random_sequence(slen, seed = None):
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice('ACTG') for _ in range(slen))

def kmer_to_int(km):
    ret = 0
    for c in km:
        ret = ret * 4 + chmap[c]
    return ret

def num_to_mask(_w, _n):
    assert _n < 2 ** _w
    binstr = bin(_n)
    mask = torch.zeros(_w)
    for i, j in enumerate(range(len(binstr)-1, 1, -1)):
        mask[_w - i - 1] = torch.tensor(int(binstr[j]))
    return mask