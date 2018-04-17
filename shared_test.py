# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:45:34 2018

@author: JAE
"""

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
from multiprocessing import Value , Queue, Lock
import datetime
#import torchvision.transforms as T
from collections import defaultdict, deque
import sys
import os 

def init_weights(m):
    print(m)
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(1.0)
        print(m.weight)

from model import Net
    
share_model = Net(6,6)
share_model.share_memory()



def act_process(rank,share_model):
    print(rank, share_model.act_conv1.weight[0][0])
    if rank is 0:
        time.sleep(1)
        share_model.load_state_dict( Net(6,6).state_dict())
        print(rank, share_model.act_conv1.weight[0][0])
    else :
        time.sleep(2)
        print(rank, share_model.act_conv1.weight[0][0])
    
    pass

if __name__ =='__main__':
    processes = []
    num_processes = 2
    for rank in range(num_processes):
        p = mp.Process(target=act_process, args=(rank,share_model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    