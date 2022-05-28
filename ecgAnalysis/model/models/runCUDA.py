import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import sys
import os
import warnings
import numpy as np
import random
import time
warnings.filterwarnings("ignore")

class CUDA_:
    def __init__(self):
        pass

    def set_random_seed(self, seed_value, use_cuda = True):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if use_cuda:
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run(self, seed_value):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.set_random_seed(seed_value, True)
            print('CUDA ON!')
        else:
            self.set_random_seed(seed_value, False)
            print('CPU ON!')
        return device