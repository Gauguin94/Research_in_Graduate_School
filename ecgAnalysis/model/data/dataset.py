import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F #1. 라이브러리 가져오기
from torch.utils.data import DataLoader, Dataset
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from . import *

BATCH_SIZE = 128
TEST_BATCH = 21892

class setData:
    def __init__(self):
        self.trainset = ecgData(train = True)
        self.testset = ecgData(train = False)
    def save_data(self):
        train_batch = DataLoader(self.trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 6)
        val_batch = DataLoader(self.trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 6)
        test_batch = DataLoader(self.testset, batch_size = BATCH_SIZE, shuffle = False, num_workers= 6)
        whole_test_batch = DataLoader(self.testset, batch_size = TEST_BATCH, shuffle = False, num_workers= 6)
        return train_batch, val_batch, test_batch, whole_test_batch, self.trainset, self.testset