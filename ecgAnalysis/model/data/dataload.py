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
import time
warnings.filterwarnings("ignore")

class ecgData(torch.utils.data.Dataset):
    def __init__(self, train = True):
        self.train = train
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.path.dirname(data_dir) + '/ecgdata_not_mine/archive'
        if self.train == True:
            print('Load train data...')
            data = np.loadtxt('{}/mitbih_train.csv'.format(input_dir), delimiter = ',', dtype = np.float32)
            
            self.data_len = data.shape[0]
            self.train_data = torch.from_numpy(data[:,:187])
            self.train_label = torch.from_numpy(data[:,[187]])

        else:
            print('Load test data...')
            test = np.loadtxt('{}/mitbih_test.csv'.format(input_dir), delimiter = ',', dtype = np.float32)
            self.test_len = test.shape[0]
            self.test_data = torch.from_numpy(test[:,:187])
            self.test_label = torch.from_numpy(test[:,[187]])

    def __len__(self):
        if self.train:
            return self.data_len
        else:
            return self.test_len

    def __getitem__(self, index):
        if self.train:
            return self.train_data[index], self.train_label[index]
        else:
            return self.test_data[index], self.test_label[index]