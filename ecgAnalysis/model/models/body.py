import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from .runCUDA import*
SEED_VALUE = 777
device = CUDA_().run(SEED_VALUE)

#def concat_path(out_dim): # 
def concat_path(out_dim, floor_num):
    layers = []
    dim = out_dim
    layers.append(nn.AvgPool1d(2)) # nn.AvgPool1d(floor_num) ##
    layers.extend([
        nn.BatchNorm1d(dim),
        nn.LeakyReLU(),
        nn.Conv1d(dim, dim, kernel_size = 1),
    ])
    return nn.Sequential(*layers).to(device)

def new_identity(dim, floor_num, div_num):
    layers = []
    in_dim = dim
    out_dim = dim * 4 * (2**(floor_num - 1))
    
    if floor_num < 3:
        layers.append(nn.Conv1d(1, in_dim, kernel_size = 7))    
        layers.extend([
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size = 1),
            nn.AvgPool1d(3 * (2**(div_num - 1))),
            nn.BatchNorm1d(out_dim)
        ])
    else:
        layers.append(nn.Conv1d(1, in_dim, kernel_size = 7, padding = 2))    
        layers.extend([
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size = 1, padding = 2),
            nn.AvgPool1d(3 * (2**(div_num - 1))),
            nn.BatchNorm1d(out_dim)
        ])        
    return nn.Sequential(*layers).to(device)

def start_layer(dim):
    layers = []
    layers.append(nn.Conv1d(1, dim, kernel_size = 7))
    layers.extend([
        nn.BatchNorm1d(dim),
        nn.LeakyReLU(),
        nn.MaxPool1d(3)
    ])
    return nn.Sequential(*layers).to(device)

def make_conv(in_dim, mid_dim, out_dim, down = False):
    layers = []
    width = mid_dim // 64 * 32 * 4
    downsizing = 2 if down else 1
    layers.append(nn.Conv1d(in_dim, width, kernel_size = 1, stride = downsizing))
    layers.extend([
        nn.BatchNorm1d(width),
        nn.LeakyReLU(),
        nn.Conv1d(width, width, groups = 32, kernel_size = 3, padding = 1),
        nn.BatchNorm1d(width),
        nn.LeakyReLU(),
        nn.Conv1d(width, out_dim, kernel_size = 1),
        nn.BatchNorm1d(out_dim)
    ])
    return nn.Sequential(*layers).to(device)

def floor_layer(repeat_count, in_dim, mid_dim, out_dim, start = False):
    layers = []
    layers.append(tenantLayer(in_dim, mid_dim, out_dim, down = True, start = start))
    for _ in range(1, repeat_count):
        layers.append(tenantLayer(out_dim, mid_dim, out_dim, down = False))
    return nn.Sequential(*layers).to(device)

class tenantLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down = False, start = False):
        super(tenantLayer, self).__init__()
        if start:
            down = False
        self.block = make_conv(in_dim, mid_dim, out_dim, down)
        self.leaky_relu = nn.LeakyReLU()
        downsizing = 2 if down else 1
        resize_layer = nn.Conv1d(in_dim, out_dim, kernel_size = 1, stride = downsizing)
        self.resize = nn.Sequential(resize_layer, nn.BatchNorm1d(out_dim))

    def forward(self, x):
        identity = self.resize(x)
        x = self.block(x)
        x += identity
        x = self.leaky_relu(x)
        return x

class kModel(nn.Module):
    def __init__(self, repeat_count = 3, label = 5):
        super(kModel, self).__init__()
        dim = 64

        self.stage0 = start_layer(dim)
        
        self.stage1_1 = floor_layer(repeat_count, dim, dim, dim*4, start = True)
        self.stage1_2 = floor_layer(repeat_count, dim*4, dim, dim*4)
        self.stage1_3 = floor_layer(repeat_count, dim*4, dim, dim*4)
        self.stage1_4 = floor_layer(repeat_count, dim*4, dim, dim*4)

        self.stage2_1 = floor_layer(repeat_count, dim*4, dim*2, dim*8)
        self.stage2_2 = floor_layer(repeat_count, dim*8, dim*2, dim*8)
        self.stage2_3 = floor_layer(repeat_count, dim*8, dim*2, dim*8)
        self.stage2_4 = floor_layer(repeat_count, dim*8, dim*2, dim*8)

        self.stage3_1 = floor_layer(repeat_count, dim*8, dim*4, dim*16)
        self.stage3_2 = floor_layer(repeat_count, dim*16, dim*4, dim*16)
        self.stage3_3 = floor_layer(repeat_count, dim*16, dim*4, dim*16)
        self.stage3_4 = floor_layer(repeat_count, dim*16, dim*4, dim*16)

        self.stage4 = floor_layer(repeat_count, dim*16, dim*8, dim*32)

        self.stage1_ = new_identity(dim, 1, 1)
        self.stage2_ = new_identity(dim, 2, 2)
        self.stage3_ = new_identity(dim, 3, 3)

        self.layer1_path = concat_path(dim*4, 1)
        self.layer2_path = concat_path(dim*8, 2)
        self.layer3_path = concat_path(dim*16, 3)

        self.avgpool = nn.AvgPool1d(8).to(device)

        self.fc1 = nn.Linear(2048, 32).to(device)
        self.fc2 = nn.Linear(32, label).to(device)

        self.leaky_relu = nn.LeakyReLU().to(device)
        self.dropout = nn.Dropout(0.3).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight).to(device)
        torch.nn.init.xavier_uniform_(self.fc2.weight).to(device)

    def forward(self, x):
        in_size = x.size(0) # 128, 1, 187
        x_origin = x # 128, 1, 187

        x = self.stage0(x) # 128 64 60

        #####################################################
        x = self.stage1_1(x) # 128 256 60
        x = [x, self.stage1_(x_origin)] # right: 128 256 60
        x = torch.cat(x, 2).to(device) # 128 256 120

        x = self.stage1_2(x) # 128 256 60
        x = [x, self.stage1_(x_origin)] # right: 128 256 60
        x = torch.cat(x, 2).to(device) # 128 256 120

        x = self.stage1_3(x) # 128 256 60
        x = [x, self.stage1_(x_origin)] # right: 128 256 60
        x = torch.cat(x, 2).to(device) # 128 256 120

        x = self.stage1_4(x) # 128 256 60
        x = [x, self.stage1_(x_origin)] # right: 128 256 60
        x = torch.cat(x, 2).to(device) # 128 256 120
        x = self.layer1_path(x) # 128 256 60
        #####################################################

        #####################################################
        x = self.stage2_1(x) # 128 512 30
        x = [x, self.stage2_(x_origin)] # right: 128 512 30
        x = torch.cat(x, 2).to(device) # 128 512 60

        x = self.stage2_2(x) # 128 512 30
        x = [x, self.stage2_(x_origin)] # right: 128 512 30
        x = torch.cat(x, 2).to(device) # 128 512 60

        x = self.stage2_3(x) # 128 512 30
        x = [x, self.stage2_(x_origin)] # right: 128 512 30
        x = torch.cat(x, 2).to(device) # 128 512 60

        x = self.stage2_4(x) # 128 512 30
        x = [x, self.stage2_(x_origin)] # right: 128 512 30
        x = torch.cat(x, 2).to(device) # 128 512 60
        x = self.layer2_path(x) # 128 512 30
        #####################################################

        #####################################################
        x = self.stage3_1(x) # 128 1024 15
        x = [x, self.stage3_(x_origin)] # right: 128 1024 15
        x = torch.cat(x, 2).to(device) # 128 1024 30

        x = self.stage3_2(x) # 128 1024 15
        x = [x, self.stage3_(x_origin)] # right: 128 1024 15
        x = torch.cat(x, 2).to(device) # right: 128 1024 30

        x = self.stage3_3(x) # 128 1024 15
        x = [x, self.stage3_(x_origin)] # right: 128 1024 15
        x = torch.cat(x, 2).to(device) # right: 128 1024 30

        x = self.stage3_4(x) # 128 1024 15
        x = [x, self.stage3_(x_origin)] # right: 128 1024 15
        x = torch.cat(x, 2).to(device) # right: 128 1024 30
        x = self.layer3_path(x) # 128 1024 15
        #####################################################


        x_stage_out = self.stage4(x) # 128 2048 8

        x_out = self.avgpool(x_stage_out) # 128 2048 1

        out = x_out

        out = out.view(in_size, -1).to(device) # 128, 2048
        out = self.fc1(out) # 128, 32
        out = self.dropout(out)
        out = self.leaky_relu(out)
        out = self.fc2(out) # 128, 5

        return out        