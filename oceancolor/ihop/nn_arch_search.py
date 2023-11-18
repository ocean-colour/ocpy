""" Generate/Use NN to 'predict' Rs"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

#from oceancolor.ihop import io as ihop_io
import sys

from io_test import *
from IPython import embed
from nn import *

def archt_search(hidden_lists, lr_list, p_drop_list, epochs):
    hidden_list_opt, lr_opt, p_drop_opt = None, None, None
    loss_opt = 100
    batchnorm_opt = None
    for hidden_list in hidden_lists:
        for lr in lr_list:
            for p_drop in p_drop_list:
                print(f"Experiment for lr: {lr}, hidden_list: {hidden_list}, p: {p_drop} starts.")
                loss_i = build_densenet(
                    hidden_list,
                    epochs,
                    lr,
                    True,
                    p_drop,
                    False,
                    True,
                    f"densenet_{hidden_list}_epochs_{epochs}_p_{p_drop}_lr_{lr}",
                )
                if loss_i < loss_opt:
                    batchnorm_opt = False
                    hidden_list_opt = hidden_list
                    lr_opt = lr
                    p_drop_opt = p_drop
                print(f"Experiment for lr: {lr}, hidden_list: {hidden_list}, p: {p_drop} with batchnorm starts.")
                loss_i_bn = build_densenet(
                    hidden_list,
                    epochs,
                    lr,
                    True,
                    p_drop,
                    True,
                    True,
                    f"densenet_{hidden_list}_batchnorm_epochs_{epochs}_p_{p_drop}_lr_{lr}",
                )
                if loss_i_bn < loss_opt:
                    batchnorm_opt = True
                    hidden_list_opt = hidden_list
                    lr_opt = lr
                    p_drop_opt = p_drop
    return batchnorm_opt, hidden_list_opt, lr_opt, p_drop_opt, loss_opt

if __name__ == "__main__":
    epochs = 2500
    lr_list = [1e-2, 1e-3]
    hidden_lists = [
        [512, 128, 128],
        [512, 256, 128],
        [512, 512, 128],
        [512, 512, 256],
        [512, 512, 512, 256],
    ]
    p_drop_list = [0.0, 0.05]
    print("search starts.")
    opt_result = archt_search(hidden_lists, lr_list, p_drop_list, epochs)
    print("search ends.")
    print("opt loss is: ", opt_result[-1])
    print("opt hyperparas are: ", opt_result[:-1])
    #############################################
    ### best results provided by above models:
    ### epochs: 2500
    ### opt_loss: 0.001484
    ### model: (True,  [512, 512, 512, 256], 0.01, 0.0)
    #############################################
    
