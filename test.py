from config import cfg
import pickle
import numpy as np
import torch
from model import BinaryPredConv
from data_argumentation import DATA_ARGUMENT_DICT
import os
from torch.nn.parallel import DistributedDataParallel
from dataset import BinaryFetalHeartDataset
from train import eval
from torch.utils.data.dataloader import DataLoader
#from detectron2.utils import comm
#from detectron2.engine import launch
import sys
import argparse
from utils.log import setup_logger
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1.2, 2.5, 4.5, 7.3]



def main():
    Draw = True
    best_model_path = cfg.get('test', 'model')
    model = BinaryPredConv()
    with open(best_model_path, 'rb') as f:
        info = torch.load(f,map_location=torch.device('cpu'))
        for k in info.keys():
            if k == 'model':
                continue
            print(k, ':', info[k])
        weight_dict = {}
        for k in info['model']:
            weight_dict[k[7:]] = info['model'][k]

        model.load_state_dict(weight_dict)
    #test_list = cfg.getlistint('test', 'test_list')
    test_train = False
    if test_train:
        test_start_list = cfg.getlistint('data', 'eval_start')
        test_end_list = cfg.getlistint('data', 'eval_end')
    else:
        test_start_list = cfg.getlistint('data', 'test_start')
        test_end_list = cfg.getlistint('data', 'test_end')
    #print(test_start_list)
    #test_list = [_ for _ in range(test_start_list[2],test_end_list[2])]
    #print(len(test_list))
    #test_list = []
    #for i in range(3):
        #test_list.extend([_ for _ in range(test_start_list[i], test_end_list[i])])
    test_list = [_ for _ in range(6663)]
    argument_list = cfg.getliststr('data', 'argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]

    #print(model.conv1.weight)
    #return

    if Draw:
        test_dataset = \
            BinaryFetalHeartDataset(test_list, transforms, reidx=True)
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False, drop_last=True)
        print(len(test_dataloader))

        eval(model, test_dataloader, device='cpu', verbose=True, details=True,draw=True)
    else:
        test_dataset = \
            BinaryFetalHeartDataset(test_list, transforms)
        test_dataloader = DataLoader(test_dataset, 128, shuffle=False, drop_last=True)
        eval(model, test_dataloader, device='cpu', verbose=True)

if __name__ == '__main__':
    main()


