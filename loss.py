import torch.nn as nn
import torch
from config import cfg

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.pos_weight = cfg.getint('loss','pos_weight')
    def forward(self,output,target):
        output = output.view(-1)
        target = target.view(-1)
        pos_ids = torch.where(target>0)[0]
        neg_ids = torch.where(target<0)[0]
        return -self.pos_weight*output[pos_ids].sum()+output[neg_ids].sum()