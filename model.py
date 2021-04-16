import torch
import torch.nn as nn
from config import cfg
from utils.model_init import normal_init
import torch.nn.functional as F
import numpy as np
from config import cfg


class BinaryPrediction(nn.Module):
    #TODO:explict calculate the padding size
    def __init__(self):
        super(BinaryPrediction, self).__init__()
        self.input_dim = cfg.getint('data','input_dim')
        self.sequence_len = cfg.getint('data','seq_len')

        self.output_dim = cfg.getint('model','output_dim')
        self.hidden_dim = cfg.getint('model','hidden_dim')
        self.num_layers = cfg.getint('model','num_layers')
        self.dropout = cfg.getfloat('model','dropout')
        self.conv_kernel = cfg.getint('model','kernel_size')
        self.output_len = cfg.getint('model','output_len')
        self.bidirectional = cfg.getboolean('model','bidirectional')

        self.batch_size = cfg.getint('train','batch_size')

        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                            dropout=self.dropout,num_layers=self.num_layers,bias=True,
                            bidirectional=self.bidirectional)
        self.bi = 2 if self.bidirectional else 1
        #self.batch_norm = nn.BatchNorm1d(self.hidden_dim*self.bi)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim*self.bi,self.output_dim),nn.ReLU(inplace=True),
                                 nn.Linear(self.output_dim,self.output_dim))
        self.h0 = torch.zeros([self.num_layers*self.bi,self.batch_size,self.hidden_dim],device=self.device)
        self.c0 = torch.zeros([self.num_layers*self.bi,self.batch_size,self.hidden_dim],device=self.device)
        self.conv = nn.Conv1d(self.output_dim,self.output_dim,
                              kernel_size=self.conv_kernel,padding=(self.conv_kernel-1)//2,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=self.output_len)
        self.pred_head = nn.Linear(self.output_len*self.output_dim,1,bias=False)


    def __str__(self):
        str = 'model config:\n'
        str += 'input_dim:{},output_dim:{},hidden_dim:{}'\
            .format(self.input_dim,self.output_dim,self.hidden_dim)
        str += 'num_layers:{},dropout:{}'.format(self.num_layers,self.dropout)
        return str

    def forward(self,x:torch.Tensor):
        #print(x.size())

        x,_ = self.lstm(x,(self.h0,self.c0))
        #x,(len,batch,feature)
        x = x.permute(1, 2, 0)
        # x,(batch,feature,len)
        #x = self.batch_norm(x)
        x = x.permute(0,2,1)
        # x,(batch,len,feature)
        x = self.mlp(x)
        x = x.permute(0,2,1)
        # x,(b,f,l)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0),-1)

        x = self.pred_head(x)
        x.sigmoid_()
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=cfg.getint('model','dilation'), bias=False,
                               dilation=cfg.getint('model','dilation'))
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=cfg.getint('model','dilation'), bias=False,
                               dilation=cfg.getint('model','dilation'))
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=stride, padding=cfg.getint('model','dilation'), bias=False,
                               dilation=cfg.getint('model','dilation'))
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes,
                               kernel_size=3,padding=cfg.getint('model','dilation'), bias=False,
                               dilation=cfg.getint('model','dilation'))
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        pooler_type = cfg.get('model','pooler')
        if pooler_type == 'linear':
            self.linear_pool = True
        else:
            self.linear_pool = False
        self.device = 'cuda:0'
        self.in_planes = 64
        try:
            _fks = cfg.getint('model','first_kernel')
        except KeyError:
            print('Using an old version cfg,set first kernel size to 7')
            _fks = 7
        assert _fks%2==1, 'kernel size should be odd'
        self.conv1 = nn.Conv1d(1, 64, kernel_size=_fks,
                               stride=1, padding=(_fks-1)//2, bias=True)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.pooler = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5,inplace=True)
        if self.linear_pool:
            self.linear_drop = nn.Dropout(0.2,inplace=True)
            xlen = 4800
            print('warning: using linear_pooler need input length = {}'.format(xlen))
            print('dw_len will cause this term different')
            self.linear_pooler = nn.Conv1d(512,512,xlen//8,groups=512)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        #print(torch.min(x))
        out = F.relu(self.bn1(self.conv1(x)))
        #print(dataset_changed.size())
        #print(torch.min(dataset_changed))
        out = self.layer1(out)
        #print(dataset_changed.size())
        #print(torch.min(dataset_changed))
        out = self.layer2(out)
        #print(dataset_changed.size())
        #print(torch.min(dataset_changed))
        out = self.layer3(out)
        #print(dataset_changed.size())
        #print(torch.min(dataset_changed))
        out = self.layer4(out)
        #print(dataset_changed.size())
        #print(torch.min(dataset_changed))
        if self.linear_pool:
            #print(dataset_changed.size())
            out = self.linear_drop(out)
            out = self.linear_pooler(out)
        else:
            out = self.pooler(out)
        #print(torch.min(dataset_changed))
        out = self.dropout(out)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out




def BinaryPredConv():
    if cfg.get('model','block') == 'BottleNeck':
        return ResNet(Bottleneck,cfg.getlistint('model','block_list'))
    else:
        return ResNet(BasicBlock, cfg.getlistint('model', 'block_list'))

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(2400,1)
        self.norm1 = nn.BatchNorm1d(2400)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        #print('1',x.size())
        #x = self.norm1(self.relu(self.linear1(x)))
        #print('2',x.size())
        #x = self.norm2(self.relu(self.linear2(x)))
        #x = self.linear3(x)
        x = self.linear1(self.norm1(x))
        x = x.sigmoid()
        return x



if __name__ == '__main__':
    model = BinaryPrediction()
    print(model)
    x = torch.randn([50,2,10],device=model.device)
    pred = model(x)
    print(pred)
