# this file request detectron2 installing
import pickle
from train import train_eval_loop
import argparse
import torch
import sys
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import random
import torch.optim as optim
from model import BinaryPrediction
from config import cfg
from utils.log import setup_logger
from utils.pather import PathManager
from dataset import BinaryFetalHeartDataset
from data_argumentation import DATA_ARGUMENT_DICT
import logging
import os
from train import train_eval_loop
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils import comm
from detectron2.engine import launch
from model import BinaryPrediction,BinaryPredConv,MLP
import time

def main(train_list,eval_list,remain_list):
    seed = cfg.getint('other','seed')
    random.seed(seed)
    torch.manual_seed(seed)
    verbose = False
    logger = setup_logger()
    if comm.is_main_process():
        output_dir = cfg.get('other','output_dir')
        PathManager.mkdirs(output_dir)
        exp_name = \
               time.asctime( time.localtime(time.time())).replace(' ','_').replace(':','_')
        PathManager.mkdirs(output_dir + '/' + exp_name)
        cfg_name = output_dir + '/' + exp_name + '/' + 'config.conf'
        with open(cfg_name,'w') as f:
            cfg.write(f)
        verbose=True
        rm_list = np.array(remain_list)
        np.save(output_dir + '/' + exp_name + '/remain_list.npy',rm_list)


    model_type = cfg.get('model','type')
    if model_type == 'conv':
        model = BinaryPredConv()
    elif model_type == 'lstm':
        model = BinaryPrediction()
    elif model_type == 'fft':
        model = MLP()
    else:
        logger.error('please using exist model')
        raise NotImplementedError
    model_weight = ''
    try:
        model_weight = cfg.get('model','init_weight')
    except KeyError:
        logger.warn('no init weight in .conf, it may be an old version config')

    opt_name = cfg.get('train','opt_name')
    #print(opt_name)
    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                           lr = cfg.getfloat('train','lr'),
                           betas=(cfg.getfloat('train','beta1'),cfg.getfloat('train','beta2')),
                           weight_decay=cfg.getfloat('train','weight_decay'))
    elif opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(),momentum=cfg.getfloat('train','momentum'),
                              lr = cfg.getfloat('train','lr'),weight_decay=cfg.getfloat('train','weight_decay'))



    #print(max(eval_list))
    #print(comm.get_world_size())
    model.to(comm.get_local_rank())
    #print(comm.get_local_rank())
    world_size = comm.get_world_size()
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )
    if model_weight and model_weight != 'None':
        weight_dict = torch.load(model_weight)['model']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in weight_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        keep_last_layer = cfg.getboolean('model','keep_last_layer')

        if verbose:
            logger.info('using exist weight')
        if not keep_last_layer:
            if verbose:
                logger.info('reinit last layer')
            del model_dict['linear.weight']
            del model_dict['linear.bias']

        model.load_state_dict(model_dict)
    #print(model.device)

    argument_list = cfg.getliststr('data','argument')
    transforms = [DATA_ARGUMENT_DICT[ag]() for ag in argument_list]
    train_dataset = \
        BinaryFetalHeartDataset(train_list,transforms,device=model.device)
    eval_dataset = \
        BinaryFetalHeartDataset(eval_list,transforms,device=model.device)
    train_dataloader = DataLoader(train_dataset,cfg.getint('train','batch_size')//world_size,
                                shuffle=True,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset,cfg.getint('eval','batch_size'),shuffle=False,drop_last=True)


    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=cfg.getlistint('train','steps'),
                                    gamma = cfg.getfloat('train','lr_decay'),
                                    )
    if verbose:
        df = train_eval_loop(model,optimizer,lr_schedule,train_dataloader,eval_dataloader,model.device,
                    cfg.getint('train','epoch'),verbose=verbose,exp_dir=output_dir+'/'+exp_name+'/')
        df.to_csv(output_dir + '/' + exp_name + '/losses.csv', sep=',', header=True, index=True)
    else:
        train_eval_loop(model, optimizer, lr_schedule, train_dataloader, eval_dataloader, model.device,
                        cfg.getint('train', 'epoch'), verbose=verbose, exp_dir=None)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FetalHeart')
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )


    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    args = parser.parse_args()
    logger = setup_logger()
    data_num = len(os.listdir(cfg.get('data', 'data_dir')))
    # print('data_num:',data_num)
    train_num = cfg.getint('train', 'train_num')
    eval_num = cfg.getint('eval', 'eval_num')
    try:
        assert train_num + eval_num < data_num
    except AssertionError:
        logger.error('main:train_num + eval_num must smaller than data_num. check dataset or your cfg')
        quit()
    assigned = False
    try:
        assigned = cfg.getboolean('eval','assigned')
    except KeyError:
        logger.warn('no assigned in .conf, it may be an old version config')
    if assigned:
        try:
            train_start_list = cfg.getlistint('data','train_start')
            train_end_list = cfg.getlistint('data','train_end')
            eval_start_list = cfg.getlistint('data','eval_start')
            eval_end_list = cfg.getlistint('data','eval_end')
        except KeyError:
            logger.error('using an old config which will cause data leak,'
                         ' pleasce check [data] in your config')
        eval_list = []
        train_list = []
        for i in range(3):
            eval_list.extend([_ for _ in range(eval_start_list[i],eval_end_list[i])])
            train_list.extend([_ for _ in range(train_start_list[i],train_end_list[i])])

        remain_list = [1,]
    else:
        fulllist = [_ for _ in range(data_num)]
        random.shuffle(fulllist)
        train_list = fulllist[:train_num]
        eval_list = fulllist[train_num:train_num + eval_num]
        remain_list = fulllist[train_num+eval_num:]
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(train_list,eval_list,remain_list),
    )
