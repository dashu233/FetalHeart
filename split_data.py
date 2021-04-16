from config import cfg
from path import Path
import numpy as np
import pandas as pd
import os
import pickle

if __name__ == "__main__":
    split_num = cfg.getint('data', 'split_num')
    split_data = Path("split_data").makedirs_p()
    data_num = len(os.listdir('dataset'))
    idx = 0
    train_start_list = cfg.getlistint('data', 'train_start')
    train_end_list = cfg.getlistint('data', 'train_end')
    train_list = []
    for i in range(3):

        train_list.extend([_ for _ in range(train_start_list[i], train_end_list[i])])
    for file_id in train_list:
        with open('dataset/{}.pkl'.format(file_id),'rb') as f:
            dt = pickle.load(f)
        x_len = len(dt['x'])
        sp_len = x_len//split_num
        for sp_id in range(split_num):
            obj = {"x": dt['x'][sp_id * sp_len:(sp_id + 1) * sp_len], "y": dt['y']}
            nonzero = np.where(obj['x']>0.001)[0]
            if not len(nonzero):
                continue
            with open(split_data / str(idx) + ".pkl", "wb") as fp:

                if np.isnan(np.min(obj['x'])):
                    print(np.min(obj['x']))
                pickle.dump(obj, fp)
                idx += 1

