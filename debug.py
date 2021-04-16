import pickle
import os
import numpy as np

for file_id in range(6663):
    with open('named_data/{}.pkl'.format(file_id), 'rb') as f:
        dt = pickle.load(f)
    print('num:{} '.format(file_id) + dt['name'] + ' {}'.format(dt['y']))