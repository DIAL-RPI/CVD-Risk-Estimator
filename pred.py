# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import argparse
import os

import numpy as np

from model import Model

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--iter', default='8000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 8000')
parser.add_argument('--path', default='./demos/Positive_CAC_1.npy', type=str,
                    help='path: path of the input image. Default: ./demos/Positive_CAC_1.npy')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_config = {
    'dout': True,
    'lr': 1e-4,
    'num_workers': 8,
    'batch_size': 32,
    'restore_iter': 0,
    'total_iter': 20000,
    'model_name': 'NLST-CVD3x2D-Res18',
    'train_source': None,
    'val_source': None,
    'test_source': None
}
model_config['save_name'] = '_'.join([
    '{}'.format(model_config['model_name']),
    '{}'.format(model_config['dout']),
    '{}'.format(model_config['lr']),
    '{}'.format(model_config['batch_size']),
])

m = Model(**model_config)
m.load_model(opt.iter)
data = np.load(opt.path)
print('Estimated CVD Risk:', m.aug_transform(data)[1])
