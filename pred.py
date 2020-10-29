# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import argparse
import os

import numpy as np

from init_model import init_model

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--iter', default='8000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 8000')
parser.add_argument('--path', default='./demos/Positive_CAC_1.npy', type=str,
                    help='path: path of the input image. Default: ./demos/Positive_CAC_1.npy')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

m = init_model()
m.load_model(opt.iter)
data = np.load(opt.path)
print('Estimated CVD Risk:', m.aug_transform(data)[1])
