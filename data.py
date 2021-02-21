# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import os.path as osp
import random

import numpy as np
import pandas as pd
import torch.utils.data as tordata
from scipy.ndimage import gaussian_filter


def load_dataset(datainfo_path, dataset_path, test_groups=[1], validation_groups=[]):
    assert len(set(test_groups) & set(validation_groups)) <= 0, 'Overlap between validation and test'
    data_info_list = pd.read_csv(datainfo_path, index_col=None)
    test_set = {'data_path': list(), 'label': list()}
    validation_set = {'data_path': list(), 'label': list()}
    train_set = {'data_path': list(), 'label': list()}

    def subset_insert(_subset, _path, _label):
        _subset['data_path'].append(_path)
        _subset['label'].append(_label)

    data_size = data_info_list.shape[0]
    for i in range(data_size):
        _info = data_info_list.iloc[i]
        label = _info['detail_label']
        group = _info['group']
        pid = _info['pid']
        case = _info['case']
        vol = _info['vol']
        v_name = '_'.join([str(pid), case, vol]) + '.npy'
        v_path = osp.join(dataset_path, v_name)
        if group in test_groups:
            subset_insert(test_set, v_path, label)
        elif group in validation_groups:
            subset_insert(validation_set, v_path, label)
        else:
            subset_insert(train_set, v_path, label)

    return DataSet(train_set, True), DataSet(validation_set), DataSet(test_set)


class Augmentor():
    def __call__(self, image):
        s = random.randint(0, 16)
        h = random.randint(0, 16)
        w = random.randint(0, 16)
        return image[s:s + 112, h:h + 112, w:w + 112]


class DataSet(tordata.Dataset):
    def __init__(self, subset_dict, aug=False):
        super(DataSet, self).__init__()
        self.data_path = subset_dict['data_path']
        self.label = subset_dict['label']
        self.data_size = len(self.label)
        self.label_set = set(self.label)
        self.if_aug = aug
        self.augmentor = Augmentor()

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def clear_cache(self):
        self.data = [None] * self.data_size

    def __loader__(self, path):
        return np.load(path)

    def __getitem__(self, index):
        data = self.__loader__(self.data_path[index])
        if self.if_aug:
            data = self.augmentor(data)
        mask = np.clip(
            (data > 0.1375).astype('float') * (data < 0.3375).astype('float')
            + (data > 0.5375).astype('float'), 0, 1)
        mask = gaussian_filter(mask, sigma=3)
        data = np.stack([data, data*mask]).astype('float32')
        return data, self.label[index]

    def __len__(self):
        return len(self.label)


class SoftmaxSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = random.sample(
                range(self.dataset.data_size),
                self.batch_size)
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
