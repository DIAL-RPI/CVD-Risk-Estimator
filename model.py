# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import os.path as osp
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
# from apex import amp
from scipy.special import softmax

from data import SoftmaxSampler
from net import Net3x2D


class Model:
    def __init__(
            self,
            dout,
            lr,
            num_workers,
            batch_size,
            restore_iter,
            total_iter,
            save_name,
            model_name,
            train_source,
            val_source,
            test_source, ):

        self.dout = dout
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.save_name = save_name
        self.model_name = model_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source

        encoder = Net3x2D(dout=self.dout).cuda()
        ce = nn.CrossEntropyLoss(reduction='none').cuda()

        optimizer = optim.Adam([
            {'params': encoder.parameters()},
        ], lr=self.lr)
        models = [encoder, ce]
        #         models, optimizer = amp.initialize(models, optimizer, opt_level="O1")
        self.encoder = nn.DataParallel(models[0])
        self.ce = nn.DataParallel(models[1])
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [8000, 11000, 14000, 17000], gamma=0.5)

        self.loss = []
        self.m_loss = []
        self.td_loss = []
        self.sa_loss = []
        self.co_loss = []
        self.ax_loss = []
        self.f_list = []
        self.label_list = []
        self.LOSS = []

    def fit(self):
        if self.restore_iter != 0:
            self.load_model()

        self.encoder.train()

        softmax_sampler = SoftmaxSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=softmax_sampler,
            num_workers=self.num_workers)

        _time1 = datetime.now()
        for volumes, labels in train_loader:
            if self.restore_iter > self.total_iter:
                break
            self.restore_iter += 1
            self.optimizer.zero_grad()

            (pred, aux_pred_sagittal, aux_pred_coronal,
             aux_pred_axial) = self.encoder(volumes.cuda())

            labels = (labels > 0).int()
            main_ce_loss = self.ce(pred, labels.cuda().long()).mean()
            sagittal_ce_loss = self.ce(aux_pred_sagittal, labels.cuda().long()).mean()
            axial_ce_loss = self.ce(aux_pred_axial, labels.cuda().long()).mean()
            coronal_ce_loss = self.ce(aux_pred_coronal, labels.cuda().long()).mean()
            total_loss = (main_ce_loss + sagittal_ce_loss + axial_ce_loss + coronal_ce_loss) / 4
            _total_loss = total_loss.cpu().data.numpy()
            self.loss.append(_total_loss)
            self.m_loss.append(main_ce_loss.cpu().data.numpy())
            self.sa_loss.append(sagittal_ce_loss.cpu().data.numpy())
            self.co_loss.append(coronal_ce_loss.cpu().data.numpy())
            self.ax_loss.append(axial_ce_loss.cpu().data.numpy())

            if _total_loss > 1e-9:
                # with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                total_loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if self.restore_iter % 100 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save_model()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', loss={0:.8f}'.format(np.mean(self.loss)), end='')
                print(', m_loss={0:.8f}'.format(np.mean(self.m_loss)), end='')
                print(', sa_loss={0:.8f}'.format(np.mean(self.sa_loss)), end='')
                print(', co_loss={0:.8f}'.format(np.mean(self.co_loss)), end='')
                print(', ax_loss={0:.8f}'.format(np.mean(self.ax_loss)), end='')
                print(', lr=', end='')
                print([self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))])
                sys.stdout.flush()

                self.LOSS.append(np.mean(self.loss))
                self.loss = []
                self.m_loss = []
                self.td_loss = []
                self.sa_loss = []
                self.co_loss = []
                self.ax_loss = []

            if self.restore_iter % 1000 == 0:
                plt.plot(self.LOSS)
                plt.show()

    def aug_test(self, subset='test', batch_size=1):
        self.encoder.eval()
        assert subset in ['train', 'val', 'test']
        source = self.test_source
        if subset == 'train':
            source = self.train_source
        elif subset == 'val':
            source = self.val_source
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            num_workers=self.num_workers)

        pred_list = list()
        label_list = list()
        crop = []

        def get_crop(_c):
            if len(_c) == 3:
                crop.append(_c)
                return
            else:
                for i in [0, 16]:
                    get_crop(_c + [i])

        get_crop([])

        with torch.no_grad():
            for i, x in enumerate(data_loader):
                volumes, labels = x
                volumes = volumes.cuda()
                b_s = volumes.size()[0]
                _v = []
                for _c in crop:
                    s = _c[0]
                    h = _c[1]
                    w = _c[2]
                    _v.append(volumes[:, s:s + 112, h:h + 112, w:w + 112])
                _v = torch.cat(_v, 0).contiguous()
                (pred, aux_pred_sagittal, aux_pred_coronal,
                 aux_pred_axial) = self.encoder(_v)
                pred = pred.view(len(crop), b_s, 2)
                pred_prob = softmax(pred.data.cpu().numpy(), axis=2).mean(axis=0)
                pred_list.append(pred_prob)
                label_list.append(labels.numpy())

        pred_list = np.concatenate(pred_list, 0)
        label_list = np.concatenate(label_list, 0)

        return pred_list, label_list

    def aug_transform(self, volumes):
        if isinstance(volumes, np.ndarray):
            volumes = torch.from_numpy(volumes)
        self.encoder.eval()

        crop = []

        def get_crop(_c):
            if len(_c) == 3:
                crop.append(_c)
                return
            else:
                for i in [0, 16]:
                    get_crop(_c + [i])

        get_crop([])

        with torch.no_grad():
            volumes = volumes.cuda()
            volumes = volumes.unsqueeze(0)
            _v = []
            for _c in crop:
                s = _c[0]
                h = _c[1]
                w = _c[2]
                _v.append(volumes[:, s:s + 112, h:h + 112, w:w + 112])
            _v = torch.cat(_v, 0).contiguous()
            (pred, aux_pred_sagittal, aux_pred_coronal,
             aux_pred_axial) = self.encoder(_v)
            pred = pred.view(len(crop), 2)
            pred_prob = softmax(pred.data.cpu().numpy(), axis=1).mean(axis=0)

        return pred_prob

    def save_model(self):
        torch.save(self.encoder.state_dict(), osp.join(
            'checkpoint',
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(), osp.join(
            'checkpoint',
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, self.restore_iter)))

    def load_model(self, restore_iter=None):
        if restore_iter is None:
            restore_iter = self.restore_iter
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint',
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        opt_path = osp.join(
            'checkpoint',
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))
        if osp.isfile(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path))
