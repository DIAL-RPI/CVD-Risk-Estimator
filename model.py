# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import os.path as osp
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tordata
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import gaussian_filter
# from apex import amp
from scipy.special import softmax
from skimage.transform import resize as imresize

from data import SoftmaxSampler
from net import Tri2DNet, Branch
from visualization import GradCam


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
            test_source,
            accumulate_steps,
            prt_path, ):

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
        self.accumulate_steps = accumulate_steps
        self.prt_path = prt_path

        encoder = Tri2DNet(dout=self.dout).cuda()
        ce = nn.CrossEntropyLoss(reduction='none').cuda()

        att_id = []
        aux_id = []
        for m in encoder.modules():
            if isinstance(m, Branch):
                att_id += list(map(id, m.att_branch.parameters()))
                aux_id += list(map(id, m.aux.parameters()))
        pretrained_params = filter(
            lambda p: id(p) not in att_id + aux_id,
            encoder.parameters())
        aux_params = filter(
            lambda p: id(p) in aux_id,
            encoder.parameters())
        att_params = filter(
            lambda p: id(p) in att_id,
            encoder.parameters())
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': self.lr / 10},
            {'params': aux_params, 'lr': self.lr / 5},
            {'params': att_params, 'lr': self.lr},
        ], lr=self.lr)

        models = [encoder, ce]
        # models, optimizer = amp.initialize(models, optimizer, opt_level="O1")
        self.encoder = nn.DataParallel(models[0])
        self.ce = nn.DataParallel(models[1])
        self.optimizer = optimizer
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [8000], gamma=0.5)

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

        self.load_pretrain()
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

            total_loss = total_loss / self.accumulate_steps
            if _total_loss > 1e-9:
                # with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                total_loss.backward()
            if self.restore_iter % self.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
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
                    _v.append(volumes[:, :, s:s + 112, h:h + 112, w:w + 112])
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
                _v.append(volumes[:, :, s:s + 112, h:h + 112, w:w + 112])
            _v = torch.cat(_v, 0).contiguous()
            (pred, aux_pred_sagittal, aux_pred_coronal,
             aux_pred_axial) = self.encoder(_v)
            pred = pred.view(len(crop), 2)
            pred_prob = softmax(pred.data.cpu().numpy(), axis=1).mean(axis=0)

        return pred_prob

    def grad_cam_visual(self, volumes):
        if isinstance(volumes, np.ndarray):
            volumes = torch.from_numpy(volumes)
        self.encoder.eval()
        grad_cam = GradCam(self.encoder)

        color = cv2.COLORMAP_JET
        color_sample = np.asarray(list(range(0, 10)) * 3).reshape(3, 10)
        color_sample = imresize(color_sample.astype('float'), (12, 128))
        color_sample = color_sample / 10
        color_sample = cv2.applyColorMap(np.uint8(255 * color_sample), color)
        color_sample = cv2.cvtColor(color_sample,cv2.COLOR_BGR2RGB)
        plt.imshow(color_sample)
        plt.yticks(np.arange(0))
        plt.xticks(np.arange(-1, 128, 32), [0, 0.25, 0.5, 0.75, 1.0])
        plt.show()

        def show_cam_on_image(img, mask):
            img = np.float32(img) / 255
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), color)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap * .5 + np.float32(img) * .5
            cam = np.clip(cam * 1.1 + 0.15, 0, 1)
            cam = np.uint8(255 * cam)
            return cam

        def v_2D(output, grad):
            weight = grad.mean(dim=(2, 3))
            s, c = weight.size()
            cam = F.relu((weight.view(s, c, 1, 1) * output).sum(dim=1))
            cam = cam.data.cpu().numpy().astype('float')
            cam = imresize(cam, (128, 128, 128))
            return cam

        volumes = volumes.unsqueeze(0)
        grad_cam.model.zero_grad()
        pred = grad_cam(volumes.cuda())
        one_hot = torch.zeros(pred.size())
        one_hot[:, 1] = 1
        one_hot = one_hot.cuda().float()
        y = (one_hot * pred).sum()
        y.backward()

        (axial_output, coronal_output, sagittal_output,
         axial_grad, coronal_grad, sagittal_grad,
         ) = grad_cam.get_intermediate_data()
        # axial: d, h, w
        axial_cam = v_2D(axial_output, axial_grad[0])
        axial_cam = gaussian_filter(axial_cam, sigma=(3, 0, 0))
        # coronal: h, d, w
        coronal_cam = v_2D(coronal_output, coronal_grad[0])
        coronal_cam = np.transpose(coronal_cam, (1, 0, 2))
        coronal_cam = gaussian_filter(coronal_cam, sigma=(0, 3, 0))
        # sagittal: w, d, h
        sagittal_cam = v_2D(sagittal_output, sagittal_grad[0])
        sagittal_cam = np.transpose(sagittal_cam, (2, 1, 0))
        sagittal_cam = gaussian_filter(sagittal_cam, sigma=(0, 0, 3))

        cam_combine = axial_cam + coronal_cam + sagittal_cam
        cam_combine = (cam_combine - cam_combine.min()) / (cam_combine.max() - cam_combine.min() + 1e-9)

        _v = volumes.data.numpy()[0][0]
        total_img_num = _v.shape[0]
        fig = plt.figure(figsize=(15, 240))
        grid = ImageGrid(fig, 111, nrows_ncols=(32, 2), axes_pad=0.05)
        for i in range(64):
            frame_dix = i * int(total_img_num / 64)
            org_img = cv2.cvtColor(np.uint8(255 * _v[frame_dix].reshape(128, 128)), cv2.COLOR_GRAY2BGR)
            merged = show_cam_on_image(org_img, cam_combine[frame_dix])
            coupled = np.concatenate([org_img, merged], axis=1)
            coupled = cv2.cvtColor(coupled, cv2.COLOR_BGR2RGB)
            grid[i].imshow(coupled)
        plt.show()

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

    def load_pretrain(self):
        self.encoder.load_state_dict(torch.load(self.prt_path), False)
