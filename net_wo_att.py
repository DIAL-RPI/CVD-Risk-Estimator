# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/29

import torch
import torch.nn as nn
import torchvision.models as models


class Branch(nn.Module):
    def __init__(self, num_classes=2, dout=False):
        super(Branch, self).__init__()
        _net = models.resnet18()
        _net_list = list(_net.children())
        self.backbone2d = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, dilation=2, padding=2, bias=False),
            *_net_list[1:-3])
        self.aux = nn.Sequential(
            *_net_list[-3:-1],
            nn.Flatten())
        self.fc = nn.Linear(512, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()
        # print(self.dout)

    def forward(self, x):
        n, d, h, w = x.size()
        x = x.view(n * d, 1, h, w).contiguous()
        x = self.backbone2d(x)
        _, c, h, w = x.size()
        x = x.view(n, d, c, h, w)
        # -> n, c, d, h, w
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        aux_feature = x.max(dim=2)[0]
        aux_feature = self.aux(aux_feature)
        aux_feature = aux_feature / aux_feature.norm(dim=1, keepdim=True)
        if self.dout is not None:
            aux_feature = self.dout(aux_feature)
        aux_pred = self.fc(aux_feature)

        return aux_pred, aux_feature


class Tri2DNet(nn.Module):
    def __init__(self, num_classes=2, dout=False):
        super(Tri2DNet, self).__init__()
        self.num_classes = num_classes
        self.branch_axial = Branch(num_classes, dout)
        self.branch_sagittal = Branch(num_classes, dout)
        self.branch_coronal = Branch(num_classes, dout)
        self.fc_fuse = nn.Linear(512 * 3, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, d, h, w = x.size()
        # -> n, w, d, h
        x_sagittal = x.permute(0, 3, 2, 1).contiguous()
        # -> n, h, d, w
        x_coronal = x.permute(0, 2, 1, 3).contiguous()
        x_axial = x
        del x
        aux_pred_sagittal, aux_feature_sagittal = self.branch_sagittal(x_sagittal)
        aux_pred_coronal, aux_feature_coronal = self.branch_coronal(x_coronal)
        aux_pred_axial, aux_feature_axial = self.branch_axial(x_axial)
        feature = torch.cat([aux_feature_sagittal, aux_feature_coronal, aux_feature_axial], dim=1)
        feature = feature / feature.norm(dim=1, keepdim=True)
        pred = self.fc_fuse(feature)

        return pred, aux_pred_sagittal, aux_pred_coronal, aux_pred_axial
