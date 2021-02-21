# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/12/10

# Heatmap Visualization with Grad-CAM

import torch.nn as nn

from net import AttBranch, Branch


class GradCam(nn.Module):
    def __init__(self, model):
        super(GradCam, self).__init__()
        self.model = model
        self.model.eval()

        for m in self.model.modules():
            if isinstance(m, AttBranch):
                for param in m.parameters():
                    param.requires_grad = False
            if isinstance(m, Branch):
                for _layer in m.backbone2d[:-1]:
                    for param in _layer.parameters():
                        param.requires_grad = False

        self.regist()
        self.clean()

    def clean(self):
        self.axial_output = None
        self.coronal_output = None
        self.sagittal_output = None
        self.axial_grad = None
        self.coronal_grad = None
        self.sagittal_grad = None
        self.f_output = None

    def regist(self):
        self.model.module.branch_axial.backbone2d.register_backward_hook(self.save_axial_grad)
        self.model.module.branch_axial.backbone2d.register_forward_hook(self.save_axial_output)
        self.model.module.branch_coronal.backbone2d.register_backward_hook(self.save_coronal_grad)
        self.model.module.branch_coronal.backbone2d.register_forward_hook(self.save_coronal_output)
        self.model.module.branch_sagittal.backbone2d.register_backward_hook(self.save_sagittal_grad)
        self.model.module.branch_sagittal.backbone2d.register_forward_hook(self.save_sagittal_output)

    def save_axial_grad(self, model, grad_input, grad_output):
        self.axial_grad = grad_output

    def save_coronal_grad(self, model, grad_input, grad_output):
        self.coronal_grad = grad_output

    def save_sagittal_grad(self, model, grad_input, grad_output):
        self.sagittal_grad = grad_output

    def save_axial_output(self, model, input, output):
        self.axial_output = output

    def save_coronal_output(self, model, input, output):
        self.coronal_output = output

    def save_sagittal_output(self, model, input, output):
        self.sagittal_output = output

    def forward(self, input):
        self.clean()
        return self.model(input)[0]

    def get_intermediate_data(self):
        return (self.axial_output, self.coronal_output, self.sagittal_output,
                self.axial_grad, self.coronal_grad, self.sagittal_grad)
