"""
    @Project: pytorch-learning-tutorials
    @File   : fashion_mnist_cnn.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 16:46:53
    @url: https://panjinquan.blog.csdn.net/article/details/88426648
"""

import torch
import torch.nn as nn


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, position, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.pos = position
        self.weight_list = self.get_weight(model)
        self.weight_names = []
        for name, w in self.weight_list:
            self.weight_names.append(name)

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        if self.pos == 'Wcom_last':
            for name, param in model.named_parameters():
                if ('weight' in name) and ('conv2d_last' in name):
                    weight = (name, param)
                    weight_list.append(weight)
        elif self.pos == 'Gconv_last':
            for name, param in model.named_parameters():
                if ('weight' in name) and ('fc' in name):
                    weight = (name, param)
                    weight_list.append(weight)
        elif self.pos == 'final':
            for name, param in model.named_parameters():
                if ('weight' in name) and ('layer_last.fc' in name):
                    weight = (name, param)
                    weight_list.append(weight)
        elif self.pos == 'all':
            for name, param in model.named_parameters():
                if 'weight' in name:  # 所有权重列表
                    weight = (name, param)
                    weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=1):
        reg_loss = 0
        for name, w in weight_list:
            l1_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l1_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

