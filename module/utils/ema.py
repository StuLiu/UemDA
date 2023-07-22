"""
@Project : Unsupervised_Domian_Adaptation
@File    : ema.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/7/22 上午11:48
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn


class ExponentialMovingAverage():
    def __init__(self, decay):
        self.decay = decay
        self.shadow_params = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] -= (1 - self.decay) * (self.shadow_params[name] - param.data)

    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow_params[name]
