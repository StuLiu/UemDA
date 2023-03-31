"""
@Project : rsda
@File    : contrastive.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午8:53
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.5, class_num=7):
        super().__init__()
        self.margin = margin
        self.class_num = class_num

    def forward(self, feat: torch.Tensor, label: torch.Tensor, prototypes: torch.Tensor):
        """

        Args:
            feat: [n, k], n vectors with length k.
            label: [n,]
            prototypes:

        Returns:

        """
        assert len(feat.shape) == 2
        label = label.flat()

        for class_i in range(self.class_num):
            label_i = torch.where(label == class_i, 1, 0)

