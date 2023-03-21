"""
@Project : Unsupervised_Domian_Adaptation
@File    : alignment.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/13 下午9:43
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.ext_params = kwargs

    @staticmethod
    def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def forward_linear(f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        """

        Args:
            source: features from source domain, shape as [batch_size, 1]
            target: features from target domain, shape as [batch_size, 1]

        Returns:

        """
        if self.kernel_type == 'linear':
            return self.forward_linear(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            xx = torch.mean(kernels[:batch_size, :batch_size])
            yy = torch.mean(kernels[batch_size:, batch_size:])
            xy = torch.mean(kernels[:batch_size, batch_size:])
            yx = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(xx + yy - xy - yx)
            return loss


class Aligner:

    def __init__(self, feat_channels=64, class_num=7, ignore_label=-1):
        # self.mmd = MMDLoss()
        self.feat_channels = feat_channels
        self.class_num = class_num
        self.ignore_label = ignore_label
        # statistics of domains
        self.domain_u_s = torch.zeros([feat_channels], requires_grad=False).cuda()
        self.domain_u_t = torch.zeros([feat_channels], requires_grad=False).cuda()
        self.domain_sigma_s = torch.zeros([feat_channels], requires_grad=False).cuda()
        self.domain_sigma_t = torch.zeros([feat_channels], requires_grad=False).cuda()
        # statistics of classes
        self.class_u_s = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()
        self.class_u_t = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()
        self.class_sigma_s = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()
        self.class_sigma_t = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        u_s = torch.mean(feat_s, dim=[0, 2, 3], keepdim=False)  # (feat_channels,)
        u_t = torch.mean(feat_t, dim=[0, 2, 3], keepdim=False)  # (feat_channels,)
        sigma_s = torch.var(feat_s, dim=[0, 2, 3], unbiased=True)  # (feat_channels,)
        sigma_t = torch.var(feat_t, dim=[0, 2, 3], unbiased=True)  # (feat_channels,)
        # update the statistics of domains
        self.domain_u_s = self.ema(self.domain_u_s, u_s)
        self.domain_u_t = self.ema(self.domain_u_t, u_t)
        self.domain_sigma_s = self.ema(self.domain_sigma_s, sigma_s)
        self.domain_sigma_t = self.ema(self.domain_sigma_t, sigma_t)
        # compute loss and return
        return nnf.mse_loss(u_s, u_t) + nnf.mse_loss(sigma_s, sigma_t)

    def align_category(self, feat_s, label_s, feat_t, label_t):
        """ Compute the loss for discrepancy between class distribution.

        Args:
            feat_s:  features from source, shape as (batch_size, feature_channels, height, width)
            label_s: labels from source  , shape as (batch_size, height, width)
            feat_t:  features from source, shape as (batch_size, feature_channels, height, width)
            label_t: pseudo labels from target, shape as (batch_size, height, width)

        Returns:
            loss for discrepancy between class distribution.
        """
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        assert label_s.shape == label_t.shape, 'tensor "label_s" has the same shape as tensor "label_t"'
        assert len(label_s.shape) == 3, 'tensor "label_s" and "feat_t" must have 3 dimensions'
        # compute statistics within the mini-batch
        feat_s, label_s = feat_s.cuda(), label_s.cuda()
        feat_t, label_t = feat_t.cuda(), label_t.cuda()
        label_s = nnf.interpolate(torch.unsqueeze(label_s.float(), dim=1), size=feat_s.shape[-2:])
        label_t = nnf.interpolate(torch.unsqueeze(label_t.float(), dim=1), size=feat_t.shape[-2:])
        label_s = label_s.expand(*feat_s.shape)
        label_t = label_t.expand(*feat_t.shape)
        u_s_list, u_t_list, sigma_s_list, sigma_t_list = [], [], [], []
        float_zero = torch.tensor(0).float().cuda()
        for class_i in range(self.class_num):
            class_i_feat_s = torch.where(label_s == class_i, feat_s, float_zero)
            class_i_feat_t = torch.where(label_t == class_i, feat_t, float_zero)
            u_s_list.append(torch.mean(class_i_feat_s, dim=[0, 2, 3], keepdim=False).unsqueeze(dim=0))
            u_t_list.append(torch.mean(class_i_feat_t, dim=[0, 2, 3], keepdim=False).unsqueeze(dim=0))
            sigma_s_list.append(torch.var(class_i_feat_s, dim=[0, 2, 3], unbiased=True).unsqueeze(dim=0))
            sigma_t_list.append(torch.var(class_i_feat_t, dim=[0, 2, 3], unbiased=True).unsqueeze(dim=0))
        u_s = torch.cat(u_s_list, dim=0)  # (class_num, feat_channels)
        u_t = torch.cat(u_t_list, dim=0)  # (class_num, feat_channels)
        sigma_s = torch.cat(sigma_s_list, dim=0)  # (class_num, feat_channels)
        sigma_t = torch.cat(sigma_t_list, dim=0)  # (class_num, feat_channels)
        # update the statistics of classes
        self.class_u_s = self.ema(self.class_u_s, u_s)
        self.class_u_t = self.ema(self.class_u_t, u_t)
        self.class_sigma_s = self.ema(self.class_sigma_s, sigma_s)
        self.class_sigma_t = self.ema(self.class_sigma_t, sigma_t)
        # compute loss and return
        loss = -1 * torch.mean(nnf.cosine_similarity(u_s, u_t, dim=1))
        loss += torch.mean(sigma_s) + torch.mean(sigma_t)
        return loss

    @staticmethod
    def align_instance(feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'

    @staticmethod
    def ema(history, curr, decay=0.99):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average

    def display(self):
        pass


if __name__ == '__main__':
    from module.Encoder import Deeplabv2
    import torch.optim as optim
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    aligner = Aligner(feat_channels=2048, class_num=7)
    x_s = torch.randn([8, 3, 512, 512]).cuda() * 2
    x_t = torch.randn([8, 3, 512, 512]).cuda() + 1
    l_s = torch.randint(0, 7, [8, 512, 512]).long().cuda()
    l_t = torch.randint(0, 7, [8, 512, 512]).long().cuda()
    # f_s = torch.randn([8, 128]).cuda() * 1
    # f_t = torch.randn([8, 128]).cuda()
    _, _, f_s = model(x_s)
    _, _, f_t = model(x_t)
    print(f_s.device)
    loss_domain = aligner.align_domain(f_s, f_t)
    loss_class = aligner.align_category(f_s, l_s, f_t, l_t)
    optimizer.zero_grad()
    # loss_domain.backward()
    loss_class.backward()
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
        print(param.grad)
        break
    #
    # # print(aligner.align_domain(f_s, f_t))
    # print(aligner.align_category(f_s, l_s, f_t, l_t))
    # pass
