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
        bandwidth_list = [bandwidth * (kernel_mul ** _i) for _i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def forward_linear(f_of_x, f_of_y):
        delta = f_of_x.float().mean(0) - f_of_y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
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


class CoralLoss2(nn.Module):

    def __init__(self, is_sqrt=False):
        """
        Coral loss implement for eq(1, 2, 3) in https://arxiv.org/pdf/1607.01719.pdf
        Args:
            is_sqrt:
        """
        super().__init__()
        self.is_sqrt = is_sqrt

    def forward(self, source, target):
        d = source.size(1)
        ns, nt = source.size(0), target.size(0)

        # source covariance
        tmp_s = torch.ones((1, ns)).cuda() @ source
        cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

        # target covariance
        tmp_t = torch.ones((1, nt)).cuda() @ target
        ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

        # frobenius norm
        loss = (cs - ct).pow(2).sum()
        loss = loss.sqrt() if self.is_sqrt else loss
        loss = loss / (4 * d * d)

        return loss


class CoralLoss(nn.Module):

    def __init__(self, is_sqrt=False):
        """
        Coral loss implement for eq(1) in https://arxiv.org/pdf/1607.01719.pdf
        Args:
            is_sqrt:
        """
        super().__init__()
        self.is_sqrt = is_sqrt

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
        loss = (loss.sqrt() if self.is_sqrt else loss) / (4 * d * d)
        return loss


class DownscaleLabel(nn.Module):

    def __init__(self, scale_factor=16, n_classes=7, ignore_index=-1, min_ratio=0.75):
        super().__init__()
        assert scale_factor > 1
        self.scale_factor = scale_factor
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.min_ratio = min_ratio

    def forward(self, label):
        label = label.clone()
        if len(label.shape) == 3:
            label = label.unsqueeze(dim=1)
        bs, orig_c, orig_h, orig_w = label.shape
        assert orig_c == 1
        trg_h, trg_w = orig_h // self.scale_factor, orig_w // self.scale_factor
        label[label == self.ignore_index] = self.n_classes
        out = nnf.one_hot(label.squeeze(1), num_classes=self.n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, self.n_classes + 1, orig_h, orig_w], out.shape
        out = nnf.avg_pool2d(out.float(), kernel_size=self.scale_factor)
        max_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == self.n_classes] = self.ignore_index
        out[max_ratio < self.min_ratio] = self.ignore_index
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out


class Aligner:

    def __init__(self, logger, feat_channels=64, class_num=7, ignore_label=-1):
        # self.mmd = MMDLoss()
        self.feat_channels = feat_channels
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.logger = logger

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

        self.coral = CoralLoss()
        self.mmd = MMDLoss(kernel_type='rbf')
        self.downscale_gt = DownscaleLabel(scale_factor=16,
                                           n_classes=7,
                                           ignore_index=-1,
                                           min_ratio=0.75)

    def update_class_prototypes(self, feat, label, is_source=True):
        feat, label = feat.cuda(), label.cuda()
        # label = nnf.interpolate(torch.unsqueeze(label.float(), dim=1), size=feat.shape[-2:])
        label = self.downscale_gt(label)
        label = label.expand(*feat.shape)
        u_list = []
        float_zero = torch.tensor(0).float().cuda()
        for class_i in range(self.class_num):
            class_i_feat = torch.where(label == class_i, feat, float_zero)
            class_i_num = (label == class_i).sum() / self.feat_channels
            class_i_sum = torch.sum(class_i_feat, dim=[0, 2, 3], keepdim=False).unsqueeze(dim=0)    # [1, channel_num]
            if class_i_num <= 0:
                # self.logger.info(f"class {class_i} has no elements")
                if is_source:
                    u_list.append(self.class_u_s[class_i: class_i + 1, :].detach())
                else:
                    u_list.append(self.class_u_t[class_i: class_i + 1, :].detach())
            else:
                u_list.append(class_i_sum / class_i_num)
        u_s = torch.cat(u_list, dim=0)  # (class_num, feat_channels)
        if is_source:
            self.class_u_s = self.ema(self.class_u_s.detach(), u_s, decay=0.99)
        else:
            self.class_u_t = self.ema(self.class_u_t.detach(), u_s, decay=0.99)
        return u_s

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        feat_s = torch.mean(feat_s, dim=[2, 3])
        feat_t = torch.mean(feat_t, dim=[2, 3])
        return self.mmd(feat_s, feat_t)

    def align_domain_all_example(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        feat_s = feat_s.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        feat_t = feat_t.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        return self.coral(feat_s, feat_t)

    def align_domain_0(self, feat_s, feat_t):
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

    def align_class(self, feat_s, label_s, feat_t, label_t):
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
        self.update_class_prototypes(feat_s, label_s, is_source=True)
        self.update_class_prototypes(feat_t, label_t, is_source=False)
        return nnf.mse_loss(self.class_u_t, self.class_u_s.detach())
        # return -1 * torch.mean(nnf.cosine_similarity(self.class_u_t, self.class_u_s.detach()))

    def align_category_0(self, feat_s, label_s, feat_t, label_t):
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

    def show(self, save_path=None, display=True):

        pass


if __name__ == '__main__':
    from module.Encoder import Deeplabv2
    import torch.optim as optim
    import logging
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

    aligner = Aligner(logger=logging.getLogger(''), feat_channels=2048, class_num=7)

    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).cuda() * 255, \
               torch.randn([8, 3, 512, 512]).cuda() * 255, \
               torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
               torch.randint(1, 2, [8, 512, 512]).long().cuda()
    # x_s = torch.randn([8, 3, 512, 512]).cuda() * 255
    # x_t = torch.randn([8, 3, 512, 512]).cuda() * 255
    # l_s = torch.randint(0, 1, [8, 512, 512]).long().cuda()
    # l_t = torch.randint(1, 2, [8, 512, 512]).long().cuda()
    # x_s, x_t, l_s, l_t = rand_x_l()
    # zero_count = (l_s == 0).sum() / 16 / 16

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        loss_class = aligner.align_class(f_s, l_s, f_t, l_t)
        print(loss_class)
        optimizer.zero_grad()
        loss_class.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss class')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        loss_domain = aligner.align_domain(f_s, f_t)
        print(loss_domain)
        optimizer.zero_grad()
        loss_domain.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss domain')
    print('=========================================================')

    from utils.tools import loss_calc
    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        os1, os2, f_s = model(x_s)
        ot1, ot2, f_t = model(x_t)
        loss_seg = loss_calc([os1, os2], l_s, multi=True)
        loss_seg += loss_calc([ot1, ot2], l_t, multi=True)
        print(loss_seg)
        optimizer.zero_grad()
        loss_seg.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss seg')
    print('=========================================================')
    print('end')
