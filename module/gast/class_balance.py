"""
@Project : Unsupervised_Domian_Adaptation
@File    : class_balance.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/5 下午3:59
@e-mail  : liuwa@hnu.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import math


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_label=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, preds, targets):
        ce_loss = tnf.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_label)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets].view(-1, 1)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GHMLoss(nn.Module):

    def __init__(self, bins=30, momentum=0.0, ignore_label=-1):
        super(GHMLoss, self).__init__()
        self.bins_num = bins
        self.momentum = momentum
        self.ignore_label = ignore_label
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = self.edges[-1] + 1e-3
        self.edges = torch.FloatTensor(self.edges).cuda()
        self.acc_sum = torch.zeros(bins).cuda()

    def forward(self, preds, targets):
        # Compute the number of classes
        n_classes = preds.size(1)

        # Flatten the prediction and target tensors
        preds = preds.permute((0, 2, 3, 1)).reshape(-1, n_classes)
        probs = torch.softmax(preds, dim=1)
        targets = targets.view(-1)

        # Convert id labels to one_hot
        labels = targets.clone().detach()
        labels[labels == self.ignore_label] = n_classes
        labels_onehot = tnf.one_hot(labels, num_classes=n_classes + 1)[:, : -1]

        # Calculate the gradient of the prediction
        prob_y = torch.sum(probs * labels_onehot, dim=1)
        gradient = torch.abs(prob_y - 1.0)
        gradient[targets == self.ignore_label] = -1      # ignore the invalid or ignored targets

        # Sort the gradient and prediction values
        bins = torch.histc(gradient, bins=self.bins_num, min=0, max=1)  # out-bounded values will not be statistic
        inds = torch.bucketize(gradient, self.edges)  # lower than min will be 0, lager than max will be len(bins)

        # Calculate the weights for each sample based on the gradient
        weights = torch.zeros_like(gradient).cuda()
        # print(f'inds.min={inds.min()}, inds.max={inds.max()}, g.min={gradient.min()}, '
        #       f'g.max={gradient.max()}, target-range={torch.unique(targets)}')
        if self.momentum > 0:
            self.acc_sum = self.momentum * self.acc_sum.detach() + (1 - self.momentum) * bins
        else:
            self.acc_sum = bins
        weights = torch.where((inds > 0) & (inds <= self.bins_num).to(torch.bool),
                              1.0 / (self.acc_sum[inds - 1]),
                              weights)
        weights = weights.detach()
        # Calculate the GHM loss
        loss = tnf.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_label)
        loss = loss * weights
        loss = loss.sum() / (weights.sum() + 1e-7)
        return loss

    def get_g_distribution(self):
        return self.acc_sum / (self.acc_sum.sum() + 1e-7)


class ClassBalance(nn.Module):

    def __init__(self, class_num=7, ignore_label=-1, decay=0.99, temperature=0.5):
        super().__init__()
        assert temperature > 0
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.decay = decay
        self.temperature = temperature
        self.eps = 1e-7
        self.freq = torch.ones([class_num]).float().cuda() / class_num

    def get_class_weight_4pixel(self, label):
        self.ema_update(label)
        label_onehot = self._one_hot(label)  # (b*h*w, c)
        # loss weight computed by class frequency
        class_prob = self._get_class_wight()  # (c,)
        weight = (label_onehot * class_prob.unsqueeze(dim=0)).sum(dim=1)  # (b*h*w,)
        return weight.detach()  # (b*h*w, )

    def ema_update(self, label):
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

    def _get_class_wight(self):
        _prob = (1.0 - self.freq) / self.temperature
        _prob = torch.softmax(_prob, dim=0)
        _max, _ = torch.max(_prob, dim=0, keepdim=True)
        prob_normed = _prob / (_max + self.eps)  # 0 ~ 1
        return prob_normed

    def _local_freq(self, label):
        lbl = label.clone()
        if len(lbl.shape) > 3:
            lbl = torch.squeeze(lbl, dim=1)
        local_cnt = torch.sum((lbl != self.ignore_label).float())
        label_onehot = self._one_hot(label)
        class_cnt = label_onehot.sum(dim=0).float()
        class_freq = class_cnt / (local_cnt + self.eps)
        return class_freq

    @staticmethod
    def _ema(history, curr, decay):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average

    def _one_hot(self, label):
        label = label.clone()
        if len(label.shape) > 3:
            label = label.squeeze(dim=1)
        local_cnt = torch.numel(label)
        label[label == self.ignore_label] = self.class_num
        label_onehot = tnf.one_hot(label, num_classes=self.class_num + 1).view(local_cnt, -1)[:, : -1]
        return label_onehot  # (b*h*w, c)

    def __str__(self):
        freq = self.freq.cpu().numpy()
        res = f'class frequency: {freq[0]:.3f}'
        for _freq in freq[1:]:
            res += f', {_freq:.3f}'
        freq = self._get_class_wight()
        res += f';\tselect probability: {freq[0]:.3f}'
        for _freq in freq[1:]:
            res += f', {_freq:.3f}'
        return res


class GDPLoss(nn.Module):

    def __init__(self, bins=30, momentum=0.99, class_num=7, ignore_label=-1,
                 class_balance=False, prototype_refine=False, temp=0.5):
        super(GDPLoss, self).__init__()
        self.bins_num = bins
        self.momentum = momentum
        self.ignore_label = ignore_label
        self.class_balance = class_balance
        self.prototype_refine = prototype_refine
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = self.edges[-1] + 1e-3
        self.edges = torch.FloatTensor(self.edges).cuda()
        self.acc_sum = torch.zeros(bins).cuda()
        self.bins_weight = None

        # for each pixel
        self.weight_bins = None
        if prototype_refine:
            self.weight_prototype = None
        self.class_balancer = ClassBalance(class_num=class_num, ignore_label=ignore_label, decay=0.99,
                                           temperature=temp)

    def forward(self, preds, targets):
        # Compute the number of classes
        n_classes = preds.size(1)

        # Flatten the prediction and target tensors
        preds = preds.permute((0, 2, 3, 1)).reshape(-1, n_classes)
        probs = torch.softmax(preds, dim=1)
        targets = targets.view(-1)

        # Convert id labels to one_hot
        labels = targets.clone().detach()
        labels[labels == self.ignore_label] = n_classes
        labels_onehot = tnf.one_hot(labels, num_classes=n_classes + 1)[:, : -1]

        # Calculate the gradient of the prediction
        prob_y = torch.sum(probs * labels_onehot, dim=1)
        gradient = torch.abs(prob_y - 1.0)
        gradient[targets == self.ignore_label] = -1      # ignore the invalid or ignored targets

        # Sort the gradient and prediction values
        bins = torch.histc(gradient, bins=self.bins_num, min=0, max=1)  # out-bounded values will not be statistic
        bins = (bins + torch.flip(bins, dims=[0])) * 0.5
        inds = torch.bucketize(gradient, self.edges)  # lower than min will be 0, lager than max will be len(bins)

        # print(f'inds.min={inds.min()}, inds.max={inds.max()}, g.min={gradient.min()}, '
        #       f'g.max={gradient.max()}, target-range={torch.unique(targets)}')
        if self.momentum > 0:
            self.acc_sum = self.momentum * self.acc_sum.detach() + (1 - self.momentum) * bins
        else:
            self.acc_sum = bins

        # Calculate the weights for each sample based on the gradient
        weight_bins = self._get_dense_weight(self.acc_sum, inds)

        # Calculate the gdp loss
        loss = tnf.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_label)
        weight_pixels = weight_bins
        if self.prototype_refine:
            weight_pixels = weight_pixels + self.weight_prototype
        if self.class_balance:
            weight_pixels = weight_pixels + self.class_balancer.get_class_weight_4pixel(targets)
        # print((1.0 + int(self.prototype_refine) + int(self.class_balance)))
        loss = loss * weight_pixels / (1.0 + int(self.prototype_refine) + int(self.class_balance))
        loss = loss.sum() / (torch.sum(targets != -1) + 1e-7)
        return loss

    def set_prototype_weight_4pixel(self, weight_prototype):
        self.weight_prototype = weight_prototype

    def _get_dense_weight(self, bins, inds):
        cond = (bins != 0).to(torch.bool)
        _bins = 1 - bins / (bins.sum() + 1e-7)
        _bins = torch.where(cond, _bins, torch.zeros_like(bins).cuda())
        _bins = _bins / (_bins.max() + 1e-7)
        self.bins_weight = _bins

        # calculate
        weight_bins = torch.zeros_like(inds, dtype=torch.float).cuda()
        weight_bins = torch.where((inds > 0) & (inds <= self.bins_num).to(torch.bool), _bins[inds - 1], weight_bins)
        return weight_bins.detach()

    def get_g_distribution(self):
        return self.acc_sum / (self.acc_sum.sum() + 1e-7), self.bins_weight, str(self.class_balancer)


class UVEMLoss(nn.Module):

    def __init__(self, m=0.1, threshold=0.75, gamma=8.0, class_balance=False, temp=0.5, class_num=7, ignore_label=-1):
        super(UVEMLoss, self).__init__()
        self.m = m
        self.threshold = threshold
        self.gamma = gamma
        self.class_balancer = None
        if class_balance:
            self.class_balancer = ClassBalance(class_num=class_num, ignore_label=ignore_label, decay=0.99,
                                               temperature=temp)
        self.class_num = class_num
        self.ignore_label = ignore_label

    def forward(self, preds, targets, label_t_soft):
        """
        Args:
            preds: Tensor, (b, c, h, w), without softmax
            targets: Tensor, (b, h, w)
            label_t_soft: Tensor, (b, c, h, w), softmax-ed
        Returns:
            loss: Tensor, (1,)
        """
        # Flatten the prediction and target tensors
        preds_ = preds.permute((0, 2, 3, 1)).reshape(-1, self.class_num)        # (-1, c)
        targets_ = targets.view(-1)                                             # (-1,)
        lts_ = label_t_soft.permute((0, 2, 3, 1)).reshape(-1, self.class_num)   # (-1,)
        # original cross-entropy loss
        ce_loss = tnf.cross_entropy(preds_, targets_, reduction='none', ignore_index=self.ignore_label)     # (-1,)
        # >>>>> uncertainty-based valuable example mining(UVEM)
        # cls = preds_
        uncertainty = torch.sum(-lts_ * torch.log(lts_), dim=1).detach()      # (-1,)
        # ce_loss[uncertainty > self.threshold] = 0         # gated uncertain example removing
        weight_uncer = self.get_weight(uncertainty)         # get example weight, (-1,)
        # compute loss
        loss = weight_uncer * ce_loss
        valid_cnt = torch.sum((uncertainty <= self.threshold) & (targets_ != self.ignore_label))
        loss = loss.sum() / (valid_cnt + 1e-7)
        # <<<<<<
        return loss

    def get_weight(self, uncertainties):
        unce_ = uncertainties.clone()
        unce_[unce_ > self.threshold] = self.threshold      # gated uncertain example removing
        weight_left = (-1 / (self.m ** 2)) * (unce_ - self.m) ** 2 + 1
        weight_right = (-1 / ((self.threshold - self.m) ** 2)) * (unce_ - self.m) ** 2 + 1
        weight = torch.where(unce_ < self.m,
                             weight_left ** (1.0 / self.gamma),
                             weight_right ** (1.0 / self.gamma))
        return weight

    def drow_weight_curve(self):
        import numpy as np
        import matplotlib.pyplot as plt
        n = 1000.0
        unce = [i / n for i in range(int(n))]
        unce_ = torch.from_numpy(np.array(unce)).float().cuda()
        weight_unce = self.get_weight(unce_)
        plt.plot(unce, weight_unce.cpu().numpy(), "r-", label="weight")  # "r"为红色, "s"为方块, "-"为实线
        plt.show(block=True)


def loss_calc_uvem(pred, label, label_soft, loss_fn, multi=True):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    if multi is True:
        loss = 0
        num = 0
        for p in pred:
            if p.size()[-2:] != label.size()[-2:]:
                p = tnf.interpolate(p, size=label.size()[-2:], mode='bilinear', align_corners=True)
            # l = tnf.cross_entropy(p, label.long(), ignore_index=-1, reduction=reduction)
            loss += loss_fn(p, label.long(), label_soft)
            num += 1
        loss = loss / num
    else:
        if pred.size()[-2:] != label.size()[-2:]:
            pred = tnf.interpolate(pred, size=label.size()[-2:], mode='bilinear', align_corners=True)
        loss = loss_fn(pred, label.long(), label_soft)

    return loss


if __name__ == '__main__':
    # prob = torch.rand(8, 6, 512, 512).cuda()
    import cv2
    prob = torch.load('../../log/GAST/2potsdam/pseudo_label/2_10_0_0_512_512.png.pt', map_location='cpu').cuda()
    prob = prob.unsqueeze(dim=0).repeat(8, 1, 1, 1).float()
    target = torch.from_numpy(cv2.imread('../../data/IsprsDA/Potsdam/ann_dir/train/2_10_0_0_512_512.png', cv2.IMREAD_UNCHANGED)).cuda()
    target = target.unsqueeze(dim=0).repeat(8, 1, 1).long()
    label_t_soft = prob.clone()
    FL = FocalLoss()
    # print('focal loss: ', FL(tnf.cross_entropy(prob, target, reduction='none'), target))

    gdp = GDPLoss(momentum=0.9, class_balance=True, prototype_refine=False)
    uvem = UVEMLoss(m=0.1, threshold=0.75, gamma=8.0, class_balance=False, temp=0.5, class_num=6, ignore_label=-1)
    print('UVEM loss: ', uvem(prob, target, label_t_soft))
    print('UVEM loss: ', loss_calc_uvem([prob, prob], target, label_t_soft, loss_fn=uvem, multi=True))
    uvem.drow_weight_curve()
    # print(gdp.get_g_distribution())
    #
    # def rand_x_l():
    #     return torch.randn([8, 3, 512, 512]).float().cuda(), \
    #         torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
    #         torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
    #         torch.randint(1, 2, [8, 512, 512]).long().cuda()
    #
    #
    # x_s, x_t, l_s, l_t = rand_x_l()
    #
    # cbl = ClassBalance(class_num=3)
    # _rand_val = torch.rand_like(l_t, dtype=torch.float).cuda()
    # for _ in range(1000):
    #     cbl.ema_update(l_t)
    # print(cbl)
