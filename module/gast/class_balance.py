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


class ClassBalanceLoss(nn.Module):

    def __init__(self, class_num=7, ignore_label=-1, decay=0.998, min_prob=0.01, temperature=0.5, hard=True):
        super().__init__()
        assert temperature > 0
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.decay = decay
        self.min_prob = torch.FloatTensor([min_prob]).cuda()[0]
        self.temperature = temperature
        self.is_hard = hard
        self.eps = 1e-7
        self.freq = torch.ones([class_num]).float().cuda() / class_num

    def forward(self, preds: torch.Tensor, label: torch.Tensor):
        ce_loss = tnf.cross_entropy(preds, label, reduction='none')
        b, h, w = ce_loss.shape
        # update frequency for each class
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

        label_onehot = self._one_hot(label)  # (b*h*w, c)
        # loss weight computed by class frequency
        class_prob = self._get_class_wight()  # (c,)
        weight = (label_onehot * class_prob.unsqueeze(dim=0)).sum(dim=1)  # (b*h*w,)
        # loss weight computed by difficulty
        if self.is_hard:
            weight_hard = self._get_pixel_difficulty(ce_loss, label_onehot)  # (b*h*w,)
            # merged weight
            weight = ((weight + weight_hard) * 0.5)  # (b, h, w)
        weight = weight.view(b, h, w).detach()
        # balanced loss
        loss_balanced = torch.mean(weight * ce_loss)
        return loss_balanced

    def ema_update(self, label):
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

    def _get_class_wight(self):
        prob = (1.0 - self.freq) / self.temperature
        prob = torch.softmax(prob, dim=0)
        _max, _ = torch.max(prob, dim=0, keepdim=True)
        prob_normed = prob / (_max + self.eps)  # 0 ~ 1
        return prob_normed

    # def _get_pixel_difficulty(self, ce_loss, label_onehot):
    #     ce_loss = ce_loss.view(-1, 1)               # (b*h*w, 1)
    #     ce_loss_classwise = ce_loss * label_onehot  # (b*h*w, c)
    #     _max, _ = torch.max(ce_loss_classwise, dim=0, keepdim=True)     # (1, c)
    #     pixel_difficulty = ce_loss_classwise / (_max + self.eps)    # (b*h*w, c)
    #     pixel_difficulty = pixel_difficulty.sum(dim=1)      # (b*h*w,)
    #     return pixel_difficulty

    def _get_pixel_difficulty(self, ce_loss, label_onehot):
        ce_loss = ce_loss.view(-1, 1)  # (b*h*w, 1)
        ce_loss_classwise = ce_loss * label_onehot  # (b*h*w, c)
        _mean = torch.mean(ce_loss_classwise, dim=0, keepdim=True)  # (1, c)
        pixel_difficulty = ce_loss_classwise / (_mean + self.eps)  # (b*h*w, c)
        pixel_difficulty = pixel_difficulty.sum(dim=1)  # (b*h*w,)
        pixel_difficulty = torch.sigmoid(pixel_difficulty)
        return pixel_difficulty

    def _local_freq(self, label):
        assert len(label.shape) == 3
        local_cnt = torch.sum((label != self.ignore_label).float())
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
        self.bins = bins
        self.momentum = momentum
        self.ignore_label = ignore_label
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = self.edges[-1] + 1e-3
        self.edges = torch.FloatTensor(self.edges).cuda()
        if momentum > 0:
            # self.acc_sum = [torch.zeros(1).cuda() for _ in range(bins)]
            self.acc_sum = torch.zeros(bins).cuda()

    def forward(self, preds, targets):
        # Compute the number of classes
        n_classes = preds.size(1)

        # Flatten the prediction and target tensors
        preds = preds.permute((0, 2, 3, 1)).reshape(-1, n_classes)
        prob_max, _ = torch.max(torch.softmax(preds, dim=1), dim=1)
        targets = targets.view(-1)

        # Calculate the gradient of the prediction
        gradient = torch.abs(prob_max - 1.0)
        cond_ignore = (targets == self.ignore_label) | (targets < 0) | (targets >= n_classes)
        gradient[cond_ignore] = -1      # ignore the invalid or ignored targets

        # Sort the gradient and prediction values
        bins = torch.histc(gradient, bins=self.bins, min=0, max=1)  # out-bounded values will not be statistic
        inds = torch.bucketize(gradient, self.edges)  # lower than min will be 0, lager than max will be len(bins)

        # Calculate the weights for each sample based on the gradient
        weights = torch.zeros_like(gradient).cuda()
        cond_weights = (inds > 0) & (inds <= self.bins)
        if cond_weights.sum() > 0:
            # print(f'inds.min={inds.min()}, inds.max={inds.max()}, g.min={gradient.min()}, '
            #       f'g.max={gradient.max()}, target-range={torch.unique(targets)}')
            if self.momentum > 0:
                ema = self.momentum * self.acc_sum + (1 - self.momentum) * bins
                self.acc_sum = torch.where(bins != 0, ema, self.acc_sum)
                weights = torch.where(cond_weights, 1.0 / (self.acc_sum[inds - 1]), weights)
            else:
                weights = torch.where(cond_weights, 1.0 / (bins[inds - 1]), weights)
        weights = weights.detach()
        # Calculate the GHM loss
        loss = tnf.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_label)
        loss = loss * weights
        loss = loss.sum() / (weights.sum() + 1e-7)
        return loss


if __name__ == '__main__':

    prob = torch.rand(8, 6, 512, 512).cuda()
    target = torch.randint(-1, 2, (8, 512, 512)).cuda()
    FL = FocalLoss()
    # print('focal loss: ', FL(tnf.cross_entropy(prob, target, reduction='none'), target))

    ghm = GHMLoss(momentum=0)
    print('ghm loss: ', ghm(torch.softmax(prob, dim=1), target))


    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
            torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
            torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
            torch.randint(1, 2, [8, 512, 512]).long().cuda()


    x_s, x_t, l_s, l_t = rand_x_l()

    cbl = ClassBalanceLoss(class_num=3)
    _rand_val = torch.rand_like(l_t, dtype=torch.float).cuda()
    for _ in range(1000):
        cbl.ema_update(l_t)
    _ce_loss = torch.rand([8, 512, 512]).cuda()
    l_t[:, 0, :] = 0
    print(torch.mean(_ce_loss))
    _loss = cbl(x_t, l_t)
    print(_loss)
