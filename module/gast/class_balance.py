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


class ClassBalanceLoss(nn.Module):

    def __init__(self, class_num=7, ignore_label=-1, decay=0.998, min_prob=0.01, temperature=0.5,
                 is_balance=True):
        super().__init__()
        assert temperature > 0
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.decay = decay
        self.min_prob = torch.FloatTensor([min_prob]).cuda()[0]
        self.temperature = temperature
        self.is_balance = is_balance
        self.eps = 1e-7
        self.freq = torch.ones([class_num]).float().cuda() / class_num

    def forward(self, ce_loss: torch.Tensor, label: torch.Tensor):
        if not self.is_balance:
            return torch.mean(ce_loss)
        assert ce_loss.shape == label.shape
        b, h, w = ce_loss.shape
        # update frequency for each class
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

        label_onehot = self._one_hot(label)                 # (b*h*w, c)
        # loss weight computed by class frequency
        class_prob = self._get_class_prob()                 # (c,)
        weight_1 = (label_onehot * class_prob.unsqueeze(dim=0)).sum(dim=1)     # (b*h*w,)
        # loss weight computed by difficulty
        weight_2 = self._get_pixel_difficulty(ce_loss, label_onehot)           # (b*h*w,)
        # merged weight
        weight = ((weight_1 + weight_2) * 0.5).view(b, h, w).detach()                  # (b, h, w)

        # balanced loss
        loss_balanced = torch.mean(weight * ce_loss)
        return loss_balanced

    def ema_update(self, label):
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

    def _get_class_prob(self):
        prob = (1.0 - self.freq) / self.temperature
        prob = torch.softmax(prob, dim=0)
        _max, _ = torch.max(prob, dim=0, keepdim=True)
        prob_normed = prob / (_max + self.eps)      # 0 ~ 1
        return prob_normed

    def _get_pixel_difficulty(self, ce_loss, label_onehot):
        ce_loss = ce_loss.view(-1, 1)               # (b*h*w, 1)
        ce_loss_classwise = ce_loss * label_onehot  # (b*h*w, c)
        _max, _ = torch.max(ce_loss_classwise, dim=0, keepdim=True)     # (1, c)
        pixel_difficulty = ce_loss_classwise / (_max + self.eps)    # (b*h*w, c)
        pixel_difficulty = pixel_difficulty.sum(dim=1)      # (b*h*w,)
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
        freq = self._get_class_prob()
        res += f';\tselect probability: {freq[0]:.3f}'
        for _freq in freq[1:]:
            res += f', {_freq:.3f}'
        return res


if __name__ == '__main__':
    cbl = ClassBalanceLoss()


    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
            torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
            torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
            torch.randint(1, 2, [8, 512, 512]).long().cuda()


    x_s, x_t, l_s, l_t = rand_x_l()

    _rand_val = torch.rand_like(l_t, dtype=torch.float).cuda()
    for _ in range(1000):
        cbl.ema_update(l_t)
    _ce_loss = torch.rand([8, 512, 512]).cuda()
    l_t[:, 0, :] = 0
    print(torch.mean(_ce_loss))
    _loss = cbl(_ce_loss, l_t)
    print(_loss)
