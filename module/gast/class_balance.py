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

    def __init__(self, class_num=7, ignore_label=-1, decay=0.998, min_ratio=0.1, temperature=0.5,
                 is_balance=True):
        super().__init__()
        assert temperature > 0
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.decay = decay
        self.min_ratio = torch.FloatTensor([min_ratio]).cuda()[0]
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
        # selected probabilities by class frequency
        class_prob = self._get_class_prob()                 # (c,)
        select_prob_1 = (label_onehot * class_prob.unsqueeze(dim=0)).sum(dim=1)     # (b*h*w,)
        # selected probabilities by difficulty
        select_prob_2 = self._get_pixel_difficulty(ce_loss, label_onehot)           # (b*h*w,)
        select_prob_2 /= (select_prob_2.max(dim=0)[0] + self.eps)       # (b*h*w,)
        # merged probabilities
        select_prob = (select_prob_1 + select_prob_2) * 0.5             # (b*h*w,)
        select_prob = select_prob.view(b, h, w)                         # (b, h, w)
        select_prob = select_prob.max(self.min_ratio).detach()          # (b, h, w)

        # select loss
        rand_val = torch.rand_like(ce_loss, dtype=torch.float).cuda()   # (b, h, w)
        selected = rand_val <= select_prob  # (b, h, w)
        ce_loss_selected = ce_loss.where(selected, torch.FloatTensor([0]).cuda()[0])  # (b, h, w)
        cnt = torch.sum(selected.float())
        loss_mean = torch.sum(ce_loss_selected) / (cnt + self.eps)
        return loss_mean

    def ema_update(self, label):
        self.freq = self._ema(self.freq, self._local_freq(label), decay=self.decay)

    def _get_class_prob(self):
        prob = (1.0 - self.freq) / self.temperature
        prob = torch.softmax(prob, dim=0)
        prob /= (torch.max(prob, dim=0, keepdim=True)[0] + self.eps)
        return prob

    def _get_pixel_difficulty(self, ce_loss, label_onehot):
        ce_loss = ce_loss.view(-1, 1)  # (b*h*w, 1)
        ce_loss_classwise = ce_loss * label_onehot  # (b*h*w, c)
        max_loss_classwise, _ = torch.max(ce_loss_classwise, dim=0, keepdim=True)  # (1, c)
        ce_loss_normed = ce_loss_classwise / (max_loss_classwise + self.eps)  # (b*h*w, c)
        ce_loss_normed = ce_loss_normed.sum(dim=1)  # (b*h*w,)
        pixel_difficulty = torch.sigmoid(ce_loss_normed) * 2 - 1  # (b*h*w,)
        return pixel_difficulty

    def _local_freq(self, label):
        assert len(label.shape) == 3
        local_cnt = torch.numel(label)
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
