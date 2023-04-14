"""
@Project : rsda
@File    : alignment.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/13 下午9:43
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from module.gast.class_ware_whiten import ClassWareWhitening
from module.gast.coral import CoralLoss
from audtorch.metrics.functional import pearsonr
# from module.gast.mmd import MMDLoss
import math


class Aligner:

    def __init__(self, logger, feat_channels=64, class_num=7, ignore_label=-1):
        self.feat_channels = feat_channels
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.logger = logger
        self.eps = 1e-7

        # prototypes for all classes
        self.prototypes = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()

        # downscale for gt label with full size
        self.downscale_gt = DownscaleLabel(scale_factor=16, n_classes=7, ignore_label=ignore_label, min_ratio=0.75)

        # criterion for domain alignment
        self.coral = CoralLoss()
        # self.mmd = MMDLoss(kernel_type='linear')

        # criterion for feature whitening
        self.whitener = ClassWareWhitening(class_ids=range(class_num), groups=32)  # self.feat_channels // 8)

    def compute_local_prototypes(self, feat, label, update=False, decay=0.99):
        # feat, label = self._reshape_pair(feat, label)
        b, k, h, w = feat.shape
        labels = self.downscale_gt(label)  # (b, 32*h, 32*w) -> (b, 1, h, w)
        labels = self._index2onehot(labels)  # (b, 1, h, w) -> (b, c, h, w)
        feats = feat.permute(0, 2, 3, 1).reshape(-1, k)  # (b*h*w, k)
        feats = feats.view(-1, 1, k)  # (b*h*w, 1, k)
        labels = labels.view(-1, self.class_num, 1)  # (b*h*w, c, 1)
        # local_prototype = self._get_local_prototypes(feat, label)
        local_prototype = (feats * labels).sum(0) / (labels.sum(0) + self.eps)
        if update:
            self.prototypes = self._ema(self.prototypes, local_prototype, decay).detach()
        return local_prototype

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        feat_s = feat_s.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        feat_t = feat_t.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        return self.coral(feat_s, feat_t)

    def align_class(self, feat_s, label_s, feat_t=None, label_t=None):
        """ Compute the loss for class level alignment.
            Besides, update the shared prototypes by the local prototypes of source and target domain.
        Args:
            feat_s:  features from source, shape as (b, k, h, w)
            label_s: labels from source  , shape as (b, 32*h, 32*w)
            feat_t:  features from source, shape as (b, k, h, w)
            label_t: pseudo labels from target, shape as (b, 32*h, 32*w)
        Returns:
            loss_class: the loss for class level alignment
        """
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        assert len(label_s.shape) >= 3, 'tensor "label_s" and "label_t" must have 3 dimensions'
        half = feat_s.shape[0] // 2
        local_prototype_s1 = self.compute_local_prototypes(feat_s[:half], label_s[:half], update=False)
        local_prototype_s2 = self.compute_local_prototypes(feat_s[half:], label_s[half:], update=False)
        local_prototype_s = self.compute_local_prototypes(feat_s, label_s, update=True)
        loss_inter = self._class_align_loss(local_prototype_s1, local_prototype_s2)
        if feat_t is None or label_t is None:
            loss_class = loss_inter
        else:
            local_prototype_t = self.compute_local_prototypes(feat_t, label_t, update=False)
            # loss_class = tnf.mse_loss(local_prototype_s, local_prototype_t, reduction='mean')
            loss_intra = self._class_align_loss(local_prototype_s, local_prototype_t)
            loss_class = 0.5 * (loss_inter + loss_intra)
        return loss_class

    def align_instance(self, feat_s, label_s, feat_t=None, label_t=None):
        loss_instance = self._instance_align_loss(feat_s, self.downscale_gt(label_s))
        if feat_t is not None and label_t is not None:
            loss_instance += self._instance_align_loss(feat_t, self.downscale_gt(label_t))
            loss_instance /= 2.0
        return loss_instance

    def whiten_class_ware(self, feat_s, label_s, feat_t=None, label_t=None):
        loss_white = self.whitener(feat_s, self.downscale_gt(label_s))
        if feat_t is not None and label_t is not None:
            loss_white += self.whitener(feat_t, self.downscale_gt(label_t))
            loss_white /= 2.0
        return loss_white

    def show(self, save_path=None, display=True):
        pass

    def _class_align_loss(self, prototypes_1, prototypes_2, margin=0.3, hard_ratio=0.3):
        """ Compute the loss between two local prototypes.
        Args:
            prototypes_1: local prototype from a batch. shape=(c, k)
            prototypes_2: local prototype from a batch. shape=(c, k)
            margin: distance for margin loss. a no-neg float.
            hard_ratio: ratio for selecting hardest examples.
        Returns:
            loss_2p: loss for two local prototypes.
        """
        assert 0 <= margin and 0 < hard_ratio <= 1
        assert prototypes_1.shape == prototypes_2.shape
        # compute distance matrix for each class pair
        # dist_matrix = torch.cdist(prototypes_1, prototypes_2, p=2)  # (c, c), Euclidean distances
        dist_matrix = self._pearson_dist(prototypes_1, prototypes_2)  # (c, c), pearson distances

        # hard example mining
        hard_num = min(math.ceil(hard_ratio * self.class_num) + 1, self.class_num)
        eye_neg = 1 - torch.eye(self.class_num).cuda()
        dist_sorted, _ = torch.sort(dist_matrix * eye_neg, dim=1, descending=False)
        dist_hardest = dist_sorted[:, : hard_num]  # (c, hard_num)

        # the mean distance between the same classes
        d_mean_pos = torch.diag(dist_matrix).mean()
        # the mean distance across classes
        d_mean_neg = dist_hardest.sum() / (self.class_num * (hard_num - 1) + self.eps)
        # loss_p2p = (1 + d_mean_pos) / (1 + d_mean_neg + self.eps)
        loss_p2p = (d_mean_pos - d_mean_neg + margin).max(torch.Tensor([1e-6]).cuda()[0])
        return loss_p2p

    def _instance_align_loss(self, feat, label, margin=0.3, hard_ratio=0.3):
        """Compute the loss between instances and prototypes.
        Args:
            feat: deep features outputted by backbone. shape=(b, k, h, w)
            label: gt or pseudo label. shape=(b, h, w) or (b, 1, h, w)
            margin: distance for margin loss. a no-neg float.
            hard_ratio: ratio for selecting hardest examples.
        Returns:
            loss_2p: loss between instances and their prototypes.
        """
        assert 0 <= margin and 0 < hard_ratio <= 1
        feat = feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)
        ins_num = feat.shape[0]
        # compute the dist between instances and classes
        # dist_matrix = torch.cdist(feat, self.prototypes, p=2)  # Euclidean distance, (b*h*w, c)
        dist_matrix = self._pearson_dist(feat, self.prototypes)  # Euclidean distance, (b*h*w, c)

        # compute the distances for each instance with their nearest prototypes.
        mask_pos = self._index2onehot(label)  # (b*h*w, c)
        mask_neg = 1 - mask_pos  # (b*h*w, c)
        hard_num = min(math.ceil(hard_ratio * self.class_num) + 1, self.class_num)
        dist_sorted, _ = torch.sort(dist_matrix * mask_neg, dim=1, descending=False)
        dist_hardest = dist_sorted[:, : hard_num]

        # the mean distance between instances and their prototypes
        d_mean_pos = (dist_matrix * mask_pos).sum() / (ins_num + self.eps)
        d_mean_neg = dist_hardest.sum() / (ins_num * (hard_num - 1) + self.eps)
        loss_i2p = (d_mean_pos - d_mean_neg + margin).max(torch.Tensor([1e-6]).cuda()[0])
        # loss_i2p = (1 + d_mean_pos) / (1 + d_mean_neg + self.eps)
        return loss_i2p

    def _pearson_dist(self, feat1, feat2):
        """
        Compute the pearson distance between the representation vector of each instance
        Args:
            feat1: torch.FloatTensor, (n, k)
            feat2: torch.FloatTensor, (m, k)

        Returns:
            pearson_dist: (n, m), from 0 to 1
        """
        assert feat1.shape[-1] == feat2.shape[-1]
        k = feat1.shape[-1]
        centered_feat1 = feat1 - feat1.mean(dim=-1, keepdim=True)   # (n, k)
        centered_feat2 = feat2 - feat2.mean(dim=-1, keepdim=True)   # (m, k)
        centered_feat1 = centered_feat1.unsqueeze(dim=1)
        centered_feat2 = centered_feat2.unsqueeze(dim=0)
        covariance = (centered_feat1 * centered_feat2).sum(dim=-1, keepdim=False)   # (n,  m)

        bessel_corrected_covariance = covariance / (k - 1 + self.eps)   # (n,  m)

        feat1_std = feat1.std(dim=-1, keepdim=False)    # (n,)
        feat2_std = feat2.std(dim=-1, keepdim=False)    # (m,)
        feat1_std = feat1_std.unsqueeze(dim=1)          # (n, 1)
        feat2_std = feat2_std.unsqueeze(dim=0)          # (1 ,m)
        div_mat = feat1_std * feat2_std                 # (n, m)
        pearson_dist = (-1.0 * bessel_corrected_covariance / (div_mat + self.eps) + 1.0) * 0.5

        return pearson_dist     # (n, m)

    @staticmethod
    def _ema(history, curr, decay=0.99):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average

    def _index2onehot(self, label):
        labels = label.clone()
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1)  # (b*h*w, 1)
        labels[labels == self.ignore_label] = self.class_num
        labels = tnf.one_hot(labels.squeeze(1), num_classes=self.class_num + 1)[:, :-1]  # (b*h*w, c)
        return labels


class DownscaleLabel(nn.Module):

    def __init__(self, scale_factor=16, n_classes=7, ignore_label=-1, min_ratio=0.75):
        super().__init__()
        assert scale_factor > 1
        self.scale_factor = scale_factor
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.min_ratio = min_ratio

    def forward(self, label):
        label = label.clone()
        if len(label.shape) == 3:
            label = label.unsqueeze(dim=1)
        bs, orig_c, orig_h, orig_w = label.shape
        assert orig_c == 1
        trg_h, trg_w = orig_h // self.scale_factor, orig_w // self.scale_factor
        label[label == self.ignore_label] = self.n_classes
        out = tnf.one_hot(label.squeeze(1), num_classes=self.n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, self.n_classes + 1, orig_h, orig_w], out.shape
        out = tnf.avg_pool2d(out.float(), kernel_size=self.scale_factor)
        max_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == self.n_classes] = self.ignore_label
        out[max_ratio < self.min_ratio] = self.ignore_label
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out


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
        num_classes=7,
        is_ins_norm=True,
    )).cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    aligner = Aligner(logger=logging.getLogger(''), feat_channels=2048, class_num=7)


    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
               torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
               torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
               torch.randint(1, 2, [8, 512, 512]).long().cuda()


    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        _loss_white = aligner.whiten_class_ware(f_s, l_s, f_t, l_t)
        print(_loss_white.cpu().item(), '\t', i)
        optimizer.zero_grad()
        _loss_white.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss white')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        _loss_domain = aligner.align_domain(f_s, f_t)
        print(_loss_domain)
        optimizer.zero_grad()
        _loss_domain.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss domain')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        _loss_class = aligner.align_class(f_s, l_s, f_t, l_t)
        print(_loss_class)
        optimizer.zero_grad()
        _loss_class.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss class ')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        _loss_ins = aligner.align_instance(f_s, l_s, f_t, l_t)
        print(_loss_ins)
        optimizer.zero_grad()
        _loss_ins.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss instance')
    print('=========================================================')

    from utils.tools import loss_calc

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        os1, os2, f_s = model(x_s)
        ot1, ot2, f_t = model(x_t)
        _loss_seg = loss_calc([os1, os2], l_s, multi=True)
        _loss_seg += loss_calc([ot1, ot2], l_t, multi=True)
        print(_loss_seg)
        optimizer.zero_grad()
        _loss_seg.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss seg')
    print('=========================================================')
    print('end')
