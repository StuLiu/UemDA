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
from module.gast.contrastive import ContrastiveLoss


class Aligner:

    def __init__(self, logger, feat_channels=64, class_num=7, ignore_label=-1):
        # self.mmd = MMDLoss()
        self.feat_channels = feat_channels
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.logger = logger
        self.eps = 1e-5

        # statistics of classes
        self.prototypes = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()

        self.downscale_gt = DownscaleLabel(scale_factor=16, n_classes=7, ignore_label=ignore_label, min_ratio=0.75)
        self.coral = CoralLoss()
        self._whitener = ClassWareWhitening(class_ids=range(class_num), groups=8)#self.feat_channels // 8)
        self.contra = ContrastiveLoss(margin=0.5, class_num=class_num)

    def init_prototypes_by_source(self, feat_s, label_s):
        feat_s, label_s = self._reshape_pair(feat_s, label_s)
        local_prototype_s = self._get_local_prototypes(feat_s, label_s)
        self.update_prototypes(local_prototype_s)

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        feat_s = feat_s.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        feat_t = feat_t.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        return self.coral(feat_s, feat_t)

    def align_class(self, feat_s, label_s, feat_t, label_t):
        """ Compute the loss for discrepancy between class distribution.
        Args:
            feat_s:  features from source, shape as (batch_size, feature_channels, height, width)
            label_s: labels from source  , shape as (batch_size, height, width)
            feat_t:  features from source, shape as (batch_size, feature_channels, height, width)
            label_t: pseudo labels from target, shape as (batch_size, height, width)
        Returns:
            loss_whiten:
            loss_triple:
        """
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        assert len(label_s.shape) >= 3, 'tensor "label_s" and "label_t" must have 3 dimensions'

        feat_s, label_s = self._reshape_pair(feat_s, label_s)     # (b*h*w, 1, k), (b*h*w, c, 1)
        feat_t, label_t = self._reshape_pair(feat_t, label_t)     # (b*h*w, 1, k), (b*h*w, c, 1)

        # get local prototype
        local_prototype_s = self._get_local_prototypes(feat_s, label_s)
        local_prototype_t = self._get_local_prototypes(feat_t, label_t)

        # update the shared prototypes
        self.update_prototypes(local_prototype_s)
        self.update_prototypes(local_prototype_t)

        # get contrastive loss within a single domain and cross domain
        # loss_contrastive = self.loss_constractive(feat_s, label_s, feat_t, label_t)
        loss_contrastive = tnf.mse_loss(local_prototype_s, local_prototype_t, reduction='mean')

        # get class-ware whitening loss
        loss_whiten = 0#self._whitener(feat_s, label_s) + self._whitener(feat_t, label_t)
        return 0.001 * loss_whiten + loss_contrastive  # , tnf.mse_loss(class_u_t_local, self.prototypes.detach())

    def whitening(self, feat, label):
        return self._whitener(feat, self.downscale_gt(label))

    @staticmethod
    def ema(history, curr, decay=0.99):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average

    def update_prototypes(self, p_local, decay=0.99):
        self.prototypes = self.ema(self.prototypes, p_local, decay).detach()

    def show(self, save_path=None, display=True):
        pass

    def _reshape_pair(self, feat, label):
        """ Get the flattened features and the one-hot labels
        Args:
            feat:   (b, k, h, w)
            label:  (b, 1, 32*h, 32*w)
        Returns:
            feats:  (b*h*w, 1, k)
            labels: (b*h*w, c, 1), one hot label without the ignored label
        """
        labels = label.clone()
        labels = self.downscale_gt(labels)    # (b, h, w) -> (b, 1, h/32, w/32)
        b, k, h, w = feat.shape
        feats = feat.permute(0, 2, 3, 1).reshape(-1, k)  # (b*h*w, k)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1)  # (b*h*w, 1)
        labels[labels == self.ignore_label] = self.class_num
        labels = tnf.one_hot(labels.squeeze(1), num_classes=self.class_num + 1)[:, :-1]  # (b*h*w, c)
        feats = feats.view(-1, 1, k)  # (b*h*w, 1, k)
        labels = labels.view(-1, self.class_num, 1)  # (b*h*w, c, 1)
        return feats, labels

    def _get_local_prototypes(self, feats, labels):
        """ Get the prototypes of the classes in a mini-batch
        Args:
            feats: feature maps, (b*h*w, 1, k)
            label: one-hotted gt or pseudo label (b*h*w, c, 1)
        Returns:
            u_classes: local prototypes within a batch
        """
        local_p = (feats * labels).sum(0) / (labels.sum(0) + self.eps)      # (b*h*w, c, k)
        return local_p


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
        loss_class = aligner.align_class(f_s, l_s, f_t, l_t)
        loss_white = aligner.whitening(f_s, l_s) + aligner.whitening(f_t, l_t)
        print(loss_class, loss_white)
        optimizer.zero_grad()
        loss_white.backward()
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
