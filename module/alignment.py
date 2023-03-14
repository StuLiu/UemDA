"""
@Project : Unsupervised_Domian_Adaptation
@File    : alignment.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/13 下午9:43
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn.functional as fun


class Aligner:

    def __init__(self, feat_channels=64, class_num=7, ignore_label=255):
        self.class_num = class_num
        self.ignore_label = ignore_label
        self.prototypes = torch.zeros([class_num, feat_channels]).cuda()

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        u_s = torch.mean(feat_s, dim=[0, 2, 3], keepdim=True)
        u_t = torch.mean(feat_t, dim=[0, 2, 3], keepdim=True)
        sigma_s = torch.sqrt(torch.mean((feat_s - u_s) ** 2, dim=[0, 2, 3]))
        sigma_t = torch.sqrt(torch.mean((feat_t - u_t) ** 2, dim=[0, 2, 3]))
        return fun.mse_loss(u_s, u_t) + fun.mse_loss(sigma_s, sigma_t)

    def align_category(self, feat_s, label_s, feat_t, label_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'


    def align_instance(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'


if __name__ == '__main__':
    aligner = Aligner(feat_channels=128, class_num=7)
    print(f'prototypes size={aligner.prototypes.shape}')
    f_s = torch.randn([8, 128, 32, 32]).cuda() * 2
    f_t = torch.randn([8, 128, 32, 32]).cuda()
    print(f_s.device)
    print(aligner.align_domain(f_s, f_t))
    pass
