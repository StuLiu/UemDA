"""
@Filename: GAST_train
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""
import logging


import importlib
import time
import os
import cv2
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np
import ttach
import ever as er
import argparse
import torch.multiprocessing

from skimage.io import imsave
from functools import reduce
from collections import OrderedDict
from tqdm import tqdm
from math import *
from scipy import ndimage

import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from eval import evaluate
from uemda.utils.tools import *
from uemda.models.Encoder import Deeplabv2
from uemda.datasets.daLoader import DALoader
from uemda.datasets import LoveDA, IsprsDA
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
from uemda.viz import VisualizeSegmm
from uemda.gast.alignment import Aligner
from uemda.gast.pseudo_generation import pseudo_selection
from uemda.gast.balance import *
from uemda.utils.ema import ExponentialMovingAverage
from uemda.utils.tools import pad_image


# --config-path st.gast.2urban --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 0.5
# --config-path st.gast.2rural --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 1000

parser = argparse.ArgumentParser(description='Run GAST methods.')
parser.add_argument('--config-path', type=str, default='st.gast.2potsdam', help='config path')
parser.add_argument('--ckpt-path', type=str, default='ckpts/Vaihingen4000.pth', help='ckpt path')

parser.add_argument('--refine-label', type=str2bool, default=1, help='whether refine the pseudo label')
parser.add_argument('--refine-mode', type=str, default='all', choices=['s', 'p', 'n', 'l', 'all'],
                    help='refine by prototype, label, or both')
parser.add_argument('--refine-temp', type=float, default=2.0, help='whether refine the pseudo label')

parser.add_argument('--rm-pseudo', type=str2bool, default=0, help='remove pseudo label directory')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path)
assert cfg.FIRST_STAGE_STEP <= cfg.NUM_STEPS_STOP, 'FIRST_STAGE_STEP must no larger than NUM_STEPS_STOP'


class Deeplabv2_(Deeplabv2):
    def __init__(self, config):
        super(Deeplabv2_, self).__init__(config)

    def forward(self, x):
        feat = self.encoder(x)[-1]
        if self.config.is_ins_norm:
            feat = self.instance_norm(feat)
        x1 = self.layer5(feat)
        x2 = self.layer6(feat)
        if self.training:
            return x1, x2, feat
        else:
            x1 = tnf.interpolate(x1, x.shape[-2:], mode='bilinear', align_corners=True)
            x2 = tnf.interpolate(x2, x.shape[-2:], mode='bilinear', align_corners=True)
            return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2, x1, x2, feat


def tta_predict(model, img):
    tta_transforms = ttach.Compose(
        [
            ttach.HorizontalFlip(),
            ttach.Rotate90(angles=[0, 90, 180, 270]),
        ])

    xs = []

    for t in tta_transforms:
        aug_img = t.augment_image(img)
        aug_x, _, _, _  = model(aug_img)
        # aug_x = tnf.softmax(aug_x, dim=1)

        x = t.deaugment_mask(aug_x)
        xs.append(x)

    xs = torch.cat(xs, 0)
    x = torch.mean(xs, dim=0, keepdim=True)

    return x


def pre_slide(model, image, num_classes=7, tile_size=(512, 512), tta=False):
    image_size = image.shape  # i.e. (1,3,1024,1024)
    overlap = 1 / 2  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:512*(1-1/2) = 256
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-512)/256 + 1 = 3
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(1024-512)/256 + 1 = 3

    full_probs = torch.zeros((image_size[0], num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 (1,7,1024,1024)
    count_predictions = torch.zeros((image_size[0], 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 (1,1,1024,1024)

    for row in range(tile_rows):  # row = 0,1,2
        for col in range(tile_cols):  # col = 0,1,2
            x1 = int(col * stride)  # 起始位置x1 = 0 * 256 = 0
            y1 = int(row * stride)  # y1 = 0 * 256 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+512, 1024)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+512, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(512-512, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(512-512, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:512, 0:512]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为512*512

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta is True:
                padded = tta_predict(model, padded_img)
            else:
                padded, _, _, _ = model(padded_img)
                # padded = tnf.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(1,7,512,512)
            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果
    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs  # 返回整张图的平均概率 shape(1, 1, 1024,1024)


def gener_target_pseudo(_cfg, model, pseudo_loader, save_pseudo_label_path, aligner,
                        slide=True, save_prob=False, size=(1024, 1024), ignore_label=-1):
    """
    Generate pseudo label for target domain. The saved probabilities should be processed by softmax.
    Args:
        _cfg: cfg loaded from configs/
        model: nn.Module, deeplabv2
        pseudo_loader: DataLoader, dataloader for target domain image, batch_size=1
        save_pseudo_label_path: os.path.Path or str, the path for pseudo labels
        slide: bool, if use slide mode when do inferring.
        save_prob: bool, if save probabilities or class ids.
        size: tuple or list, the height and width of the pseudo labels.

    Returns:
        None
    """
    save_pseudo_color_path = save_pseudo_label_path + '_color'
    os.makedirs(save_pseudo_color_path, exist_ok=True)
    viz_op = VisualizeSegmm(save_pseudo_color_path, eval(_cfg.DATASETS).PALETTE)
    num_classes = len(eval(_cfg.DATASETS).LABEL_MAP)
    model.eval()
    _i = 0
    with torch.no_grad():
        for ret, ret_gt in tqdm(pseudo_loader):
            # _i += 1
            # if _i >= 10:
            #     break

            ret, label_t_sup = ret.cuda(), ret_gt['sup'].cuda()

            _, pred_t1, pred_t2, feat_t = model(ret)
            cls = pre_slide(model, ret, num_classes=num_classes, tta=True) if slide else model(ret)  # (b, c, h, w)

            cls = aligner.label_refine(label_t_sup, feat_t, [pred_t1, pred_t2], cls,
                                       refine=args.refine_label, mode=args.refine_mode, temp=args.refine_temp)

            cls = pseudo_selection(cls, ignore_label=ignore_label,
                                   cutoff_top=_cfg.CUTOFF_TOP, cutoff_low=_cfg.CUTOFF_LOW)  # (b, h, w)

            cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                        (cls + 1).reshape(*size).astype(np.uint8))

            if _cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('.tif', '.png'))

def main():
    time_from = time.time()
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(save_pseudo_label_path, exist_ok=True)

    logger = get_console_file_logger(name='GAST', logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    cudnn.enabled = True
    model = Deeplabv2_(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=True,
    ))
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'), strict=True)
    model = model.cuda()

    aligner = Aligner(logger=logger,
                      feat_channels=2048,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.99)

    # source loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    # pseudo loader (target)
    pd_cfg = cfg.PSEUDO_DATA_CONFIG
    pd_cfg['read_sup'] = True
    pseudo_loader = DALoader(pd_cfg, cfg.DATASETS)

    # init prototypes
    logger.info('###### init prototypes ######')
    for _ in tqdm(range(500)):
        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        _, _, feat_s = model(images_s)
        # update prototypes
        aligner.update_prototype(feat_s, label_s)

    # Generate pseudo label begin
    logger.info('###### Start generate pseudo dataset in round ######')
    # save pseudo label for target domain
    gener_target_pseudo(cfg, model, pseudo_loader, save_pseudo_label_path, aligner,
                        size=eval(cfg.DATASETS).SIZE, save_prob=False, slide=True, ignore_label=ignore_label)
    # Generate pseudo label finish
    logger.info('###### Ended generate pseudo dataset in round ######')

    if args.rm_pseudo:
        logger.info('removing pseudo labels begin >>>>>>>>>>>>')
        shutil.rmtree(save_pseudo_label_path, ignore_errors=True)
        logger.info('removing pseudo labels end <<<<<<<<<<<<<<')

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
