"""
@Project : rsda 
@File    : pseudo_generation.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/2/19 16:06
@e-mail  : liuwa@hnu.edu.cn
"""

import os
import torch
import cv2
from tqdm import tqdm
from utils.tools import *
from module.viz import VisualizeSegmm


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6):
    """
    Convert continuous mask into binary mask
    Args:
        mask: torch.Tensor, shape=(b, c, h, w)
        cutoff_top:
        cutoff_low:

    Returns:

    """
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)
    # remove ambiguous pixels, ambiguous = 1 means ignore
    ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)

    pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
    pseudo_gt[ambiguous == 1] = -1

    return pseudo_gt.view(bs, h, w).cpu().numpy()


def gener_target_pseudo(_cfg, model, pseudo_loader, save_pseudo_label_path, slide=True):
    model.eval()

    save_pseudo_color_path = save_pseudo_label_path + '_color'
    if not os.path.exists(save_pseudo_color_path):
        os.makedirs(save_pseudo_color_path)
    viz_op = VisualizeSegmm(save_pseudo_color_path, palette)

    with torch.no_grad():
        for ret, ret_gt in tqdm(pseudo_loader):
            ret = ret.cuda()

            cls = pre_slide(model, ret, tta=True) if slide else model(ret)
            # cls = pre_slide(model, ret, tta=True)
            # pseudo selection, from -1~6
            if _cfg.PSEUDO_SELECT:
                cls = pseudo_selection(cls)
            else:
                cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                        (cls + 1).reshape(1024, 1024).astype(np.uint8))

            if _cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))

