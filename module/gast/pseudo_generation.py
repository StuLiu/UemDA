"""
@Project : rsda 
@File    : pseudo_generation.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/2/19 16:06
@e-mail  : liuwa@hnu.edu.cn
"""

import cv2
import numpy as np
import torch

from glob import glob
import matplotlib.pyplot as plt
from module.utils.tools import *
from module.viz import VisualizeSegmm
from module.datasets import *


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6, return_type='ndarray', ignore_label=-1):
    """
    Convert continuous mask into binary mask
    Args:
        mask: torch.Tensor, the predicted probabilities for each examples, shape=(b, c, h, w).
        cutoff_top: float, the ratio for computing threshold.
        cutoff_low: float, the minimum threshold.
        return_type: str, the format of the return item, should be in ['ndarray', 'tensor'].
    Returns:
        ret: Tensor, pseudo label, shape=(b, h, w).
    """
    assert return_type in ['ndarray', 'tensor']
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)  # (b, c, 1)
    mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)
    # remove ambiguous pixels, ambiguous = 1 means ignore
    ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)

    pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
    pseudo_gt[ambiguous == 1] = ignore_label
    if return_type == 'ndarray':
        ret = pseudo_gt.view(bs, h, w).cpu().numpy()
    else:
        ret = pseudo_gt.view(bs, h, w)
    return ret


def gener_target_pseudo(_cfg, model, pseudo_loader, save_pseudo_label_path,
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
    model.eval()

    save_pseudo_color_path = save_pseudo_label_path + '_color'
    if not os.path.exists(save_pseudo_color_path):
        os.makedirs(save_pseudo_color_path)
    viz_op = VisualizeSegmm(save_pseudo_color_path, eval(_cfg.DATASETS).PALETTE)
    num_classes = len(eval(_cfg.DATASETS).LABEL_MAP)

    with torch.no_grad():
        _i = 0
        for ret, ret_gt in tqdm(pseudo_loader):
            # _i += 1
            # if _i >= 2:
            #     break

            ret = ret.cuda()
            cls = pre_slide(model, ret, num_classes=num_classes, tta=True) if slide else model(ret)  # (b, c, h, w)

            if save_prob:
                # np.save(save_pseudo_label_path + '/' + ret_gt['fname'][0] + '.npy',
                #         tnf.interpolate(cls, size, mode='bilinear', align_corners=True).squeeze(
                #             dim=0
                #         ).cpu().numpy())       # (c, h, w)
                torch.save(tnf.interpolate(cls, size, mode='bilinear', align_corners=True).squeeze(dim=0).cpu(),
                           save_pseudo_label_path + '/' + ret_gt['fname'][0] + '.pt')   # (c, h, w)
                if _cfg.SNAPSHOT_DIR is not None:
                    cls = pseudo_selection(cls, ignore_label=ignore_label,
                                           cutoff_top=_cfg.CUTOFF_TOP, cutoff_low=_cfg.CUTOFF_LOW)  # (b, h, w)
                    for fname, pred in zip(ret_gt['fname'], cls):
                        viz_op(pred, fname.replace('.tif', '.png'))
            else:
                # pseudo selection, from -1~6
                if _cfg.PSEUDO_SELECT:
                    lbl = pseudo_selection(cls, ignore_label=ignore_label)  # (b, h, w)
                    cls = lbl.cpu().numpy()  # (b, h, w)
                else:
                    cls = cls.argmax(dim=1).cpu().numpy()

                cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                            (cls + 1).reshape(*size).astype(np.uint8))

                if _cfg.SNAPSHOT_DIR is not None:
                    for fname, pred in zip(ret_gt['fname'], cls):
                        viz_op(pred, fname.replace('.tif', '.png'))


def analysis_pseudo_labels(label_dir='data/IsprsDA/Vaihingen/ann_dir/train',
                           pseudo_dir='log/GAST/2vaihingen20230910153121/pseudo_label',
                           ignore_label=-1, n_classes=6):
    labels = glob(label_dir + r'/*.png')
    pseudos = glob(pseudo_dir + r'/*.pt')
    assert len(labels) == len(pseudos)
    labels.sort()
    pseudos.sort()
    range_cnt = 100
    acc_list = np.zeros((range_cnt))
    cnt_used_list = np.zeros((range_cnt))
    cnt_true_list = np.zeros((range_cnt))
    for i in tqdm(range(len(labels))):
        lbl = cv2.imread(labels[i], cv2.IMREAD_UNCHANGED)
        lbl = torch.from_numpy(lbl).unsqueeze(0).to(torch.long).cuda()          # (1, h, w)
        gt = lbl.detach().clone()                                               # (1, h, w)
        lbl[lbl == ignore_label] = n_classes
        lbl_onehot = tnf.one_hot(lbl, num_classes=n_classes + 1)[:,:,:,:-1]     # (1, h, w, c)
        lbl_onehot = lbl_onehot.permute(0, 3, 1, 2)                             # (1, c, h, w)
        cls = torch.load(pseudos[i], map_location=torch.device('cpu')
                         ).unsqueeze(0).cuda()                                  # (1, c, h, w)
        pseudo = pseudo_selection(cls, cutoff_top=0.8, cutoff_low=0.6,
                                  return_type='tensor', ignore_label=-1)        # (1, h, w)
        pseudo[pseudo == ignore_label] = n_classes                              # (1, h, w)
        gradient = 1 - torch.sum(cls * lbl_onehot, dim=1)                       # (1, h, w)
        assert torch.sum(gradient < 0) == 0
        gradient[pseudo == n_classes] = 100.0
        for i in range(range_cnt):
            v_fr = 1.0 * i / range_cnt
            v_to = v_fr + 1.0 / range_cnt
            cnt_true, cnt_used, acc_local = range_static(gradient, pseudo, gt, v_fr, v_to, n_classes)
            cnt_used_list[i] = cnt_used_list[i] + cnt_used
            cnt_true_list[i] = cnt_true_list[i] + cnt_true
            acc_list[i] =  acc_list[i] + acc_local
    # cnt_true_list = cnt_true_list / len(labels)
    # cnt_used_list = cnt_used_list / len(labels)
    acc_list = acc_list / len(labels)
    x = [(1.0 * i / range_cnt) for i in range(range_cnt)]
    print(cnt_true_list)
    print(cnt_used_list)
    # print(cnt_used_list - cnt_true_list)
    print(1.0 * cnt_true_list / (1e-7 + cnt_used_list))
    print(acc_list)
    # cnt_true_list = np.where(cnt_true_list > 1, np.log10(cnt_true_list), 0)
    # cnt_used_list = np.where(cnt_used_list > 1, np.log10(cnt_used_list), 0)
    plot_noise_rate(x, cnt_true_list, cnt_used_list, acc_list)


def range_static(gradient, pseudo, gt, v_fr=0.0, v_to=1.0, n_classes=6):
    pseudo_range = pseudo.detach().clone()
    pseudo_range[(gradient < v_fr) | (gradient >= v_to)] = n_classes
    cnt_used = torch.sum(pseudo_range != n_classes)
    cnt_true = torch.sum(pseudo_range == gt)
    # print(cnt_used, cnt_true)
    return cnt_true, cnt_used, 1.0 * cnt_true / (cnt_used + 1e-7)


def plot_noise_rate(x, y1, y2, acc_list):
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    columns = ['psavert', 'uempmed']

    # Draw Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax.fill_between(x, y1=y1, y2=0, label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
    ax.fill_between(x, y1=y2, y2=0, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)

    # Decorations
    ax.set_title('Personal Savings Rate vs Median Duration of Unemployment', fontsize=18)
    # ax.set(ylim=[0, 30])
    ax.legend(loc='best', fontsize=12)
    # plt.xticks(x[::50], fontsize=10, horizontalalignment='center')
    # plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
    # plt.xlim(-10, x[-1])

    # # Draw Tick lines
    # for y in np.arange(2.5, 30.0, 2.5):
    #     plt.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()

if __name__ == "__main__":
    analysis_pseudo_labels(label_dir='data/IsprsDA/Vaihingen/ann_dir/train',
                           pseudo_dir='log/GAST/2vaihingen20230910153121/pseudo_label',
                           ignore_label=-1, n_classes=6)

    # analysis_pseudo_labels(label_dir='data/IsprsDA/Potsdam/ann_dir/train',
    #                        pseudo_dir='log/GAST/2potsdam20230910153254/pseudo_label',
    #                        ignore_label=-1, n_classes=6)

