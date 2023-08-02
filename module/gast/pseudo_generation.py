"""
@Project : rsda 
@File    : pseudo_generation.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/2/19 16:06
@e-mail  : liuwa@hnu.edu.cn
"""

import cv2
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
