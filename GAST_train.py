# -*- coding:utf-8 -*-

# @Filename: GAST_train
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-10 19:14
# @Author  : Linshan
# @update  : 2023-03-16 21:55
# @Author  : WangLiu

import cv2
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math
from eval import evaluate
from utils.tools import *
from module.Encoder import Deeplabv2
from module.dca_modules import *
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
import ever as er
from skimage.io import imsave, imread
from module.viz import VisualizeSegmm
from module.alignment import Aligner

palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
parser = argparse.ArgumentParser(description='Run GAST methods.')
parser.add_argument('--config-path', type=str, default='st.gast.2urban', help='config path')
parser.add_argument('--align-class', type=int, default=None, help='the first iteration from which align the classes')
parser.add_argument('--gpu', type=lambda v: int(v) >= 0, default=0, help='device to inference')
args = parser.parse_args()
cfg = import_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}')


def main():
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(save_pseudo_label_path, exist_ok=True)

    logger = get_console_file_logger(name='MY', logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    cudnn.enabled = True
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
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=7
    )).to(device)
    aligner = Aligner(feat_channels=2048, class_num=7, ignore_label=-1, device=device)
    # source loader
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    # eval loader (target)
    evalloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    # target loader
    targetloader = None
    targetloader_iter = None

    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        if i_iter < cfg.FIRST_STAGE_STEP:
            # Train with Source
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, cfg)
            batch = trainloader_iter.next()
            images_s, labels_s = batch[0]
            preds1, preds2, feats = model(images_s.to(device))

            # Loss: segmentation + regularization
            loss_seg = loss_calc([preds1, preds2], labels_s['cls'].to(device), multi=True)
            loss = loss_seg

            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                      max_norm=35, norm_type=2)
            optimizer.step()

            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                text = 'iter = %d, total = %.3f, seg = %.3f, ' \
                       'lr = %.3f' % (
                           i_iter, loss, loss_seg, lr)
                logger.info(text)

            if i_iter >= cfg.NUM_STEPS_STOP - 1:
                print('save model ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                break
            if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                model.train()
        else:
            # Second Stage
            # Generate pseudo label
            if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                # save pseudo label for target domain
                gener_target_pseudo(cfg, model, evalloader, save_pseudo_label_path)
                # save finish
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_pseudo_label_path]
                logger.info(target_config)
                targetloader = LoveDALoader(target_config)
                targetloader_iter = Iterator(targetloader)
                logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))
            if i_iter == (cfg.FIRST_STAGE_STEP + 1):
                logger.info('###### Start the Second Stage in round {}! ######'.format(i_iter))
            # if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
            #     target_config = cfg.SOURCE_DATA_CONFIG
            #     logger.info(target_config)
            #     targetloader = LoveDALoader(target_config)
            #     targetloader_iter = Iterator(targetloader)
            torch.cuda.synchronize()
            # Second Stage
            if i_iter < cfg.NUM_STEPS_STOP and targetloader is not None:
                model.train()
                lr = adjust_learning_rate(optimizer, i_iter, cfg)

                # source output
                batch_s = trainloader_iter.next()
                images_s, label_s = batch_s[0]
                images_s, lab_s = images_s.to(device), label_s['cls'].to(device)
                # target output
                batch_t = targetloader_iter.next()
                images_t, label_t = batch_t[0]
                images_t, lab_t = images_t.to(device), label_t['cls'].to(device)

                # model forward
                # source
                pred_s1, pred_s2, feat_s = model(images_s)
                # target
                pred_t1, pred_t2, feat_t = model(images_t)

                # loss
                loss_seg = loss_calc([pred_s1, pred_s2], lab_s, multi=True)
                loss_pseudo = loss_calc([pred_t1, pred_t2], lab_t, multi=True)
                loss_domain = aligner.align_domain(feat_s, feat_t)
                loss_class = aligner.align_category(feat_s, label_s['cls'], feat_t, label_t['cls']) \
                    if i_iter >= (args.align_class if args.align_class else cfg.ALIGN_CLASS) else 0
                loss = loss_seg + loss_pseudo + loss_domain + loss_class

                optimizer.zero_grad()
                loss.backward()
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          max_norm=35, norm_type=2)
                optimizer.step()

                if i_iter % 50 == 0:
                    logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                    text = f'iter={i_iter:d}, total={loss:.3f}, seg={loss_seg:.3f}, pseudo={loss_pseudo:.3f},' \
                           f'domain={loss_domain:.3f}, class={loss_class:.3f}, lr = {lr:.3f}'
                    logger.info(text)

                if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                    ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    # evaluate_nj(model, cfg, True, ckpt_path, logger)
                    evaluate(model, cfg, True, ckpt_path, logger)
                    model.train()


def gener_target_pseudo(cfg, model, evalloader, save_pseudo_label_path, slide=True):
    model.eval()

    save_pseudo_color_path = save_pseudo_label_path + '_color'
    if not os.path.exists(save_pseudo_color_path):
        os.makedirs(save_pseudo_color_path)
    viz_op = VisualizeSegmm(save_pseudo_color_path, palette)

    with torch.no_grad():
        for ret, ret_gt in tqdm(evalloader):
            ret = ret.to(device)

            cls = pre_slide(model, ret, tta=True) if slide else model(ret)
            # cls = pre_slide(model, ret, tta=True)
            # pseudo selection, from -1~6
            if cfg.PSEUDO_SELECT:
                cls = pseudo_selection(cls)
            else:
                cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                        (cls + 1).reshape(1024, 1024).astype(np.uint8))

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6):
    """Convert continuous mask into binary mask"""
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


if __name__ == '__main__':
    seed_torch(2333)
    main()
