# -*- coding:utf-8 -*-

# @Filename: DCA_train
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-10 19:14
# @Author  : Linshan
# @update  : 2023-03-16 21:55
# @Author  : WangLiu

import cv2
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
from uemda.utils.eval import evaluate
from uemda.utils.tools import *
from uemda.models.Encoder import Deeplabv2
from uemda.dca_modules import *
from uemda.datasets.daLoader import DALoader
from uemda.datasets import *
# from uemda.utils.tools import COLOR_MAP
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
from uemda.viz import VisualizeSegmm
from uemda.gast.pseudo_generation import gener_target_pseudo

parser = argparse.ArgumentParser(description='Run DCA methods.')
parser.add_argument('--config-path', type=str, default='st.dca.2vaihingen',
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(save_pseudo_label_path, exist_ok=True)

    logger = get_console_file_logger(name='DCA', logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)

    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    cudnn.enabled = True

    model = Deeplabv2(dict(
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
    )).cuda()

    from uemda.gast.balance import CrossEntropy
    loss_fn = CrossEntropy(ignore_label=ignore_label, class_balancer=None)

    # palette = np.asarray(list(eval(cfg.DATASETS).COLOR_MAP.values())).reshape((-1,)).tolist()
    # source loader
    trainloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    trainloader_iter = Iterator(trainloader)
    # eval loader (target)
    pseudo_loader = DALoader(cfg.PSEUDO_DATA_CONFIG, cfg.DATASETS)
    # target loader
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)

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
            preds1, preds2, feats = model(images_s.cuda())

            # Loss: segmentation + regularization
            loss_seg = loss_calc([preds1, preds2], labels_s['cls'].cuda(), loss_fn, multi=True)
            source_intra = ICR([preds1, preds2, feats], num_classes=class_num,
                               multi_layer=True)
            loss = loss_seg + source_intra

            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                      max_norm=35, norm_type=2)
            optimizer.step()

            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                text = 'iter = %d, total = %.3f, seg = %.3f, ' \
                       'source_intra = %.3f, lr = %.3f' % (
                           i_iter, loss, loss_seg, source_intra, lr)
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
            if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or i_iter == cfg.FIRST_STAGE_STEP:
                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                # save pseudo label for target domain
                if i_iter != cfg.FIRST_STAGE_STEP:
                    shutil.move(f'{save_pseudo_label_path}_color',
                                f'{save_pseudo_label_path}_color_{i_iter - cfg.GENERATE_PSEDO_EVERY}')
                gener_target_pseudo(cfg, model, pseudo_loader, save_pseudo_label_path,
                                    size=eval(cfg.DATASETS).SIZE, save_prob=False, slide=True,
                                    ignore_label=ignore_label)
                # save finish
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_pseudo_label_path]
                logger.info(target_config)
                targetloader = DALoader(target_config, cfg.DATASETS)
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
            if i_iter < cfg.NUM_STEPS_STOP:
                model.train()
                lr = adjust_learning_rate(optimizer, i_iter, cfg)

                # source output
                batch_s = trainloader_iter.next()
                images_s, label_s = batch_s[0]
                images_s, lab_s = images_s.cuda(), label_s['cls'].cuda()
                # target output
                batch_t = targetloader_iter.next()
                images_t, label_t = batch_t[0]
                images_t, lab_t = images_t.cuda(), label_t['cls'].cuda()

                # model forward
                # source
                pred_s1, pred_s2, feat_s = model(images_s)
                # target
                pred_t1, pred_t2, feat_t = model(images_t)

                # loss
                loss_seg = loss_calc([pred_s1, pred_s2], lab_s, loss_fn, multi=True)
                loss_pseudo = loss_calc([pred_t1, pred_t2], lab_t, loss_fn, multi=True)

                source_intra = ICR([pred_s1, pred_s2, feat_s], num_classes=class_num,
                                   multi_layer=True)
                # target_intra = intra_domain_regularize([pred_t1, feat_t1, pred_t2, feat_t2],
                #                                        multi_layer=True)

                domain_cross = CCR([pred_s1, pred_s2, feat_s],
                                   [pred_t1, pred_t2, feat_t],
                                   num_classes=class_num,
                                   multi_layer=True)

                loss = loss_seg + loss_pseudo #+ (source_intra + domain_cross)

                optimizer.zero_grad()
                loss.backward()
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          max_norm=35, norm_type=2)
                optimizer.step()

                if i_iter % 50 == 0:
                    logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                    text = 'iter = %d, total = %.3f, seg = %.3f, pseudo = %.3f, ' \
                           'sour_intra = %.3f, cross = %.3f, lr = %.3f' % \
                           (i_iter, loss, loss_seg, loss_pseudo,
                            source_intra, domain_cross, lr)
                    logger.info(text)

                if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                    ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    # evaluate_nj(model, cfg, True, ckpt_path, logger)
                    evaluate(model, cfg, True, ckpt_path, logger)
                    model.train()


# def gener_target_pseudo(cfg, model, pseudo_loader, save_pseudo_label_path, palette, slide=True):
#     model.eval()
#
#     save_pseudo_color_path = save_pseudo_label_path + '_color'
#     if not os.path.exists(save_pseudo_color_path):
#         os.makedirs(save_pseudo_color_path)
#     viz_op = VisualizeSegmm(save_pseudo_color_path, palette)
#     num_classes = len(eval(cfg.DATASETS).LABEL_MAP)
#     img_size = eval(cfg.DATASETS).SIZE
#     with torch.no_grad():
#         for ret, ret_gt in tqdm(pseudo_loader):
#             ret = ret.to(torch.device('cuda'))
#
#             # cls = model(ret)
#             cls = pre_slide(model, ret, num_classes=num_classes, tta=True) if slide else model(ret)
#             # pseudo selection, from -1~6
#             if cfg.PSEUDO_SELECT:
#                 cls = pseudo_selection(cls)
#             else:
#                 cls = cls.argmax(dim=1).cpu().numpy()
#
#             cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
#                         (cls + 1).reshape(*img_size).astype(np.uint8))
#
#             if cfg.SNAPSHOT_DIR is not None:
#                 for fname, pred in zip(ret_gt['fname'], cls):
#                     viz_op(pred, fname.replace('tif', 'png'))
#
#
# def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6):
#     """Convert continuous mask into binary mask"""
#     assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
#     bs, c, h, w = mask.size()
#     mask = mask.view(bs, c, -1)
#
#     # for each class extract the max confidence
#     mask_max, _ = mask.max(-1, keepdim=True)
#     mask_max *= cutoff_top
#
#     # if the top score is too low, ignore it
#     lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
#     mask_max = mask_max.max(lowest)
#
#     pseudo_gt = (mask > mask_max).type_as(mask)
#     # remove ambiguous pixels, ambiguous = 1 means ignore
#     ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)
#
#     pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
#     pseudo_gt[ambiguous == 1] = -1
#
#     return pseudo_gt.view(bs, h, w).cpu().numpy()


if __name__ == '__main__':
    seed_torch(2333)
    main()
