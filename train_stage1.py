"""
@Filename: GAST_train
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""
import logging

import torch.multiprocessing

import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
from eval import evaluate
from module.utils.tools import *
from module.models.Encoder import Deeplabv2
from module.datasets.daLoader import DALoader
from module.datasets import LoveDA, IsprsDA
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
# from module.viz import VisualizeSegmm
from module.gast.alignment import Aligner
from module.gast.pseudo_generation import gener_target_pseudo, pseudo_selection
from module.gast.balance import *
from module.utils.ema import ExponentialMovingAverage
# from module.gast.domain_balance import examples_cnt, get_target_weight

# palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

# arg parser
# --config-path st.gast.2urban --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 0.5
# --config-path st.gast.2rural --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 1000

parser = argparse.ArgumentParser(description='Run GAST methods.')
parser.add_argument('--config-path', type=str, default='st.gast.2rural', help='config path')

parser.add_argument('--align-domain', type=str2bool, default=0, help='whether align domain or not')

parser.add_argument('--refine-label', type=str2bool, default=1, help='whether refine the pseudo label')
parser.add_argument('--refine-mode', type=str, default='all', choices=['s', 'p', 'n', 'l', 'all'],
                    help='refine by prototype, label, or both')
parser.add_argument('--refine-temp', type=float, default=2.0, help='whether refine the pseudo label')
# source loss
parser.add_argument('--ls', type=str, default="CrossEntropy",
                    choices=['CrossEntropy', 'OhemCrossEntropy'], help='source loss function')
parser.add_argument('--bcs', type=str2bool, default=0, help='whether balance class for source')
# target loss
parser.add_argument('--lt', type=str, default='uvem',
                    choices=['ours', 'uvem', 'ohem', 'focal', 'ghm', 'ups', 'none'], help='target loss function')
parser.add_argument('--bct', type=str2bool, default=1, help='whether balance class for target')
parser.add_argument('--class-temp', type=float, default=2.0, help='smooth factor')
parser.add_argument('--uvem-m', type=float, default=0, help='whether balance class')
parser.add_argument('--uvem-t', type=float, default=0.7, help='whether balance class')
parser.add_argument('--uvem-g', type=float, default=4, help='whether balance class')

parser.add_argument('--rm-pseudo', type=str2bool, default=1, help='remove pseudo label directory')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path)
assert cfg.FIRST_STAGE_STEP <= cfg.NUM_STEPS_STOP, 'FIRST_STAGE_STEP must no larger than NUM_STEPS_STOP'


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
    ema_model = ExponentialMovingAverage(model=model, decay=0.99)

    aligner = Aligner(logger=logger,
                      feat_channels=2048,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.996)

    class_balancer_s = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)
    class_balancer_t = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)

    loss_fn_s = eval(args.ls)(ignore_label=ignore_label, class_balancer=class_balancer_s if args.bcs else None)
    if args.lt in ['ours', 'uvem']:
        logger.info('>>>>>>> using ours/uvem_loss.')
        loss_fn_t = UVEMLoss(m=args.uvem_m,
                             threshold=args.uvem_t,
                             gamma=args.uvem_g,
                             class_balancer=class_balancer_t if args.bct else None,
                             class_num=class_num,
                             ignore_label=ignore_label)
    elif args.lt == 'ohem':
        logger.info('>>>>>>> using OhemCrossEntropy.')
        loss_fn_t = OhemCrossEntropy(ignore_label=ignore_label)
    elif args.lt == 'focal':
        logger.info('>>>>>>> using FocalLoss.')
        loss_fn_t = FocalLoss(gamma=2.0, reduction='mean', ignore_label=ignore_label)
    elif args.lt == 'ghm':
        logger.info('>>>>>>> using GHMLoss.')
        loss_fn_t = GHMLoss(bins=30, momentum=0.99, ignore_label=ignore_label)
    elif args.lt == 'ups':
        logger.info('>>>>>>> using UPSLoss.')
        loss_fn_t = UPSLoss(threshold=0.7, class_balancer=class_balancer_t if args.bct else None,
                            class_num=class_num, ignore_label=ignore_label)
    else:
        logger.info('>>>>>>> using CrossEntropyLoss.')
        loss_fn_t = CrossEntropy(ignore_label=ignore_label, class_balancer=class_balancer_t if args.bct else None)

    # source loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)

    # cnt_s, ratio_s = examples_cnt(sourceloader, ignore_label=ignore_label, save_prob=False)
    # logger.info(f'source domain valid examples cnt={cnt_s}, ratio={ratio_s}')

    # pseudo loader (target)
    pseudo_loader = DALoader(cfg.PSEUDO_DATA_CONFIG, cfg.DATASETS)

    # target loader
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)
    logger.info(f'batch num: source={len(sourceloader)}, target={len(targetloader)}, pseudo={len(pseudo_loader)}')
    # print(len(targetloader))
    epochs = cfg.NUM_STEPS_STOP / len(sourceloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    for i_iter in tqdm(range(cfg.FIRST_STAGE_STEP)):

        lmd_1 = portion_warmup(i_iter=i_iter, start_iter=0, end_iter=cfg.NUM_STEPS_STOP)
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        # Train with Source
        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        pred_s1, pred_s2, feat_s = model(images_s)

        # update prototypes
        aligner.update_prototype(feat_s, label_s)

        # target infer
        batch = targetloader_iter.next()
        images_t, _ = batch[0]
        images_t = images_t.cuda()
        _, _, feat_t = model(images_t)

        loss_seg = loss_calc([pred_s1, pred_s2], label_s, loss_fn=loss_fn_s, multi=True)

        loss_domain = aligner.align_domain(feat_s, feat_t) if args.align_domain else 0
        loss = loss_seg + lmd_1 * loss_domain

        optimizer.zero_grad()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                  max_norm=35, norm_type=2)
        optimizer.step()
        log_loss = f'iter={i_iter + 1}, total={loss:.3f}, loss_seg={loss_seg:.3f}, ' \
                   f'loss_domain={loss_domain:.3e}, ' \
                   f'lr={lr:.3e}, lmd_1={lmd_1:.3f}'

        # logging training process, evaluating and saving
        if i_iter == 0 or i_iter == cfg.FIRST_STAGE_STEP or (i_iter + 1) % 50 == 0:
            logger.info(log_loss)
            if args.bcs:
                logger.info(str(loss_fn_s.class_balancer))
            if args.bct and i_iter >= cfg.FIRST_STAGE_STEP:
                logger.info(str(loss_fn_t.class_balancer))

        if (i_iter + 1) % cfg.EVAL_EVERY == 0 and (i_iter + 1) >= cfg.EVAL_FROM:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter + 1) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            if i_iter + 1 > cfg.FIRST_STAGE_STEP:
                ema_model.apply_shadow()
                evaluate(model, cfg, True, ckpt_path, logger)
                torch.save(model.state_dict(), ckpt_path.replace('.pth', '_ema.pth'))
                ema_model.restore()
            model.train()
        elif (i_iter + 1) >= cfg.NUM_STEPS_STOP:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            # restore the ema model
            ema_model.apply_shadow()
            evaluate(model, cfg, True, ckpt_path, logger)
            ema_model.restore()
            break
        # if (i_iter + 1) >= 4000:
        #     break

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
