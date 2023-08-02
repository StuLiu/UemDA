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
from module.gast.class_balance import ClassBalanceLoss, FocalLoss
from module.utils.ema import ExponentialMovingAverage
from module.gast.domain_balance import examples_cnt, get_target_weight


# palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

# arg parser
# --config-path st.gast.2urban --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 0.5
# --config-path st.gast.2rural --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 1000

parser = argparse.ArgumentParser(description='Run GAST methods.')
parser.add_argument('--config-path', type=str, default='st.gast.2rural', help='config path')

parser.add_argument('--align-domain', type=str2bool, default=0, help='whether align domain or not')

parser.add_argument('--refine-label', type=str2bool, default=1, help='whether refine the pseudo label or not')
parser.add_argument('--refine-mode', type=str, default='all', help='whether refine the pseudo label or not')
parser.add_argument('--refine-temp', type=float, default=2.0, help='whether refine the pseudo label or not')

parser.add_argument('--balance-class', type=str2bool, default=0, help='whether balance class or not')
parser.add_argument('--balance-temp', type=float, default=0.5, help='whether balance class or not')
parser.add_argument('--balance-type', type=str, default='focal', help='focal, ohem, or ours')

# parser.add_argument('--balance-domain', type=str2bool, default=0, help='whether balance domain examples or not')

parser.add_argument('--rm-pseudo', type=str2bool, default=1, help='remove pseudo label directory')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path)
assert cfg.FIRST_STAGE_STEP <= cfg.NUM_STEPS_STOP, 'FIRST_STAGE_STEP must no larger than NUM_STEPS_STOP'
assert args.balance_type in ['ours', 'focal', 'ohem']

def main():
    time_from = time.time()
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(save_pseudo_label_path, exist_ok=True)

    logger = get_console_file_logger(name='GAST', logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    # logger.info(args)
    # logger.info(cfg)
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)

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
            num_classes=class_num,
            use_aux=False,
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=True,
    )).cuda()
    ema_model = ExponentialMovingAverage(model=model, decay=0.99)
    aligner = Aligner(logger=logger, feat_channels=2048, class_num=class_num,
                      ignore_label=ignore_label, decay=0.996)
    if args.balance_type == 'ours':
        cb_loss_s = ClassBalanceLoss(class_num=class_num, ignore_label=ignore_label, decay=0.996,
                                     is_balance=args.balance_class, temperature=args.balance_temp)
        cb_loss_t = ClassBalanceLoss(class_num=class_num, ignore_label=ignore_label, decay=0.996,
                                     is_balance=args.balance_class, temperature=args.balance_temp)
    elif args.balance_type == 'focal':
        cb_loss_s = FocalLoss(gamma=2.0, reduction='mean')
        cb_loss_t = FocalLoss(gamma=2.0, reduction='mean')
    else:
        exit(-1)
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

    balance_domain_weight = 1.0
    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):

        lmd_1 = portion_warmup(i_iter=i_iter, start_iter=0, end_iter=cfg.NUM_STEPS_STOP)
        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        if i_iter < cfg.FIRST_STAGE_STEP:
            # Train with Source

            # source infer
            batch = sourceloader_iter.next()
            images_s, label_s = batch[0]
            images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
            pred_s1, pred_s2, feat_s = model(images_s)

            # update prototypes
            # print(label_s.unique())
            aligner.update_prototype(feat_s, label_s)

            # target infer
            batch = targetloader_iter.next()
            images_t, _ = batch[0]
            images_t = images_t.cuda()
            _, _, feat_t = model(images_t)

            # Loss: source segmentation + global alignment
            loss_seg = loss_calc([pred_s1, pred_s2], label_s, reduction='none', multi=True)
            loss_seg = cb_loss_s(loss_seg, label_s)

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
        else:
            log_loss = ''
            lmd_2 = portion_warmup(i_iter=i_iter, start_iter=cfg.FIRST_STAGE_STEP, end_iter=cfg.NUM_STEPS_STOP)
            # Second Stage
            if i_iter == cfg.FIRST_STAGE_STEP:
                logger.info('###### Start the Second Stage in round {}! ######'.format(i_iter))
                ema_model.register()
            # Generate pseudo label
            if i_iter == cfg.FIRST_STAGE_STEP or i_iter % cfg.GENERATE_PSEDO_EVERY == 0:
                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                ema_model.apply_shadow()
                # save pseudo label for target domain
                gener_target_pseudo(cfg, model, pseudo_loader, save_pseudo_label_path,
                                                     size=eval(cfg.DATASETS).SIZE, save_prob=True, slide=True,
                                                     ignore_label=ignore_label)
                ema_model.restore()

                # save finish
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_pseudo_label_path]
                logger.info(target_config)
                targetloader = DALoader(target_config, cfg.DATASETS)
                targetloader_iter = Iterator(targetloader)
                logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))
            torch.cuda.synchronize()

            # Train with source and target domain
            if i_iter < cfg.NUM_STEPS_STOP:
                model.train()
                # source output
                batch_s = sourceloader_iter.next()
                images_s, label_s = batch_s[0]
                images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
                # target output
                batch_t = targetloader_iter.next()
                images_t, label_t_soft = batch_t[0]
                images_t, label_t_soft = images_t.cuda(), label_t_soft['cls'].cuda()

                # model forward
                # source
                pred_s1, pred_s2, feat_s = model(images_s)
                # target
                pred_t1, pred_t2, feat_t = model(images_t)

                label_t_soft = aligner.label_refine(feat_t, [pred_t1, pred_t2], label_t_soft,
                                                    refine=args.refine_label,
                                                    mode=args.refine_mode,
                                                    temp=args.refine_temp)
                label_t_hard = pseudo_selection(label_t_soft, cutoff_top=cfg.CUTOFF_TOP, cutoff_low=cfg.CUTOFF_LOW,
                                                return_type='tensor', ignore_label=ignore_label)
                # logger.info(np.unique(label_t_hard.cpu().numpy()))
                # aligner.update_prototype(feat_s, label_s)
                aligner.update_prototype(feat_t, label_t_hard)

                # loss
                loss_source = loss_calc([pred_s1, pred_s2], label_s, reduction='none', multi=True)
                loss_source = cb_loss_s(loss_source, label_s)  # balance op
                loss_pseudo = loss_calc([pred_t1, pred_t2], label_t_hard, reduction='none', multi=True)
                loss_pseudo = cb_loss_t(loss_pseudo, label_t_hard)  # balance op
                loss_domain = aligner.align_domain(feat_s, feat_t) if args.align_domain else 0
                loss = loss_source + lmd_2 * loss_pseudo + lmd_1 * loss_domain

                optimizer.zero_grad()
                loss.backward()
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          max_norm=35, norm_type=2)
                optimizer.step()
                log_loss = f'iter={i_iter + 1}, total={loss:.3f}, source={loss_source:.3f}, pseudo={loss_pseudo:.3f},' \
                           f' domain={loss_domain:.3e},' \
                           f' lr = {lr:.3e}, lmd_1={lmd_1:.3f}, lmd_2={lmd_2:.3f}'

                ema_model.update()

        # logging training process, evaluating and saving
        if i_iter == 0 or i_iter == cfg.FIRST_STAGE_STEP or (i_iter + 1) % 50 == 0:
            # logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(log_loss)
        if (i_iter + 1) % cfg.EVAL_EVERY == 0 and (i_iter + 1) >= cfg.EVAL_FROM:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter + 1) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            if i_iter + 1 > cfg.FIRST_STAGE_STEP:
                ema_model.apply_shadow()
                evaluate(model, cfg, True, ckpt_path, logger)
                ema_model.restore()
            model.train()
        elif (i_iter + 1) >= cfg.NUM_STEPS_STOP:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)

            ema_model.apply_shadow()
            evaluate(model, cfg, True, ckpt_path, logger)
            ema_model.restore()
            break

    if args.rm_pseudo:
        logger.info('removing pseudo labels begin >>>>>>>>>>>>')
        shutil.rmtree(save_pseudo_label_path, ignore_errors=True)
        logger.info('removing pseudo labels end <<<<<<<<<<<<<<')

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
