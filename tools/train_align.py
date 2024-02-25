"""
@Filename:
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""
import os
import time
import torch
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim

from uemda.utils.eval import evaluate

from tqdm import tqdm
from torch.nn.utils import clip_grad
from ever.core.iterator import Iterator
from uemda.datasets import *
from uemda.gast.alignment import Aligner
from uemda.gast.balance import *
from uemda.utils.tools import *
from uemda.models.Encoder import Deeplabv2
from uemda.datasets.daLoader import DALoader
from uemda.loss import PrototypeContrastiveLoss
from uemda.gast.pseudo_generation import pseudo_selection

# CUDA_VISIBLE_DEVICES=6 python tools/train_align.py --config-path st.gast.2potsdam \
# --ckpt-model log/proca/2potsdam/src/Potsdam_best.pth
# align instance to flow-prototypes of two domain
parser = argparse.ArgumentParser(description='Train align by pcl.')

parser.add_argument('--config-path', type=str, default='st.proca.2vaihingen', help='config path')

parser.add_argument('--ckpt-model', type=str,
                    default='log/proca/2vaihingen/src/Vaihingen_best.pth', help='model ckpt from stage1')
parser.add_argument('--ckpt-proto', type=str,
                    default='log/proca/2vaihingen/src/prototypes_best.pth', help='proto ckpt from stage1')
parser.add_argument('--align-domain', type=str2bool, default=0, help='whether align domain or not')
# source loss
parser.add_argument('--ls', type=str, default="CrossEntropy",
                    choices=['CrossEntropy', 'OhemCrossEntropy'], help='source loss function')
parser.add_argument('--bcs', type=str2bool, default=0, help='whether balance class for source')
parser.add_argument('--class-temp', type=float, default=2.0, help='smooth factor')
# alignment
parser.add_argument('--pcl-temp', type=float, default=8.0, help='loss factor')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path, create=True, copy=True, postfix='/align')


def main():
    time_from = time.time()

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    stop_steps = cfg.STAGE2_STEPS
    cfg.NUM_STEPS = stop_steps * 1.5            # for learning rate poly
    cfg.PREHEAT_STEPS = int(stop_steps / 20)    # for warm-up

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
    ))
    ckpt_model = torch.load(args.ckpt_model, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_model)
    model = model.cuda()

    aligner = Aligner(logger=logger,
                      feat_channels=2048,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.996,
                      resume=args.ckpt_proto)

    class_balancer_s = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)

    loss_fn_s = eval(args.ls)(ignore_label=ignore_label, class_balancer=class_balancer_s if args.bcs else None)
    loss_fn_pcl = PrototypeContrastiveLoss(temperature=args.pcl_temp, ignore_label=ignore_label)

    # source and target loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)

    epochs = stop_steps / len(sourceloader)
    logger.info(f'batch num: source={len(sourceloader)}, target={len(targetloader)}')
    logger.info('epochs ~= %.3f' % epochs)

    mIoU_max, iter_max = 0, 0

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    for i_iter in tqdm(range(stop_steps)):

        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        pred_s1, pred_s2, feat_s = model(images_s)

        # ema-updating prototypes
        label_s_down = aligner.update_prototype(feat_s, label_s)

        # target infer
        batch = targetloader_iter.next()
        images_t, _ = batch[0]
        images_t = images_t.cuda()
        pred_t1, pred_t2, feat_t = model(images_t)

        label_t_soft = (torch.softmax(pred_t1, dim=1) + torch.softmax(pred_t2, dim=1)) * 0.5
        label_t_val, label_t = torch.max(label_t_soft, dim=1)
        label_t[label_t_val < 0.9] = ignore_label

        # compute loss
        loss_seg = loss_calc([pred_s1, pred_s2], label_s, loss_fn=loss_fn_s, multi=True)
        loss_domain = aligner.align_domain(feat_s, feat_t) if args.align_domain else 0
        loss_align = (loss_fn_pcl(aligner.prototypes, feat_s, label_s_down) +
                      loss_fn_pcl(aligner.prototypes, feat_t, label_t)) * 0.5
        loss = loss_seg + loss_domain + loss_align

        optimizer.zero_grad()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                  max_norm=32, norm_type=2)
        optimizer.step()
        log_loss = f'iter={i_iter + 1}, total={loss:.3f}, loss_seg={loss_seg:.3f}, ' \
                   f'loss_align={loss_align:.3e}, loss_domain={loss_domain:.3e} lr={lr:.3e}'

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            logger.info(log_loss)
            if args.bcs:
                logger.info(str(loss_fn_s.class_balancer))

        if i_iter == 0 or (i_iter + 1) % cfg.EVAL_EVERY == 0 or (i_iter + 1) >= stop_steps:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_curr.pth')
            torch.save(model.state_dict(), ckpt_path)
            _, mIoU_curr = evaluate(model, cfg, True, ckpt_path, logger)
            if mIoU_max <= mIoU_curr:
                mIoU_max = mIoU_curr
                iter_max = i_iter + 1
                torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_best.pth'))
                if osp.isdir(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best')):
                    shutil.rmtree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
                shutil.copytree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{os.path.basename(ckpt_path)}'),
                                os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
            logger.info(f'Best model in iter={iter_max}, best_mIoU={mIoU_max}.')
            model.train()

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
