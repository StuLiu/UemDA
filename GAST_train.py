"""
@Project : Unsupervised_Domian_Adaptation
@File    : alignment.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/13 下午8:40
@e-mail  : 1183862787@qq.com
"""
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
import math

from eval import evaluate
from utils.tools import *
from module.Encoder import Deeplabv2
from module.alignment import Aligner
from data.loveda import LoveDALoader
from utils.tools import COLOR_MAP
from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
import ever as er
from skimage.io import imsave, imread

palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

# CUDA_VISIBLE_DEVICES=3 python GAST_train.py --config_path st.gast.2urban
parser = argparse.ArgumentParser(description='Run GAST methods.')

parser.add_argument('--config_path', type=str, help='config path')
parser.add_argument('--align-class', type=int, default=4500, help='the first iteration from which align the classes')
args = parser.parse_args()
cfg = import_config(args.config_path)


def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='GAST', logdir=cfg.SNAPSHOT_DIR)
    cudnn.enabled = True

    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(cfg.SNAPSHOT_DIR, 'stats')  # in 'save_path'

    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(save_pseudo_label_path, exist_ok=True)
    os.makedirs(save_stats_path, exist_ok=True)

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
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()
    aligner = Aligner(feat_channels=2048, class_num=7, ignore_label=255)
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    evalloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    targetloader = None
    targetloader_iter = None
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()
    # mix_trainloader = None

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        if i_iter < 5:#cfg.WARMUP_STEP:
            # Train with Source
            optimizer.zero_grad()
            batch = trainloader_iter.next()
            images_s, labels_s = batch[0]
            pred_source = model(images_s.cuda())
            pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
            # Segmentation Loss
            loss = loss_calc(pred_source, labels_s['cls'].cuda())
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()
            log_text = f'Mix iter={i_iter}, lr={lr:.3f}, loss_seg={loss:.3f}, ' \
                       f'loss_domain={0:.3f}, loss_class={0:.3f}, loss={loss:.3f}'
        else:
            if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
                logger.info('###### Start generate pesudo dataset in round {}! ######'.format(i_iter))
                save_round_eval_path = osp.join(cfg.SNAPSHOT_DIR, str(i_iter))
                save_pseudo_label_color_path = osp.join(save_round_eval_path, 'pseudo_label_color')
                os.makedirs(save_round_eval_path, exist_ok=True)
                os.makedirs(save_pseudo_label_color_path, exist_ok=True)

                # evaluation & save confidence vectors
                conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list = val(
                    model, evalloader,
                    save_round_eval_path,
                    cfg
                )
                # class-balanced thresholds
                tgt_portion = min(cfg.TGT_PORTION + cfg.TGT_PORTION_STEP, cfg.MAX_TGT_PORTION)
                cls_thresh = kc_parameters(conf_dict, pred_cls_num, tgt_portion, i_iter, save_stats_path, cfg, logger)
                print('CLS THRESH', cls_thresh)
                # pseudo-label maps generation
                label_generation(cls_thresh, image_name_tgt_list, i_iter, save_prob_path, save_pred_path,
                                 save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, logger)
                # model retraining
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_pseudo_label_path]
                logger.info(target_config)
                targetloader = LoveDALoader(target_config)
                targetloader_iter = Iterator(targetloader)
                logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))

            # if targetloader is None:
            #     target_config = cfg.SOURCE_DATA_CONFIG
            #     logger.info(target_config)
            #     targetloader = LoveDALoader(target_config)
            #     targetloader_iter = Iterator(targetloader)

            # train model using both domains
            model.train()
            lr = adjust_learning_rate(optimizer, i_iter, cfg)

            batch = trainloader_iter.next()
            images_s, labels_s = batch[0]
            logits_s1, logits_s2, feat_x16_s = model(images_s.cuda())
            batch = targetloader_iter.next()
            images_t, labels_t = batch[0]
            logits_t1, logits_t2, feat_x16_t = model(images_t.cuda())

            loss_source = loss_calc(logits_s1, labels_s['cls'].cuda()) + loss_calc(logits_s2, labels_s['cls'].cuda())
            loss_target = loss_calc(logits_t1, labels_t['cls'].cuda()) + loss_calc(logits_t2, labels_t['cls'].cuda())
            loss_seg = cfg.SOURCE_LOSS_WEIGHT * loss_source + cfg.PSEUDO_LOSS_WEIGHT * loss_target
            loss_domain = aligner.align_domain(feat_x16_s, feat_x16_t)
            if i_iter >= 0:#cfg.align_class:
                loss_class = aligner.align_category(feat_x16_s, labels_s['cls'], feat_x16_t, labels_t['cls'])
            else:
                loss_class = 0
            loss = loss_seg + loss_domain + loss_class
            optimizer.zero_grad()
            loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()
            log_text = f'Mix iter={i_iter}, lr={lr:.3f}, loss_seg={loss_seg:.3f}, ' \
                       f'loss_domain={loss_domain:.3f}, loss_class={loss_class:.3f}, loss={loss:.3f}'

        # evaluating
        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(log_text)
        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            break
        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            model.train()


def val(model, targetloader, save_round_eval_path, _cfg):
    """Create the model and start the evaluation process."""

    model.eval()
    # output folder
    # save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')

    # viz_op = er.viz.VisualizeSegmm(save_pred_vis_path, palette)
    # metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=_cfg.SNAPSHOT_DIR, logger=logger)

    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    # saving output data
    conf_dict = {k: [] for k in range(_cfg.NUM_CLASSES)}
    pred_cls_num = np.zeros(_cfg.NUM_CLASSES)
    # evaluation process
    image_name_tgt_list = []
    with torch.no_grad():
        for batch in tqdm(targetloader):
            images, labels = batch
            output = model(images.cuda()).softmax(dim=1)
            output = output[0] if isinstance(output, tuple) else output
            pred_label = output.argmax(dim=1).cpu().numpy()
            output = output.cpu().numpy()
            for fname, pred_i, out_i in zip(labels['fname'], pred_label, output):
                image_name_tgt_list.append(fname.split('.')[0])
                # save prob
                # viz_op(pred_i, fname)
                np.save('%s/%s' % (save_prob_path, fname.replace('png', 'npy')), out_i)
                imsave('%s/%s' % (save_pred_path, fname), pred_i.astype(np.uint8), check_contrast=False)
                out_i = out_i.transpose(1, 2, 0)
                conf_i = np.amax(out_i, axis=2)
                # save class-wise confidence maps
                if _cfg.KC_VALUE == 'conf':
                    for idx_cls in range(_cfg.NUM_CLASSES):
                        idx_temp = pred_i == idx_cls
                        pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                        if idx_temp.any():
                            conf_cls_temp = conf_i[idx_temp].astype(np.float32)
                            len_cls_temp = conf_cls_temp.size
                            # downsampling by ds_rate
                            conf_cls = conf_cls_temp[0:len_cls_temp:_cfg.DS_RATE]
                            conf_dict[idx_cls].extend(conf_cls)
    # return the dictionary containing all the class-wise confidence vectors
    return conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list


def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, _cfg, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    cls_thresh = np.ones(_cfg.NUM_CLASSES, dtype=np.float32)
    cls_sel_size = np.zeros(_cfg.NUM_CLASSES, dtype=np.float32)
    cls_size = np.zeros(_cfg.NUM_CLASSES, dtype=np.float32)
    # if _cfg.KC_POLICY == 'cb' and _cfg.KC_VALUE == 'conf':
    for idx_cls in np.arange(0, _cfg.NUM_CLASSES):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] is not None:
            conf_dict[idx_cls].sort(reverse=True)  # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]
            conf_dict[idx_cls] = None

    # threshold for mine_id with priority
    num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < _cfg.MINE_PORT)[0])
    # chose the smallest mine_id
    id_all = np.argsort(cls_size / np.sum(cls_size))
    rare_id = id_all[:_cfg.RARE_CLS_NUM]
    mine_id = id_all[:num_mine_id]  # sort mine_id in ascending order w.r.t predication portions
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    logger.info('Mining ids : {}! {} rarest ids: {}!'.format(mine_id, _cfg.RARE_CLS_NUM, rare_id))
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info(f'###### Finish kc generation in round {round_idx}! '
                f'Time cost: {time.time() - start_kc:.2f}seconds. ######')
    return cls_thresh


def label_generation(cls_thresh, image_name_tgt_list, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path,
                    save_pseudo_label_color_path, save_round_eval_path, logger):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    # viz_op = er.viz.VisualizeSegmm(save_pseudo_label_color_path, palette)
    for sample_name in image_name_tgt_list:
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_prob = np.load(probmap_path)
        weighted_prob = pred_prob / cls_thresh[:, None, None]
        weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=0), dtype=np.uint8)
        weighted_conf = np.amax(weighted_prob, axis=0)
        pred_label_trainIDs = weighted_pred_trainIDs.copy()
        pred_label_labelIDs = pred_label_trainIDs + 1

        pred_label_labelIDs[weighted_conf < 1] = 0  # '0' in LoveDA Dataset ignore
        # pseudo-labels with labelID
        # viz_op(pred_label_trainIDs, '%s_color.png' % sample_name)

        # save pseudo-label map with label IDs
        imsave(os.path.join(save_pseudo_label_path, '%s.png' % sample_name), pred_label_labelIDs, check_contrast=False)

    # remove probability maps
    if cfg.RM_PROB:
        shutil.rmtree(save_prob_path)

    logger.info(f'###### Finish pseudo-label generation in round {round_idx}! '
                f'Time cost: {time.time() - start_pl:.2f} seconds. ######')


if __name__ == '__main__':
    seed_torch(2333)
    main()
