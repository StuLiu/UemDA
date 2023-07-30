from module.datasets.daLoader import DALoader
import logging
logger = logging.getLogger(__name__)
from module.utils.tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm
from argparse import ArgumentParser
from module.datasets import *


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None, slide=True, tta=False):
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        viz_op = VisualizeSegmm(vis_dir, eval(cfg.DATASETS).PALETTE)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    num_class = len(eval(cfg.DATASETS).LABEL_MAP)
    model.eval()
    # # eval in target train datasets
    # print(cfg.EVAL_DATA_CONFIG)
    # eval_dataloader = DALoader(cfg.PSEUDO_DATA_CONFIG, cfg.DATASETS)
    # metric_op = er.metric.PixelMetric(len(eval(cfg.DATASETS).COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    # with torch.no_grad():
    #     for ret, ret_gt in tqdm(eval_dataloader):
    #         ret = ret.cuda()
    #         cls = pre_slide(model, ret, num_classes=num_class, tta=tta) if slide else model(ret)
    #         cls = cls.argmax(dim=1).cpu().numpy()
    #
    #         cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
    #         mask = cls_gt >= 0
    #
    #         y_true = cls_gt[mask].ravel()
    #         y_pred = cls[mask].ravel()
    #         metric_op.forward(y_true, y_pred)
    #
    #         # if cfg.SNAPSHOT_DIR is not None:
    #         #     for fname, pred in zip(ret_gt['fname'], cls):
    #         #         viz_op(pred, fname.replace('tif', 'png'))
    # metric_op.summary_all()
    # torch.cuda.empty_cache()

    eval_dataloader = DALoader(cfg.EVAL_DATA_CONFIG, cfg.DATASETS)
    metric_op = er.metric.PixelMetric(len(eval(cfg.DATASETS).COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.cuda()
            cls = pre_slide(model, ret, num_classes=num_class, tta=tta) if slide else model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()

            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0

            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))

    metric_op.summary_all()
    torch.cuda.empty_cache()



if __name__ == '__main__':
    seed_torch(2333)

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str, default='st.gast.2urban', help='config path')
    parser.add_argument('--ckpt-path', type=str, default='log/GAST/2urban_c_57.67_10000_40.67/URBAN10000.pth',
                        help='ckpt path')
    parser.add_argument('--multi-layer', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--tta', type=str2bool, default=False, help='save dir path')
    args = parser.parse_args()
    from module.models.Encoder import Deeplabv2
    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = os.path.dirname(args.ckpt_path)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='Baseline', logdir=log_dir)
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=args.multi_layer,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7,
        is_ins_norm=args.ins_norm
    )).cuda()
    evaluate(model, cfg, False, args.ckpt_path, logger, tta=args.tta)