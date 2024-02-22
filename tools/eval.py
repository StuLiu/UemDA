import os

from uemda.datasets.daLoader import DALoader
import logging
logger = logging.getLogger(__name__)
from uemda.utils.tools import *
from ever.util.param_util import count_model_parameters
from uemda.viz import VisualizeSegmm
from argparse import ArgumentParser
from uemda.datasets import *
from uemda.gast.metrics import PixelMetricIgnore


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None, slide=True, tta=False, test=False):
    ignore_labels = []
    if cfg.DATASETS == 'IsprsDA':
        ignore_labels = [0]
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
    viz_op = VisualizeSegmm(vis_dir, eval(cfg.DATASETS).PALETTE)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    num_class = len(eval(cfg.DATASETS).LABEL_MAP)
    model.eval()

    if test:
        eval_dataloader = DALoader(cfg.TEST_DATA_CONFIG, cfg.DATASETS)
    else:
        eval_dataloader = DALoader(cfg.EVAL_DATA_CONFIG, cfg.DATASETS)

    class_names = eval(cfg.DATASETS).COLOR_MAP.keys()
    metric_op = PixelMetricIgnore(len(class_names), class_names=list(class_names),
                                  logdir=cfg.SNAPSHOT_DIR, logger=logger,
                                  ignore_labels=ignore_labels)
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

    torch.cuda.empty_cache()
    return metric_op.summary_all()


if __name__ == '__main__':
    seed_torch(2333)

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str, default='st.gast.2urban', help='config path')
    parser.add_argument('--ckpt-path', type=str, default='log/GAST/2urban_c_57.67_10000_40.67/URBAN10000.pth',
                        help='ckpt path')
    parser.add_argument('--multi-layer', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='is instance norm in net end?')
    parser.add_argument('--test', type=str2bool, default=False, help='evaluate the test set?')
    parser.add_argument('--tta', type=str2bool, default=False, help='save dir path')
    args = parser.parse_args()
    from uemda.models.Encoder import Deeplabv2
    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = os.path.dirname(args.ckpt_path)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='Baseline', logdir=log_dir)

    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=args.multi_layer,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=args.ins_norm
    )).cuda()
    evaluate(model, cfg, False, args.ckpt_path, logger, tta=args.tta, test=args.test)
