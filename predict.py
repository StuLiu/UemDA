import os
import warnings

from data.loveda import LoveDALoader
from utils.tools import *
from skimage.io import imsave
from module.Encoder import Deeplabv2
from argparse import ArgumentParser


def predict_test(model, cfg, ckpt_path=None, save_dir='./submit_test', slide=True):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)

    os.makedirs('log/infer', exist_ok=True)
    count_model_parameters(model, get_console_file_logger(name='CBST', logdir='log/infer'))
    model.eval()
    print(cfg.TEST_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = pre_slide(model, ret, tta=True) if slide else model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                imsave(os.path.join(save_dir, fname), pred.astype(np.uint8))

    torch.cuda.empty_cache()

# python predict.py st.cbst.2urban log/cbst/2urban/URBAN10000.pth submit_test/cbst/2urban
# python predict.py st.cbst.2rural log/cbst/2rural/RURAL10000.pth submit_test/cbst/2rural
# CUDA_VISIBLE_DEVICES=0 python predict.py st.dca.2rural log/DCA/2rural20230322093612_34.47_4000/RURAL4000.pth submit_test/dca_baseline/2rural
# CUDA_VISIBLE_DEVICES=1 python predict.py st.dca.2urban log/DCA/2urban20230321233258_53.55_8000/URBAN8000.pth submit_test/dca_baseline/2urban
# CUDA_VISIBLE_DEVICES=2 python predict.py st.gast.2rural log/GAST/2rural_d_36.14_3000/RURAL3000.pth submit_test/gast_d/2rural
# CUDA_VISIBLE_DEVICES=3 python predict.py st.gast.2urban log/GAST/2urban_d_51.70_8000/URBAN8000.pth submit_test/gast_d/2urban
# CUDA_VISIBLE_DEVICES=4 python predict.py st.gast.2rural log/GAST/2rural_d_c_34.59_3000/RURAL3000.pth submit_test/gast_d_c/2rural
# CUDA_VISIBLE_DEVICES=5 python predict.py st.gast.2urban log/GAST/2urban_d_c_52.09_8000/URBAN8000.pth submit_test/gast_d_c/2urban
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('config_path', type=str, help='config path')
    parser.add_argument('ckpt_path', type=str, help='ckpt path')
    parser.add_argument('save_dir', type=str, help='save dir path')
    parser.add_argument('--ins-norm', type=str, help='save dir path')
    args = parser.parse_args()

    cfg = import_config(args.config_path, copy=False)
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
            num_classes=cfg.NUM_CLASSES,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=cfg.NUM_CLASSES,
        is_ins_norm=True,
    )).cuda()
    predict_test(model, cfg, args.ckpt_path, save_dir=args.save_dir)
