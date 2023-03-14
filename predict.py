import os
import warnings

from data.loveda import LoveDALoader
from utils.tools import *
from skimage.io import imsave
from module.Encoder import Deeplabv2
from argparse import ArgumentParser


def predict_test(model, cfg, ckpt_path=None, save_dir='./submit_test'):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)

    count_model_parameters(model, get_console_file_logger(name='CBST', logdir='log/infer'))
    model.eval()
    print(cfg.TEST_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                imsave(os.path.join(save_dir, fname), pred.astype(np.uint8))

    torch.cuda.empty_cache()

# python predict.py st.cbst.2urban log/cbst/2urban/URBAN10000.pth submit_test/cbst/2urban
# python predict.py st.cbst.2rural log/cbst/2rural/RURAL10000.pth submit_test/cbst/2rural
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('config_path', type=str, help='config path')
    parser.add_argument('ckpt_path', type=str, help='ckpt path')
    parser.add_argument('save_dir', type=str, help='save dir path')
    args = parser.parse_args()

    cfg = import_config(args.config_path)
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=cfg.NUM_CLASSES,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=cfg.NUM_CLASSES
    )).cuda()
    predict_test(model, cfg, args.ckpt_path, save_dir=args.save_dir)
