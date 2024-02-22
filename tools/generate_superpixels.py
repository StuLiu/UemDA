"""
@Project : rads2
@File    : generate_superpixels.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/10/7 下午5:22
@e-mail  : 1183862787@qq.com
"""
from uemda.gast.superpixels import get_superpixels


if __name__ == '__main__':
    # sp = SuperPixelsLSC()
    # img_cv2_1 = cv2.imread("../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_0_512_512.png")
    # img_cv2_2 = cv2.imread("../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_512_512_1024.png")

    get_superpixels(dir_path="../data/IsprsDA/Vaihingen/img_dir/train",
                    out_dir="../data/IsprsDA/Vaihingen/ann_dir/train_sup",
                    postfix='png', show=False)
    get_superpixels(dir_path="../data/IsprsDA/Potsdam/img_dir/train",
                    out_dir="../data/IsprsDA/Potsdam/ann_dir/train_sup",
                    postfix='png', show=False)
