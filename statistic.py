"""
@Project : Unsupervised_Domian_Adaptation
@File    : statistic.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/8/28 下午6:17
@e-mail  : 1183862787@qq.com
"""
import numpy as np
import cv2
import os
from tqdm import tqdm


def ema_static(history, curr, gama=0.99):
    return history * gama + curr * (1 - gama)


# 数据集路径
# dataset_paths = [
#     'data/IsprsDA/Potsdam/img_dir',
#     'data/IsprsDA/Vaihingen/img_dir',
# ]

dataset_paths = [
    'data/LoveDA/Train/Urban/images_png',
    'data/LoveDA/Train/Rural/images_png',
]

# 初始化均值和方差
mean_values = np.zeros((3,))
variance_values = np.zeros((3,))

# 遍历数据集中的图像
for dataset_path in dataset_paths:
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            # 构建图像文件的完整路径
            image_path = os.path.join(root, file)

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # 转换为浮点数
            image = image.astype(np.float32)

            # 计算均值和方差
            mean = np.zeros((image.shape[-1],))
            variance = np.zeros((image.shape[-1],))
            for i in range(image.shape[-1]):
                mean[i] = np.mean(image[i, :, :])
                variance[i] = np.std(image[i, :, :])

            # update statistics
            mean_values = ema_static(mean_values, mean)
            variance_values = ema_static(variance_values, variance)

    # 计算整个数据集的均值和方差
    print("Dataset Mean:", mean_values)
    print("Dataset Variance:", variance_values)
