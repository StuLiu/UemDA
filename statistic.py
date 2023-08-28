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
dataset_paths = [
    'data/IsprsDA/Potsdam/img_dir',
    'data/IsprsDA/Vaihingen/img_dir',
]

# 初始化均值和方差
mean_values = 0
variance_values = 0

# 遍历数据集中的图像
for dataset_path in dataset_paths:
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            # 构建图像文件的完整路径
            image_path = os.path.join(root, file)

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 转换为浮点数
            image = image.astype(np.float32)

            # 计算均值和方差
            mean = np.mean(image)
            variance = np.var(image)

            # update statistics
            mean_values = ema_static(mean_values, mean)
            variance_values = ema_static(variance_values, variance)

# 计算整个数据集的均值和方差
print("Dataset Mean:", dataset_mean)
print("Dataset Variance:", dataset_variance)
