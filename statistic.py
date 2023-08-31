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
import torch

def ema_static(history, curr, gama=0.99):
    return history * gama + curr * (1 - gama)


# 数据集路径
dataset_paths = [
    'data/IsprsDA/Potsdam/img_dir',
    'data/IsprsDA/Vaihingen/img_dir',
]

# dataset_paths = [
#     'data/LoveDA/Train/Urban/images_png',
#     'data/LoveDA/Val/Urban/images_png',
#     'data/LoveDA/Test/Urban/images_png',
#     'data/LoveDA/Train/Rural/images_png',
#     'data/LoveDA/Val/Rural/images_png',
#     'data/LoveDA/Test/Rural/images_png',
# ]

# 初始化均值和方差
mean_values = torch.zeros((3,))
std_values = torch.zeros((3,))
cnt = 0
image = None
# 遍历数据集中的图像
for dataset_path in dataset_paths:
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            # 构建图像文件的完整路径
            image_path = os.path.join(root, file)

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # cv2.imshow("", image)
            # cv2.waitKey(0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为浮点数
            image = image.astype(np.float32)
            image = torch.FloatTensor(image)
            # 计算均值和方差
            # mean = np.zeros((image.shape[-1],))
            # variance = np.zeros((image.shape[-1],))
            for i in range(image.shape[-1]):
                mean_values[i] = mean_values[i] + torch.sum(image[:, :, i])
                # std_values[i] = std_values[i] + np.std(image[:, :, i])

            # update statistics
            # mean_values = ema_static(mean_values, mean)
            # variance_values = ema_static(variance_values, variance)
            cnt += 1
    # 计算整个数据集的均值和方差
    print("Dataset Mean:", mean_values / cnt / image.shape[0] / image.shape[1])
    # print("Dataset Variance:", std_values / cnt)

mean_values = mean_values / cnt / image.shape[0] / image.shape[1]
cnt = 0
for dataset_path in dataset_paths:
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            # 构建图像文件的完整路径
            image_path = os.path.join(root, file)

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # cv2.imshow("", image)
            # cv2.waitKey(0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为浮点数
            image = image.astype(np.float32)
            image = torch.FloatTensor(image)
            # 计算均值和方差
            # mean = np.zeros((image.shape[-1],))
            # variance = np.zeros((image.shape[-1],))
            for i in range(image.shape[-1]):
                std_values[i] = std_values[i] + torch.sum((image[:, :, i] - mean_values[i]) ** 2)
            cnt += 1
    # 计算整个数据集的均值和方差
    print("Dataset Mean:", torch.sqrt(std_values / cnt / image.shape[0] / image.shape[1]))
    # print("Dataset Variance:", std_values / cnt)

