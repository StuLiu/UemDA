"""
@Project :
@File    : cutmix.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/25 15:07
@e-mail  : liuwa@hnu.edu.cn
"""
import numpy as np
import random
import cv2
import torch


def classmix(data, targets, class_num=7):
    data, targets = data.clone(), targets.clone()
    indices = torch.randperm(data.size(0))  # rand batch-wise
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    class_id = (class_num)  # rand batch-wise


    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets[:, y0:y1, x0:x1] = shuffled_targets[:, y0:y1, x0:x1]
    return data, targets


if __name__ == '__main__':
    image1 = cv2.imread('1.png')
    image1 = torch.Tensor(image1).int().permute(2, 0, 1).unsqueeze(dim=0)
    image1 = torch.cat([image1, image1], dim=0)

    image2 = cv2.imread('2.png')
    image2 = torch.Tensor(image2).int().permute(2, 0, 1).unsqueeze(dim=0)
    image2 = torch.cat([image2, image2], dim=0)

    label1 = torch.zeros_like(image1[:, 0, :, :])
    label2 = torch.ones_like(image2[:, 0, :, :])
    k = cv2.waitKey(1)

    while k != ord('q'):
        imgs, lbls = cutmix(torch.cat([image1, image2], dim=0), torch.cat([label1, label2], dim=0), alpha=1)
        imgs = imgs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        lbls = lbls.numpy().astype(np.uint8)
        cv2.imshow('i1', imgs[0, :, :, :])
        cv2.imshow('i2', imgs[1, :, :, :])
        cv2.imshow('i3', imgs[2, :, :, :])
        cv2.imshow('i4', imgs[3, :, :, :])
        cv2.imshow('l1', lbls[0, :, :] * 255)
        cv2.imshow('l2', lbls[1, :, :] * 255)
        cv2.imshow('l3', lbls[2, :, :] * 255)
        cv2.imshow('l4', lbls[3, :, :] * 255)

        k = cv2.waitKey(0)
