'''
@Project : Unsupervised_Domian_Adaptation 
@File    : vis_pseudo_labels.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/11/16 下午2:01
@e-mail  : 1183862787@qq.com
'''

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm

# 定义五个目录的路径
directory1 = "../data/IsprsDA/Potsdam/img_dir/train"
directory2 = "../data/IsprsDA/Potsdam/ann_dir/train_color"
directory3 = "../log/GAST/2potsdam_vis_pseudo_Base/pseudo_label_color_3000"
directory4 = "../log/GAST/2potsdam_vis_pseudo_ProDA/pseudo_label_color_3500"
directory5 = "../log/GAST/2potsdam_vis_pseudo_MVC/pseudo_label_color"

# 读取图片文件名列表
file_names = os.listdir(directory1)

class Ele:

    def __init__(self, name):
        p_gt = cv2.imread(os.path.join(directory2, name))
        p_base = cv2.imread(os.path.join(directory3, name))
        # p_proda = cv2.imread(os.path.join(directory4, name))
        p_mvc = cv2.imread(os.path.join(directory5, name))

        # pixels_num = 1.0 * p_gt.shape[0] * p_gt.shape[1]
        pixels_num = 1.0

        p_gt = p_gt[:,:,0] * 256 * 256 + p_gt[:,:,1] * 256 + p_gt[:,:,2]
        p_base = p_base[:,:,0] * 256 * 256 + p_base[:,:,1] * 256 + p_base[:,:,2]
        # p_proda = p_proda[:,:,0] * 256 * 256 + p_proda[:,:,1] * 256 + p_proda[:,:,2]
        p_mvc = p_mvc[:,:,0] * 256 * 256 + p_mvc[:,:,1] * 256 + p_mvc[:,:,2]

        r_base = np.sum(((p_gt == 0) + (p_gt == p_base) != 0)) * 1.0 / pixels_num
        # r_proda = np.sum(p_gt == p_proda) * 1.0 / pixels_num
        r_mvc = np.sum(((p_gt == 0) + (p_gt == p_mvc) != 0)) / pixels_num

        self.delta = r_mvc - r_base
        self.name = name

eles = []
for f_name in tqdm(file_names):
    eles.append(Ele(f_name))
print('>>>> sorting...')
eles = sorted(eles, key=lambda _ele : _ele.delta, reverse=True)
print('<<<< sorted.')
# file_names = [_ele.name for _ele in eles]


# 创建一个包含五个子图的画布
fig, axs = plt.subplots(nrows=1, ncols=5)

# 设置初始索引
index = 0

def show_next_group(event):
    global index

    # 清空之前的子图内容
    for ax in axs.flat:
        ax.clear()
    file_name = eles[index].name
    print(file_name, eles[index].delta)
    # 从五个目录读取图片
    image1_path = os.path.join(directory1, file_name)
    image2_path = os.path.join(directory2, file_name)
    image3_path = os.path.join(directory3, file_name)
    image4_path = os.path.join(directory4, file_name)
    image5_path = os.path.join(directory5, file_name)

    # 读取并显示图片
    image1 = plt.imread(image1_path)
    axs[0].imshow(image1)
    axs[0].axis('off')

    image2 = plt.imread(image2_path)
    axs[1].imshow(image2)
    axs[1].axis('off')

    image3 = plt.imread(image3_path)
    axs[2].imshow(image3)
    axs[2].axis('off')

    image4 = plt.imread(image4_path)
    axs[3].imshow(image4)
    axs[3].axis('off')

    image5 = plt.imread(image5_path)
    axs[4].imshow(image5)
    axs[4].axis('off')

    index = (index + 1) % len(file_names)

    # 更新画布
    plt.draw()


# 注册按键事件
fig.canvas.mpl_connect('key_press_event', lambda event: event.key == 'enter' and show_next_group(event))

# 显示图像
plt.show()