"""
@Project :
@File    :
@IDE     : PyCharm
@Author  : Wang Liu
@Date    :
@e-mail  : 1183862787@qq.com
"""

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler, RandomSampler
from ever.api.data import CrossValSamplerGenerator
import numpy as np
import logging


logger = logging.getLogger(__name__)


class BaseData(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, label_type='id', offset=-1):
        assert label_type in ['id', 'prob']
        self.label_type = label_type
        self.n_classes = 7
        self.ignore_label = -1
        self.offset = offset
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if self.label_type == 'prob':
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        if len(self.cls_filepath_list) > 0:
            if self.label_type == 'id':
                # 0~7 --> -1~6, 0 in mask.png represents the black area in the input.png
                mask = imread(self.cls_filepath_list[idx]).astype(np.long) + self.offset
            else:
                # mask = torch.from_numpy(np.load(f'{self.cls_filepath_list[idx]}.npy')).float()
                mask = torch.load(f'{self.cls_filepath_list[idx]}.pt', map_location=torch.device('cpu'))
            # avoid noise label
            mask[mask >= self.n_classes] = self.ignore_label
            # data augmentation
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)
