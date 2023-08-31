from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import *
import ever as er

DATASETS = 'IsprsDA'
TARGET_SET = 'Vaihingen'

source_dir = dict(
    image_dir=[
        'data/IsprsDA/Potsdam/img_dir/train',
    ],
    mask_dir=[
        'data/IsprsDA/Potsdam/ann_dir/train',
    ],
)
target_dir = dict(
    image_dir=[
        'data/IsprsDA/Vaihingen/img_dir/train',
    ],
    mask_dir=[
        None,
    ],
)
val_dir = dict(
    image_dir=[
        'data/IsprsDA/Vaihingen/img_dir/val',
    ],
    mask_dir=[
        'data/IsprsDA/Vaihingen/ann_dir/val',
    ],
)
test_dir = dict(
    image_dir=[
        'data/IsprsDA/Vaihingen/img_dir/test'
    ],
    mask_dir=[
        'data/IsprsDA/Vaihingen/ann_dir/test'
    ],
)

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(100.3855, 85.8122, 91.0087),
                  std=(39.7718, 36.2300, 35.8611),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(100.3855, 85.8122, 91.0087),
                  std=(39.7718, 36.2300, 35.8611),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
)

PSEUDO_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(100.3855, 85.8122, 91.0087),
                  std=(39.7718, 36.2300, 35.8611),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=1,
)

EVAL_DATA_CONFIG = dict(
    image_dir=val_dir['image_dir'],
    mask_dir=val_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(100.3855, 85.8122, 91.0087),
                  std=(39.7718, 36.2300, 35.8611),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=1,
)

TEST_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=test_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(100.3855, 85.8122, 91.0087),
                  std=(39.7718, 36.2300, 35.8611),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=1,
)
