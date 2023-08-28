from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import *
import ever as er

DATASETS = 'LoveDA'
TARGET_SET = 'RURAL'

source_dir = dict(
    image_dir=[
        'data/LoveDA/Train/Urban/images_png/',
    ],
    mask_dir=[
        'data/LoveDA/Train/Urban/masks_png/',
    ],
)
target_dir = dict(
    image_dir=[
        'data/LoveDA/Val/Rural/images_png/',
    ],
    mask_dir=[
        'data/LoveDA/Val/Rural/masks_png/',
    ],
)
val_dir = dict(
    image_dir=[
        'data/LoveDA/Train/Rural/images_png/',
    ],
    mask_dir=[
        'data/LoveDA/Train/Rural/masks_png/',
    ],
)
test_dir = dict(
    image_dir=[
        'data/LoveDA/Test/Rural/images_png/'
    ],
    mask_dir=[
        None,
    ]
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
        Normalize(mean=(86.33139106, 86.42503241, 86.48510096),
                  std=(28.31421307, 28.35913085, 28.41702921),
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
        Normalize(mean=(92.36257614, 92.42804661, 92.44088941),
                  std=(44.58176958, 44.5633478, 44.68216329),
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
        Normalize(mean=(92.36257614, 92.42804661, 92.44088941),
                  std=(44.58176958, 44.5633478, 44.68216329),
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
        Normalize(mean=(92.36257614, 92.42804661, 92.44088941),
                  std=(44.58176958, 44.5633478, 44.68216329),
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
        Normalize(mean=(92.36257614, 92.42804661, 92.44088941),
                  std=(44.58176958, 44.5633478, 44.68216329),
                  max_pixel_value=1, always_apply=True),
        # Normalize(mean=(123.675, 116.28, 103.53),
        #           std=(58.395, 57.12, 57.375),
        #           max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=1,
)
