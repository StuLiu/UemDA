from configs.ToVaihingen import *
import module.aug.augmentation as mag


MODEL = 'ResNet101'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SNAPSHOT_DIR = './log/GAST/2vaihingen'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 15000  # for learning rate poly
NUM_STEPS_STOP = 10000  # Use damping instead of early stopping
FIRST_STAGE_STEP = 10000  # for first stage
PREHEAT_STEPS = int(NUM_STEPS / 20)  # for warm-up
POWER = 0.9  # lr poly power
EVAL_FROM = 0#int(NUM_STEPS_STOP * 0.6) - 1
EVAL_EVERY = 1000
GENERATE_PSEDO_EVERY = 1000
MULTI_LAYER = True
IGNORE_BG = True
PSEUDO_SELECT = True
CUTOFF_TOP = 0.8
CUTOFF_LOW = 0.6

TARGET_SET = TARGET_SET
# SOURCE_DATA_CONFIG = SOURCE_DATA_CONFIG
PSEUDO_DATA_CONFIG = PSEUDO_DATA_CONFIG
EVAL_DATA_CONFIG = EVAL_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG

source_dir = dict(
    image_dir=[
        'data/IsprsDA/Vaihingen/img_dir/train',
    ],
    mask_dir=[
        'data/IsprsDA/Vaihingen/ann_dir/train',
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
        Normalize(
            mean=(120.8217, 81.8250, 81.2344),
            std=(54.7461, 39.3116, 37.9288),
        ),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
)
