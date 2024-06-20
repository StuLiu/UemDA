from configs.ToPotsdam import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS
import uemda.aug.augmentation as mag


MODEL = 'ResNet101'

IGNORE_LABEL = -1
MOMENTUM = 0.9

SNAPSHOT_DIR = './log/uemda/2potsdam'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
STAGE1_STEPS = 4000
STAGE2_STEPS = 6000
STAGE3_STEPS = 6000
FIRST_STAGE_STEP = 2000
NUM_STEPS_STOP = 5000
NUM_STEPS = 7500        # for learning rate poly
PREHEAT_STEPS = 500    # for warm-up
POWER = 0.9                 # lr poly power
EVAL_FROM = 0
EVAL_EVERY = 500
GENERATE_PSEDO_EVERY = 500
CUTOFF_TOP = 0.8
CUTOFF_LOW = 0.6

TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=[None],
    transforms=mag.Compose([
        mag.RandomCrop((512, 512)),
        mag.RandomHorizontalFlip(0.5),
        mag.RandomVerticalFlip(0.5),
        mag.RandomRotate90(0.5),
        mag.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            clamp=True,
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    label_type='prob',
    read_sup=True,
)
