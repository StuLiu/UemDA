from configs.ToVaihingen import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS
import uemda.aug.augmentation as mag


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
SOURCE_DATA_CONFIG = SOURCE_DATA_CONFIG
PSEUDO_DATA_CONFIG = PSEUDO_DATA_CONFIG
EVAL_DATA_CONFIG = EVAL_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=[None],
    transforms=mag.Compose([
        mag.RandomCrop((512, 512)),
        mag.RandomHorizontalFlip(0.5),
        mag.RandomVerticalFlip(0.5),
        mag.RandomRotate90(0.5),
        mag.Normalize(
            mean=(120.8217, 81.8250, 81.2344),
            std=(54.7461, 39.3116, 37.9288),
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    label_type='prob',
)
