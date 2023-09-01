from configs.ToPotsdam import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS
import module.aug.augmentation as mag


MODEL = 'ResNet50'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SNAPSHOT_DIR = './log/GAST/2potsdam'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 7500  # for learning rate poly
NUM_STEPS_STOP = 5000  # Use damping instead of early stopping
FIRST_STAGE_STEP = 5000  # for first stage
PREHEAT_STEPS = int(NUM_STEPS / 20)  # for warm-up
POWER = 0.9  # lr poly power
EVAL_EVERY = 500
EVAL_FROM = 0#int(NUM_STEPS_STOP * 0.6) - 1
GENERATE_PSEDO_EVERY = 500
MULTI_LAYER = False
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
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    label_type='prob',
)
