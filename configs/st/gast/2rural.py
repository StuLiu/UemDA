from configs.ToRURAL import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET, TEST_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, target_dir, DATASETS
import module.aug.augmentation as mag


MODEL = 'ResNet'

IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SNAPSHOT_DIR = './log/GAST/2rural'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 7500  # for learning rate poly
NUM_STEPS_STOP = 5000  # Use damping instead of early stopping
FIRST_STAGE_STEP = 2000  # for first stage
PREHEAT_STEPS = int(NUM_STEPS / 20)  # for warm-up
POWER = 0.9  # lr poly power
EVAL_FROM = int(NUM_STEPS_STOP * 0.6) - 1
EVAL_EVERY = 500
GENERATE_PSEDO_EVERY = 500
MULTI_LAYER = True
IGNORE_BG = True
PSEUDO_SELECT = True

TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
# TARGET_DATA_CONFIG=TARGET_DATA_CONFIG
PSEUDO_DATA_CONFIG = PSEUDO_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG
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
            mean=(73.53223948, 80.01710095, 74.59297778),
            std=(41.5113661, 35.66528876, 33.75830885)
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    label_type='prob',
)
