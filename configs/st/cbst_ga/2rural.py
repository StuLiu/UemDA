from configs.ToRURAL import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET, \
    TEST_DATA_CONFIG
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 1000

SNAPSHOT_DIR = './log/cbst_ga/2rural'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 15000
NUM_STEPS_STOP = 10000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
EVAL_EVERY=2000


DS_RATE = 4
KC_VALUE = 'conf'
KC_POLICY = 'cb'
MINE_PORT = 1e-3
RARE_CLS_NUM = 3
RM_PROB = True
WARMUP_STEP = 4000
GENERATE_PSEDO_EVERY=1000
TGT_PORTION = 1e-1
TGT_PORTION_STEP = 0.
MAX_TGT_PORTION = 1e-1
SOURCE_LOSS_WEIGHT = 1.0
PSEUDO_LOSS_WEIGHT = 0.5


TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG=TARGET_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG
TEST_DATA_CONFIG=TEST_DATA_CONFIG