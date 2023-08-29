from configs.ToPotsdam import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET, \
    TEST_DATA_CONFIG, DATASETS, PSEUDO_DATA_CONFIG
MODEL = 'ResNet'


IGNORE_LABEL = -1
IGNORE_LABEL_INFILE=-1
MOMENTUM = 0.9
NUM_CLASSES = 6

SAVE_PRED_EVERY = 500

SNAPSHOT_DIR = './log/cbst/2potsdam'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 7500
NUM_STEPS_STOP = 5000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
EVAL_EVERY=500


DS_RATE = 4
KC_VALUE = 'conf'
KC_POLICY = 'cb'
MINE_PORT = 1e-3
RARE_CLS_NUM = 1
RM_PROB = True
WARMUP_STEP = 2000
GENERATE_PSEDO_EVERY=500
TGT_PORTION = 1e-1
TGT_PORTION_STEP = 0.
MAX_TGT_PORTION = 1e-1
SOURCE_LOSS_WEIGHT = 1.0
PSEUDO_LOSS_WEIGHT = 0.5
