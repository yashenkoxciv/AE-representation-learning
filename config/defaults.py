from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.MODEL = CfgNode()
_C.MODEL.DEVICE = 'cuda'
# AE is unsupervised so we do not need number of classes
#_C.MODEL.NUM_CLASSES = 10

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 224
# Size of the image during test
# use bigger images (256) with center crop
_C.INPUT.SIZE_TEST = 224 #256
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# RECONSTRUCTION
# -----------------------------------------------------------------------------
_C.RECONSTRUCTION = CfgNode()
# Size of the image during training
_C.RECONSTRUCTION.SIZE_TRAIN = 64
# Size of the image during test
# use bigger images (256) with center crop
_C.RECONSTRUCTION.SIZE_TEST = 64 #256
# Minimum scale for the image during training
_C.RECONSTRUCTION.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.RECONSTRUCTION.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.RECONSTRUCTION.PROB = 0.5
# Values to be used for image normalization
_C.RECONSTRUCTION.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.RECONSTRUCTION.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CfgNode()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN_ROOT = ''
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST_ROOT = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CfgNode()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.OPTIMIZER_NAME = 'SGD'

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = 'linear'

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CfgNode()
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.WEIGHT = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_ROOT = ''
_C.PROJECT_NAME = 'AE-REL'
_C.EXPERIMENT_NAME = ''
