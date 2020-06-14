# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN
from .model_path import MODEL_PATH

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.NAME = "default"

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet50"
_C.MODEL.BACKBONE.LAST_STRIDE = 2

_C.MODEL.PRETRAIN = "imagenet"  #'imagenet'
_C.MODEL.PRETRIANED_PATH = MODEL_PATH

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "linear_norm"
_C.MODEL.HEAD.IN_CHANNELS = 1024
_C.MODEL.HEAD.DIM = 512
_C.MODEL.HEAD.NUM_CLASSES = 1000

_C.MODEL.WEIGHT = ""

_C.SAVE = True
# Checkpoint save dir
_C.SAVE_DIR = "output"

# tensorboard save dir
_C.TB_SAVE_DIR = "runs"

# Loss
_C.LOSSES = CN()
_C.LOSSES.NAME = "ms_loss"
# ms loss
_C.LOSSES.MULTI_SIMILARITY_LOSS = CN()
_C.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS = 2.0
_C.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG = 40.0
_C.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING = True


_C.XBM = CN()

_C.XBM.ENABLE = True
_C.XBM.WEIGHT = 1.0
_C.XBM.SIZE = 8192
_C.XBM.START_ITERATION = 2000

# Data option
_C.DATA = CN()
_C.DATA.TRAIN_IMG_SOURCE = ""
_C.DATA.TEST_IMG_SOURCE = ""
_C.DATA.QUERY_IMG_SOURCE = ""
_C.DATA.PKUVID_IMG_SOURCE = ""
_C.DATA.TRAIN_BATCHSIZE = 70
_C.DATA.TEST_BATCHSIZE = 256
_C.DATA.NUM_WORKERS = 8
_C.DATA.NUM_INSTANCES = 5
_C.DATA.SAMPLE = "RandomIdentitySampler"

# Input option
_C.INPUT = CN()

# follow standard input transforms of imagenet
_C.INPUT.MODE = "RGB"
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.ORIGIN_SIZE = [256, 256]
_C.INPUT.CROP_SCALE = [0.2, 1]
_C.INPUT.CROP_SIZE = [224, 224]

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.FIX_BN = False
_C.SOLVER.IS_FINETURN = False
_C.SOLVER.FINETURN_MODE_PATH = ""
_C.SOLVER.MAX_ITERS = 4000
_C.SOLVER.STEPS = 2
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.CHECKPOINT_PERIOD = 1000
_C.SOLVER.RNG_SEED = 1

# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = 20
_C.LOGGER.STREAM = "stdout"

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.R = [1]
_C.VALIDATION.VERBOSE = 200
_C.VALIDATION.IS_VALIDATION = True
