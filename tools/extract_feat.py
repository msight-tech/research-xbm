# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.data.collate_batch import collate_fn
from ret_benchmark.data.datasets import BaseDataSet
from ret_benchmark.data.transforms import build_transforms
from ret_benchmark.modeling import build_model
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.logger import setup_logger


def extract(cfg, img_source, model_path=None):
    logger = setup_logger(name="Feat", level=cfg.LOGGER.LEVEL)

    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if model_path is not None:
        print(f"load model {model_path} .......")
        model_dict = torch.load(model_path)["model"]
        model.load_state_dict(model_dict, strict=True)
        print("model successfully loaded")

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    transforms = build_transforms(cfg, is_train=False)

    dataset = BaseDataSet(img_source, transforms=transforms, mode=cfg.INPUT.MODE)
    data_loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=cfg.DATA.TEST_BATCHSIZE * num_gpus,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=False,
    )

    labels = dataset.label_list
    feats = feat_extractor(model, data_loader, logger)

    day_time = time.strftime("%Y-%m-%d", time.localtime())
    npz_path = f"output/{day_time}_feat.npz"
    np.savez(npz_path, feat=feats, upc=labels)
    print(f"FEATS : \t {npz_path}")


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Extract feature")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="config file", default=None, type=str
    )
    parser.add_argument(
        "--img", dest="img_source", help="img csv file", default=None, type=str
    )
    parser.add_argument(
        "--model", dest="model_path", help="model path file", default=None, type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    extract(cfg, img_source=args.img_source, model_path=args.model_path)
