import os
from collections import OrderedDict

import torch
from torch.nn.modules import Sequential

from .backbone import build_backbone
from .heads import build_head


def build_model(cfg):
    backbone = build_backbone(cfg)
    head = build_head(cfg)

    model = Sequential(OrderedDict([("backbone", backbone), ("head", head)]))

    if cfg.MODEL.PRETRAIN == "imagenet":
        pretrained_path = os.path.expanduser(
            cfg.MODEL.PRETRIANED_PATH[cfg.MODEL.BACKBONE.NAME]
        )
        print(f">>> Loading imagenet pretrianed model from {pretrained_path}...")
        model.backbone.load_param(pretrained_path)
    elif os.path.exists(cfg.MODEL.PRETRAIN):
        print(f">>> Loading cfg model from {cfg.MODEL.PRETRAIN}...")
        ckp = torch.load(cfg.MODEL.PRETRAIN)
        model.load_state_dict(ckp["model"])
    elif cfg.MODEL.PRETRAIN == "scratch":
        print(">>> train from scratch")
    else:
        assert (
            False
        ), f"not imagenet pretrain and not exist model path: {cfg.MODEL.PRETRAIN}"
    return model
