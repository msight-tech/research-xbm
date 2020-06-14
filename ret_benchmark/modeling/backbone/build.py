from ret_benchmark.modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet import ResNet18, ResNet50
from .googlenet import GoogLeNet


def build_backbone(cfg):
    assert (
        cfg.MODEL.BACKBONE.NAME in BACKBONES
    ), f"backbone {cfg.MODEL.BACKBONE} is not defined"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME](
        last_stride=cfg.MODEL.BACKBONE.LAST_STRIDE
    )
