from ret_benchmark.modeling.registry import HEADS

from .linear_norm import LinearNorm


def build_head(cfg):
    assert cfg.MODEL.HEAD.NAME in HEADS, f"head {cfg.MODEL.HEAD.NAME} is not defined"
    return HEADS[cfg.MODEL.HEAD.NAME](cfg)
