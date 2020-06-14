import torch
from torch import nn

from ret_benchmark.modeling.registry import HEADS
from ret_benchmark.utils.init_methods import weights_init_kaiming


@HEADS.register("linear_norm")
class LinearNorm(nn.Module):
    def __init__(self, cfg):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(cfg.MODEL.HEAD.IN_CHANNELS, cfg.MODEL.HEAD.DIM)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
