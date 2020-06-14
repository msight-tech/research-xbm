import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        list(model.parameters()),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=4,
        threshold=0.001,
        cooldown=2,
        min_lr=cfg.SOLVER.BASE_LR / (10 * cfg.SOLVER.STEPS),
    )
