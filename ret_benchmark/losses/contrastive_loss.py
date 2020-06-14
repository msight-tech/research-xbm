from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from ret_benchmark.losses.registry import LOSS
from ret_benchmark.utils.log_info import log_info


@LOSS.register("contrastive_loss")
class ContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.5

    def forward(self, inputs_col, targets_col, inputs_row, target_row):

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)
        if inputs_col.shape[0] == inputs_row.shape[0]:
            prefix = "batch_"
        else:
            prefix = "memory_"
        if len(neg_count) != 0:
            log_info[f"{prefix}average_neg"] = sum(neg_count) / len(neg_count)
        else:
            log_info[f"{prefix}average_neg"] = 0
        log_info[f"{prefix}non_zero"] = len(neg_count)
        loss = sum(loss) / n  # / all_targets.shape[1]
        return loss
