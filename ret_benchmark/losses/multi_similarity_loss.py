# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.: xunwang@malong.com

import torch
from torch import nn
from ret_benchmark.utils.log_info import log_info

from ret_benchmark.losses.registry import LOSS


@LOSS.register("ms_loss")
class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.hard_mining = cfg.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING

    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        batch_size = inputs_col.size(0)
        sim_mat = torch.matmul(inputs_col, inputs_row.t())

        epsilon = 1e-5
        loss = list()
        neg_count = 0
        for i in range(batch_size):
            pos_pair_ = torch.masked_select(sim_mat[i], target_row == targets_col[i])
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], target_row != targets_col[i])

            # sampling step
            if self.hard_mining:
                neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            neg_count += len(neg_pair)

            # weighting step
            pos_loss = (
                1.0
                / self.scale_pos
                * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
                )
            )
            neg_loss = (
                1.0
                / self.scale_neg
                * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
                )
            )
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).cuda()
        log_info["neg_count"] = neg_count / batch_size
        loss = sum(loss) / batch_size
        return loss
