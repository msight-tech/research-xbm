# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from ret_benchmark.utils.str2int import str2int


def collate_fn(batch):
    imgs, labels, indices = zip(*batch)
    labels = [str2int(k) for k in labels]
    labels = torch.tensor(labels, dtype=torch.int64)
    indices = torch.tensor(indices, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels, indices
