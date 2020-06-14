import torch
import numpy as np


def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()
    if logger is not None:
        logger.info("Begin extract")
    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()

        with torch.no_grad():
            out = model(imgs).data.cpu().numpy()
            feats.append(out)

        if logger is not None and (i + 1) % 100 == 0:
            logger.debug(f"Extract Features: [{i + 1}/{len(data_loader)}]")
        del out
    feats = np.vstack(feats)
    return feats
