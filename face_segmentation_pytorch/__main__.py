#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains functionality to run the face segmentation model on a
directory with images, and save the corresponding output heatmaps/masks.
"""


import os
# for OmegaConf
from dataclasses import dataclass
from typing import Optional
#
from omegaconf import OmegaConf, MISSING
import torch
from PIL import Image
#
from .model import FaceSegmentationNet
from .utils import load_model_parameters


# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar MODEL_PARAMS_DIR: Path to a directory containing the model params.
      If it doesn't contain them, parameters will be downloaded there.
    :cvar BATCH_SIZE: If the images in ``IMGS_DIR`` all have same shape and
      format, they can be processed in batches of more than 1.
    :cvar MASK_THRESHOLD: By default, heatmaps will be created as ``uint8``
      images. If a threshold between 0 and 1 is given, the heatmaps will be
      binarized.
    """
    IMGS_DIR: str = MISSING
    MODEL_PARAMS_DIR: str = MISSING
    #
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 1
    #
    MASK_THRESHOLD: Optional[float] = None


# #############################################################################
# # MAIN ROUTINE
# #############################################################################
if __name__ == "__main__":
    # figure out conf
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    # create model and load pretrained parameters
    model = FaceSegmentationNet()
    load_model_parameters(model, CONF.MODEL_PARAMS_DIR)
    model.eval()
    model = model.to(CONF.DEVICE)

    # process images:
    img_paths = [os.path.join(CONF.IMGS_DIR, p)
                 for p in os.listdir(CONF.IMGS_DIR)]
    out_suffix = "heatmap" if CONF.MASK_THRESHOLD is None else "mask"
    # batch iter
    for idx in range(0, len(img_paths), CONF.BATCH_SIZE):
        ips = img_paths[idx:idx+CONF.BATCH_SIZE]
        imgs = [Image.open(ip).convert("RGB") for ip in ips]
        with torch.no_grad():
            tensors = torch.stack([model.img_to_input(img, pre_normalize=True)
                                   for img in imgs]).to(CONF.DEVICE)
            outputs = model(tensors, as_pmap=True)
            # from . import plt_debug
            # fig, _ = plt_debug(imgs, outputs.cpu().numpy(), color_alpha=0.5)
            # fig.tight_layout()
            # fig.show()
            if CONF.MASK_THRESHOLD is not None:
                outputs = (outputs >= CONF.MASK_THRESHOLD)
            outputs = outputs.mul(255).type(torch.uint8).cpu().numpy()
            # save each img in batch
            for ip, out in zip(ips, outputs):
                out_path = f"{ip}__{out_suffix}.png"
                Image.fromarray(out).save(out_path)
                print("Saved", out_suffix, "to", out_path)
                # import matplotlib.pyplot as plt
                # plt.clf(); plt.imshow(out); plt.show()
