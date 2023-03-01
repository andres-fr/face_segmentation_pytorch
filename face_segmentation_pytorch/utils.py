#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains generic functionality:
* Input/output
* Image preprocessing
* Plotting
"""


import os
from itertools import chain
#
import numpy as np
import torch
import wget
import zipfile
from PIL import Image, ImageColor
#
from . import caffe_pb2


# ##############################################################################
# # I/O
# ##############################################################################
def load_caffemodel_as_statedict(caffemodel_path, t_type=torch.FloatTensor):
    """
    This function makes use of ``caffe_pb2``, which is the python-compiled
    version of ``caffe.proto``, to load the ``caffemodel`` into a PyTorch
    compatible state_dict, which is returned.
    """
    # Initialize NP from protobuf and load caffemodel into NP
    print("LOADING", caffemodel_path)
    np = caffe_pb2.NetParameter()
    with open(caffemodel_path, 'rb') as f:
        np.ParseFromString(f.read())
    # parse NP contents into statedict
    statedict = {}
    for lyr in chain(np.layer, np.layers):
        print("loading", lyr.name)
        for param_type, blob in zip(("weight", "bias"), lyr.blobs):
            shape = list(blob.shape.dim) if blob.shape.dim \
                else [blob.num, blob.channels, blob.height, blob.width]
            blob = t_type(blob.data).view(shape)
            statedict[f"{lyr.name}.{param_type}"] = blob
    #
    return statedict


def load_model_parameters(
        model, params_dir,
        state_dict_name="face_seg_fcn8s.pt",
        caffemodel_name="face_seg_fcn8s.caffemodel",
        statedict_online="https://github.com/andres-fr/face_segmentation_pytorch/releases/download/1.0/face_seg_fcn8s.zip"):
    """
    This function acquires the pretrained model parameters and initializes the
    ``model`` with them. Parameters are assumed to be in ``params_dir``, either
    as PyTorch statedict or Caffe caffemodel. If not, the function will try
    to download the parameters into ``params_dir``, and initialize the model.
    """
    pt_path = os.path.join(params_dir, state_dict_name)
    caffe_path = os.path.join(params_dir, caffemodel_name)
    # first try to load pytorch state_dict
    if os.path.exists(pt_path):
        statedict = torch.load(pt_path)
        model.load_state_dict(statedict, strict=True)
        print("Loaded parameters from", pt_path)
        return
    # otherwise try to load caffemodel
    elif os.path.exists(caffe_path):
        statedict = load_caffemodel_as_statedict(caffe_path)
        model.load_state_dict(statedict, strict=True)
        print("Loaded parameters from", caffe_path)
        print("Consider saving them as pytorch state_dict for faster loading")
        return
    # if none present, try to download, save in params_dir, and load
    try:
        print(f"No model parameters found in {params_dir}!")
        online_name = os.path.basename(statedict_online)
        target_path = os.path.join(params_dir, online_name)
        os.makedirs(params_dir, exist_ok=True)
        print("Downloading", online_name, "to", target_path)
        wget.download(statedict_online, target_path)
        print("Extracting statedict from zip file...")
        with zipfile.ZipFile(target_path, "r") as zip_ref:
            zip_ref.extractall(params_dir)
        os.remove(target_path)
        # now file should exist
        assert os.path.exists(pt_path), "Should never happen!"
        statedict = torch.load(pt_path)
    except Exception as e:
        print("Error encountered while downloading statedict!")
        print(e)
        print("Could not load parameters")
        return
    model.load_state_dict(statedict, strict=True)
    print("Loaded parameters from", pt_path)
    return


# ##############################################################################
# # PREPROCESSING
# ##############################################################################
def normalize_range(x, float_dtype=np.float32, out_range=(0, 255)):
    """
    :param x: Array or tensor of any shape
    :param out_range: Pair with ``(min, max)`` range of output normalized array
      (both min and max included).
    :returns: Array of same shape and dtype as input, with values stretched
      to span out_range. If ``arr`` has a constant value, the output will be
      set to the minimum ``out_range`` value.
    """
    if isinstance(x, np.ndarray):
        cast_fn = np.ndarray.astype
    elif isinstance(x, torch.Tensor):
        cast_fn = torch.Tensor.to
    else:
        raise RuntimeError("Only np.ndarray and torch.Tensor supported!")
    #
    out_min, out_max = out_range
    out_span = out_max - out_min
    assert out_min < out_max, "out_min < out_max expected!"
    #
    ori_dtype = x.dtype
    if x.min() != x.max():
        x_float = cast_fn(x, float_dtype)
        x_float -= x_float.min()
        x_float *= (out_span / x_float.max())
        x_float += out_min
        return cast_fn(x, ori_dtype)
    else:
        # if min==max, normalized version is out_min
        x = (x * 0) + out_min
        return x


# ##############################################################################
# # PLOTTING
# ##############################################################################
def plt_debug(imgs, heatmaps, mask_color="red", color_alpha=0.5,
              heatmap_thresh=0.5):
    """
    :param imgs: List of pillow images, can be of different shapes
    :param heatmaps: Collection of heatmaps with face probabilities between
      0 and 1 as output by the model. They must match ``imgs`` in quantity,
      height and width.
    :returns: ``fig, axes``, containing a top row with the images and the
      heatmaps blended on top (with given ``mask_color`` and ``color_alpha``,
      and a bottom row with the thresholded heatmaps (with ``heatmap_thresh``)
    """
    import matplotlib.pyplot as plt
    #
    rgb_color = ImageColor.getrgb(mask_color)
    colormasks = [Image.new(img.mode, img.size, rgb_color)
                  for img in imgs]
    hm_imgs = [Image.fromarray(h * 255 * color_alpha).convert("L")
               for h in heatmaps]
    fig, (mix_axes, mask_axes) = plt.subplots(nrows=2, ncols=len(imgs))
    if len(imgs) == 1:
        mix_axes, mask_axes = [mix_axes], [mask_axes]
    for img, col, hm, ax in zip(imgs, colormasks, hm_imgs, mix_axes):
        mix = Image.composite(col, img, hm)
        ax.imshow(mix)
        ax.set_xticks([])
        ax.set_yticks([])
    for hm, ax in zip(heatmaps, mask_axes):
        ax.imshow(hm >= heatmap_thresh)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, (mix_axes, mask_axes)
