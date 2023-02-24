#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains the model implementation, specifically:
* Adaptions of the ``Scale`` and ``Crop`` layers from caffe
* Class definition for the full face segmentation model
* Preprocessing helper functions (normalization)

The forward pass also includes the option of retrieving probability maps
instead of logit feature maps.
"""


import numpy as np
import torch
#
from .utils import normalize_arr


# ##############################################################################
# # LAYERS
# ##############################################################################
class AxisMultiplier(torch.nn.Module):
    """
    PyTorch adaption of the "Scale" layer in Caffe:
    https://caffe.berkeleyvision.org/tutorial/layers/scale.html
    """

    def __init__(self, num_chans=32, axis=1, init_val=1.0):
        """
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full([num_chans], init_val))
        self.axis = axis

    def forward(self, x):
        """
        """
        # reshape flat weight so it can be broadcastable with x along self.axis
        weight = self.weight
        for _ in range(len(x.shape) - 1):
            weight = weight.unsqueeze(0)
        weight = weight.swapaxes(self.axis, -1)
        # now axis multiplication is straightforward
        x = x * weight
        return x


class Crop(torch.nn.Module):
    """
    PyTorch adaption of the "Crop" layer in Caffe:
    https://caffe.berkeleyvision.org/tutorial/layers/crop.html
    """

    def __init__(self, axis=1, offset=2):
        """
        """
        super().__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, out_dims):
        """
        :param x: Tensor of shape ``(..., a1, b1, c1)``
        :param out_dims: Output dimensions ``(a2, b2, c2)``. They cannot be
          larger than the corresponding ``x`` dimensions.
        :returns: Cropped tnsor of shape ``(..., a2, b2, c2)``

        Slices ``x`` such that all dimensions upon ``self.axis`` match the
        output dimensions. If the ``self.offset`` given at construction is too
        large, it will be reduced, but it can't go below 0.
        """
        # normalize offsets into a list
        if np.isscalar(self.offset):
            offsets = [self.offset for _ in out_dims]
        else:
            assert len(self.offset) == len(out_dims)
            offsets = self.offset
        # figure out crop ranges
        assert len(x.shape) == (self.axis + len(out_dims)), \
            "Rank of input incompatible with self.axis and out_dims!"
        crop_ranges = []
        for x_dim in x.shape[:self.axis]:
            crop_ranges.append((0, x_dim))
        for x_dim, out_dim, off in zip(x.shape[self.axis:], out_dims, offsets):
            assert x_dim >= out_dim, f"Can't crop {x.shape} to {out_dims}!"
            off = min(off, (x_dim - out_dim))
            crop_ranges.append((off, off + out_dim))
        # apply crop ranges and return
        x = x[tuple(slice(a, b) for a, b in crop_ranges)]
        return x


# ##############################################################################
# # MODEL
# ##############################################################################
class FaceSegmentationNet(torch.nn.Module):
    """
    """

    MEAN_BGR = np.array([104.00699, 116.66877, 122.67892])
    DTYPE = np.float32

    @classmethod
    def img_to_input(cls, img, pre_normalize=True):
        """
        :param img: RGB pillow Image
        :param pre_normalize: If true, input image is normalized to be between
          0 and 255.
        :returns: Torch BGR tensor of ``cls.DTYPE`` and shape ``(3, h, w)``
        """
        # extract array of DTYPE and convert to BGR
        assert img.mode == "RGB", f"Only RGB supported! {img.mode}"
        arr = np.array(img)[:, :, ::-1].astype(cls.DTYPE)
        h, w, c = arr.shape
        assert c == 3, "3 Channels expected!"
        # optionally pre-normalize to 0-255
        if pre_normalize:
            arr = normalize_arr(arr, out_range=(0, 255))
        # normalize, reshape and convert to tensor
        arr -= cls.MEAN_BGR
        arr = arr.transpose((2, 0, 1))
        t = torch.from_numpy(arr)
        return t

    def __init__(self, dropout_p=0.5):
        """
        """
        super().__init__()
        # stem 1
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.relu1_1 = torch.nn.ReLU()
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = torch.nn.ReLU()
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = torch.nn.ReLU()
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = torch.nn.ReLU()
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # stem 2
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = torch.nn.ReLU()
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = torch.nn.ReLU()
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # stem 3
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = torch.nn.ReLU()
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = torch.nn.ReLU()
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = torch.nn.ReLU()
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.fc6 = torch.nn.Conv2d(512, 4096, kernel_size=7, padding=0)
        self.relu6 = torch.nn.ReLU()
        self.drop6 = torch.nn.Dropout(dropout_p, inplace=False)
        self.fc7 = torch.nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = torch.nn.ReLU()
        self.drop7 = torch.nn.Dropout(dropout_p, inplace=False)
        self.score_fr = torch.nn.Conv2d(4096, 21, kernel_size=1)
        self.upscore2 = torch.nn.ConvTranspose2d(21, 21, kernel_size=4,
                                                 stride=2, bias=False)
        # merger 3->2
        self.scale_pool4 = AxisMultiplier(512, axis=1, init_val=0.01)
        self.score_pool4 = torch.nn.Conv2d(512, 21, kernel_size=1)
        self.score_pool4c = Crop(axis=2, offset=5)
        self.upscore_pool4 = torch.nn.ConvTranspose2d(21, 21, kernel_size=4,
                                                      stride=2, bias=False)
        # merger 2->1
        self.scale_pool3 = AxisMultiplier(256, axis=1, init_val=0.0001)
        self.score_pool3 = torch.nn.Conv2d(256, 21, kernel_size=1)
        self.score_pool3c = Crop(axis=2, offset=9)
        self.upscore8 = torch.nn.ConvTranspose2d(21, 21, kernel_size=16,
                                                 stride=8, bias=False)
        # final crop
        self.score = Crop(axis=2, offset=31)

        # all upscore and scale have a lr multiplier of 0, i.e. not learned
        self.upscore2.weight.requires_grad = False
        self.scale_pool4.weight.requires_grad = False
        self.upscore_pool4.weight.requires_grad = False
        self.scale_pool3.weight.requires_grad = False
        self.upscore8.weight.requires_grad = False

    def forward(self, x_in, as_pmap=False):
        """
        :param x_in: Batch of shape ``(b, 3, h, w)``. It seems that ``(h, w)``
          should be in the ballpark of ``(500, 500)``, but can vary a bit
        :returns: Tensor of shape ``(b, 21, h, w)``, with logits. The channel
          0 corresponds to the background activations, and all other channels
          to face activations. If ``as_pmap`` is true, channels will be
          collapsed into a single ``(b, h, w)`` tensor representing the
          face probability mask.
        """
        x_in_shape = x_in.shape
        # compute stem 1
        x1 = self.conv1_1(x_in)
        del x_in
        x1 = self.relu1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.relu1_2(x1)
        x1 = self.pool1(x1)
        #
        x1 = self.conv2_1(x1)
        x1 = self.relu2_1(x1)
        x1 = self.conv2_2(x1)
        x1 = self.relu2_2(x1)
        x1 = self.pool2(x1)
        #
        x1 = self.conv3_1(x1)
        x1 = self.relu3_1(x1)
        x1 = self.conv3_2(x1)
        x1 = self.relu3_2(x1)
        x1 = self.conv3_3(x1)
        x1 = self.relu3_3(x1)
        x1 = self.pool3(x1)
        # compute stem 2
        x2 = self.conv4_1(x1)
        x2 = self.relu4_1(x2)
        x2 = self.conv4_2(x2)
        x2 = self.relu4_2(x2)
        x2 = self.conv4_3(x2)
        x2 = self.relu4_3(x2)
        x2 = self.pool4(x2)
        # compute stem 3
        x3 = self.conv5_1(x2)
        x3 = self.relu5_1(x3)
        x3 = self.conv5_2(x3)
        x3 = self.relu5_2(x3)
        x3 = self.conv5_3(x3)
        x3 = self.relu5_3(x3)
        x3 = self.pool5(x3)
        #
        x3 = self.fc6(x3)
        x3 = self.relu6(x3)
        x3 = self.drop6(x3)
        x3 = self.fc7(x3)
        x3 = self.relu7(x3)
        x3 = self.drop7(x3)
        x3 = self.score_fr(x3)
        x3 = self.upscore2(x3)
        # merge stem3 into stem2
        x2 = self.scale_pool4(x2)
        x2 = self.score_pool4(x2)
        x2 = self.score_pool4c(x2, x3.shape[2:])
        x2 = x2 + x3
        del x3
        x2 = self.upscore_pool4(x2)
        # merge stem2 into stem1
        x1 = self.scale_pool3(x1)
        x1 = self.score_pool3(x1)
        x1 = self.score_pool3c(x1, x2.shape[2:])
        x1 = x1 + x2
        del x2
        x1 = self.upscore8(x1)
        # final crop
        x1 = self.score(x1, x_in_shape[2:])
        # optionally collapse
        if as_pmap:
            max_face = x1[:, 1:].max(dim=1)[0].unsqueeze(1)
            x1 = torch.cat((x1[:, 0:1], max_face), dim=1)
            del max_face
            x1 = x1.softmax(dim=1)[:, 1]
        return x1
