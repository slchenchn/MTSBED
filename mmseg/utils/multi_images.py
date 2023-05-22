'''
Author: Shuailin Chen
Created Date: 2021-06-21
Last Modified: 2022-02-18
	content: 
'''

import os
import os.path as osp
import numpy as np
import torch
from torch import Tensor
from mmcv.parallel.data_container import DataContainer

from mylib import image_utils as iu


def split_batches(x: Tensor):
    ''' Split a 2*B batch of images into two B images per batch, in order to adapt to MMsegmentation '''

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size: , ...]
    return x1, x2


def merge_batches(x1: Tensor, x2: Tensor):
    ''' merge two batchs each contains B images into a 2*B batch of images in order to adapt to MMsegmentation '''

    assert x1.ndim == 4 and x2.ndim == 4,   f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)


def split_images(x: Tensor):
    ''' Split a 2*c channels image into two c channels images, in order to adapt to MMsegmentation '''

    # determine 3D tensor or 4D tensor
    if x.ndim == 4:
        channels = x.shape[1]
        assert channels % 2 == 0
        channels //= 2
        x1 = x[:, 0:channels, :, :]
        x2 = x[:, channels:, :, :]
    elif x.ndim == 3:
        channels = x.shape[-1]
        assert channels % 2 == 0
        channels //= 2
        x1 = x[..., 0:channels]
        x2 = x[..., channels: ]
    else:
        raise ValueError(f'dimension of x should be 3 or 4, but got {x.ndim}')
        
    return x1, x2


def visualize_multiple_images(x, dst_path, channel_per_image=3):
    ''' Visualize multiple images from a concatenated images file

    Args:
        x (ndarray | Tensor | DataContainer): concatenated images file, in
            shape of [height, width, channel]
		dst_path (str): destination path to save images
        channel_per_image (int): channels per image. Default:3
    '''
    dtype = 'numpy'
    if isinstance(x, DataContainer):
        x = x.data
        dtype = 'torch'

    # add to 4D tensor, in shape of [batch, height, width, channel]
    if x.ndim < 3:
        x = x[None, ..., None]
    elif x.ndim == 3:
        x = x[None, ...]
    elif x.ndim == 4 and dtype=='torch':
        x = x.permute((0, 2, 3, 1))
    
    # if dtype=='torch':
    #     x = x.permute((0, 2, 3, 1))

    # change to numpy
    if isinstance(x, Tensor):
        x = x.numpy()
    
    c = x.shape[-1]
    try:
        assert c % channel_per_image == 0, f'channels of image ({c}) must be divisible to by {channel_per_image}'
    except:
        pass
    root, ext = osp.splitext(dst_path)
    for kk in range(x.shape[0]):
        for ii in range(c//channel_per_image):
            img = x[kk, :, :, channel_per_image*ii:(ii+1)*channel_per_image].squeeze()
            img_path = root + f'_{kk}_{ii}' + ext
            iu.save_image_by_cv2(img, dst_path=img_path, if_norm=True, is_bgr=True)