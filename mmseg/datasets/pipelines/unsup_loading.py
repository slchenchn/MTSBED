import os
import os.path as osp

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES
from .loading import LoadImageFromFile


@PIPELINES.register_module()
class LoadUnlabeledTimeImages(LoadImageFromFile):
    """load unlabeled and co-registered images, first version.
    NOTE: Deprecated, use LoadUnlabeledTimeImagesV2 instead.    
    """

    def __init__(self, *args, temps=2, kernelsize=(5, 5), **kargs):
        super().__init__(*args, **kargs)
        self.temps = temps
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("img_prefix") is not None:
            dirname = osp.join(results["img_prefix"],
                               results["img_info"]["filename"])
        else:
            dirname = results["img_info"]["filename"]

        fns = [0] * self.temps
        all_fns = list(os.listdir(dirname))
        for i in range(self.temps):
            fns[i] = osp.join(dirname,
                              all_fns[np.random.randint(0, high=len(all_fns))])
        imgs = []
        invalids = []
        for filename in fns:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            imgs.append(img)
            invalids.append(self.get_invalid_mask(img))

        if self.to_float32:
            imgs = [img.astype(np.float32) for img in imgs]

        results["filename"] = fns
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = np.concatenate(imgs, axis=-1)
        results["img_shape"] = imgs[0].shape
        results["ori_shape"] = imgs[0].shape
        # Set initial values for default meta_keys
        results["pad_shape"] = imgs[0].shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        # the invalid pixels is set to 1, and the `gt_semantic_seg` here is the invalid pixels
        results["gt_semantic_seg"] = np.stack(invalids, axis=-1)
        results["seg_fields"].append("gt_semantic_seg")
        return results

    def get_invalid_mask(self, img):
        """invalid pixels is set to 1"""
        if img.ndim > 2:
            # img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bi = (img <= 0).astype(np.uint8)
        opening = cv2.morphologyEx(bi, cv2.MORPH_OPEN, self.kernel)
        return opening

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += f"number of co-registered images={self.temps}"
        return repr_str


@PIPELINES.register_module()
class LoadUnlabeledTimeImagesV2(LoadUnlabeledTimeImages):
    """load unlabeled and co-registered images, second version. This version additionally detects the invalid pixel masks in the images.
    """

    def __init__(self, *args, max_n_imgs=4, **kargs):
        super().__init__(*args, **kargs)
        self.max_n_imgs = max_n_imgs

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("img_prefix") is not None:
            dirname = osp.join(results["img_prefix"],
                               results["img_info"]["filename"])
        else:
            dirname = results["img_info"]["filename"]

        fns = [0] * self.temps
        all_fns = list(os.listdir(dirname))
        for i in range(self.temps):
            fns[i] = osp.join(
                dirname,
                all_fns[np.random.randint(
                    0, high=min(self.max_n_imgs, len(all_fns)))],
            )
        imgs = []
        invalids = []
        for filename in fns:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            imgs.append(img)
            invalids.append(self.get_invalid_mask(img))

        if self.to_float32:
            imgs = [img.astype(np.float32) for img in imgs]

        results["filename"] = fns
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = np.concatenate(imgs, axis=-1)
        results["img_shape"] = imgs[0].shape
        results["ori_shape"] = imgs[0].shape
        # Set initial values for default meta_keys
        results["pad_shape"] = imgs[0].shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        # the invalid pixels is set to 1, and the `gt_semantic_seg` here is the invalid pixels
        results["gt_semantic_seg"] = np.stack(invalids, axis=-1)
        results["seg_fields"].append("gt_semantic_seg")
        return results
