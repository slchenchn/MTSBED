# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import numpy as np
from PIL import Image
import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose, LoadAnnotations


@DATASETS.register_module()
class SARSuperviseDataset(CustomDataset):
    ''' SAR supervised building segmentation dataset '''
    CLASSES = ("background", "building")

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, *args, img_dir="", ratio=1.0, **kargs):
        if not isinstance(ratio, list):
            ratio = [0, ratio]
        assert len(ratio) == 2

        self.ratio = ratio
        super().__init__(*args, img_dir=img_dir, **kargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            self.img_dir = self.data_root
            self.ann_dir = self.data_root
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            lf = int(len(lines) * self.ratio[0])
            rg = int(len(lines) * self.ratio[1])
            lines = lines[lf:rg]
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name)

                seg_map = img_name.replace("imgs", "labels")
                img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            raise NotImplementedError

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class SARUnSuperviseTimeDataset(CustomDataset):
    ''' Unsupervised SAR dataset '''
    CLASSES = ("background", "building")

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 *args,
                 img_dir="",
                 ratio=1.0,
                 img_suffix=".png",
                 drop_key=None,
                 **kargs):
        self.ratio = ratio
        self.drop_key = drop_key
        super().__init__(
            *args, img_dir=img_dir, img_suffix=img_suffix, **kargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            self.img_dir = self.data_root
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            # lines = lines[:int(len(lines)*self.ratio)]
            for line in lines:
                img_name = line.strip()
                # img_names = (os.listdir())
                img_info = dict(filename=img_name)

                img_infos.append(img_info)
            print_log(
                f"Loaded {len(img_infos)} group of images",
                logger=get_root_logger())
        else:
            """if split file is not given, glob all images in img_dir.
            Currently this is only used for testing unlabeled images
            """
            imgs = mmcv.scandir(
                self.data_root, suffix=img_suffix, recursive=True)
            if self.drop_key is not None:
                imgs = [img for img in imgs if self.drop_key not in img]
            img_infos = [dict(filename=img) for img in imgs]
            print_log(
                f"Loaded {len(img_infos)} images", logger=get_root_logger())
            # raise NotImplementedError

        return img_infos

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
        # raise ValueError(f"unlabeled dataset does not support prepare_test_img")

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
