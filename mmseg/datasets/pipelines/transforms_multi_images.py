
import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from mmseg.utils import split_images
from PIL import ImageFilter
from PIL import Image

from ..builder import PIPELINES
from .transforms import PhotoMetricDistortion, Normalize


@PIPELINES.register_module()
class NormalizeMultiImages(Normalize):
    """Normalize multiple images, see class Normalize for detail
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        img1, img2 = split_images(img)

        result = dict(img=img1)
        result = super().__call__(result)
        img1 = result['img']
        img_norm_cfg = result['img_norm_cfg']
        
        result = dict(img=img2)
        img2 = super().__call__(result)['img']
    
        results['img'] = np.concatenate((img1, img2), axis=-1)
        results['img_norm_cfg'] = img_norm_cfg

        return results


@PIPELINES.register_module()
class NormalizeMultiImagesV2(Normalize):
    """Normalize multiple images, v2 version, this version can handle more than 2 stacked images
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def __call__(self, results):
        img = results['img']
        n_imgs = img.shape[-1] // 3
        split_imgs = np.split(img, n_imgs, axis=-1)

        res_imgs = []
        for tmp_img in split_imgs:
            tmp_result = dict(img=tmp_img)
            tmp_result = super().__call__(tmp_result)
            res_imgs.append(tmp_result['img'])

        img_norm_cfg = tmp_result['img_norm_cfg']
        assert len(res_imgs) == n_imgs
        results['img'] = np.concatenate(res_imgs, axis=-1)
        results['img_norm_cfg'] = img_norm_cfg

        return results