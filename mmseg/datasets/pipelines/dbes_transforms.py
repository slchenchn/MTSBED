''' tool functions of `Improving Semantic Segmentation via Decoupled Body and Edge Supervision` 
'''
import mmcv
import numpy as np
from scipy.ndimage import shift
from skimage.segmentation import find_boundaries

from ..builder import PIPELINES
from mmseg.utils.config_dbes import cfg
from .edge_utils import onehot_to_binary_edges


@PIPELINES.register_module()
class RelaxedBoundaryLossToTensor():
    """
    Boundary Relaxation
    """

    def __init__(self, num_classes, ignore_id=255):
        super().__init__()
        self.ignore_id = ignore_id
        self.num_classes = num_classes

    def new_one_hot_converter(self, a):
        ''' '''
        ncols = self.num_classes + 1
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (ncols, )
        return out

    # def apply(self, image, **params) -> np.ndarray:
    #     return image

    # def apply_to_mask(self, mask, **params):
    #     return self.vanilla_call(mask)
    def __call__(self, results):
        mask = results['gt_semantic_seg']
        mask[mask >= self.num_classes] = self.ignore_id

        img_arr = np.array(mask)
        img_arr[img_arr == self.ignore_id] = self.num_classes

        if cfg.STRICTBORDERCLASS != None:
            one_hot_orig = self.new_one_hot_converter(img_arr)
            mask = np.zeros((img_arr.shape[0], img_arr.shape[1]))
            for cls in cfg.STRICTBORDERCLASS:
                mask = np.logical_or(mask, (img_arr == cls))
        one_hot = 0

        border = cfg.BORDER_WINDOW
        if cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH:
            border = border // 2
            border_prediction = find_boundaries(
                img_arr, mode="thick").astype(np.uint8)

        for i in range(-border, border + 1):
            for j in range(-border, border + 1):
                shifted = shift(img_arr, (i, j), cval=self.num_classes)
                one_hot += self.new_one_hot_converter(shifted)

        one_hot[one_hot > 1] = 1

        if cfg.STRICTBORDERCLASS != None:
            one_hot = np.where(np.expand_dims(mask, 2), one_hot_orig, one_hot)

        one_hot = np.moveaxis(one_hot, -1, 0)

        if cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH:
            one_hot = np.where(border_prediction, 2 * one_hot, 1 * one_hot)
            # print(one_hot.shape)

        # _edgemap = np.array(mask_trained)
        _edgemap = one_hot[:-1, :, :]  # c, h, w
        _edgemap = onehot_to_binary_edges(_edgemap, 2, 2)  # h, w
        edgemap = _edgemap.astype(np.float32)

        results['gt_semantic_seg'] = np.concatenate([one_hot, edgemap], axis=0)
        return results