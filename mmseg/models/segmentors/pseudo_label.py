"""
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-09-03
	content: 
"""

from copy import deepcopy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torchvision.transforms.functional import normalize

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.datasets.pipelines import Compose
from ..builder import SEGMENTORS
from .semi_v2 import SemiV2


@SEGMENTORS.register_module()
class PseudoLabel(SemiV2):
    """my implementation of online pseudo labeling"""

    def __init__(self, ignore_index=255, **kargs):
        super().__init__(**kargs)
        self.ignore_index = ignore_index

    def forward_train(self, labeled: dict, unlabeled: dict, **kargs):

        # split two temporals
        unlabled_img = unlabeled["img"]
        assert unlabled_img.shape[1] == 6
        unlabel1_img = unlabled_img[:, :3, ...]
        unlabel2_img = unlabled_img[:, 3:, ...]

        feat_labeled = self.extract_feat(labeled["img"])
        feat_unlabel1 = self.extract_feat(unlabel1_img)
        feat_unlabel2 = self.extract_feat(unlabel2_img)

        labeled["feat"] = feat_labeled
        unlabeled["feat"] = [feat_unlabel1, feat_unlabel2]
        loss_decode = self._decode_head_forward_train(labeled, unlabeled,
                                                      **kargs)

        losses = dict()
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            raise NotImplementedError

        return losses

    def _decode_head_forward_train(self, labeled, unlabeled, **kargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(labeled, unlabeled,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, "decode"))
        return losses


@SEGMENTORS.register_module()
class PseudoLabelV2(PseudoLabel):
    ''' Second version of PseudoLabel. In this version, we use the same amount of unlabeled and labeled data in each iteration. '''

    def forward_train(self, labeled: dict, unlabeled: dict, **kargs):

        unlabled_img = unlabeled["img"]
        bs = unlabled_img.shape[0] // 2

        unlabeled["img"] = unlabled_img[:bs, ...]
        unlabeled["gt_semantic_seg"] = unlabeled["gt_semantic_seg"][:bs, ...]
        unlabeled["img_metas"] = unlabeled["img_metas"][:bs]
        return super().forward_train(labeled, unlabeled, **kargs)
