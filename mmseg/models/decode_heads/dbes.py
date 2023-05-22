import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import add_prefix
from mmseg.datasets.pipelines import RelaxedBoundaryLossToTensor
from mmseg.ops import resize
from ..builder import HEADS, build_loss
from ..losses import accuracy, onehot2label
from .dbes import SqueezeBodyEdge, Upsample
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNDBESV2(BaseDecodeHead):
    """ Second version. Use the 1/4 resolution as the detail input, not the 1/8 resolution"""

    def __init__(self, detail_idx=0, **kargs):
        super().__init__(**kargs)

        self.detail_idx = detail_idx
        c1 = sum(self.in_channels)
        c2 = self.in_channels[detail_idx]
        self.pre_conv = nn.Conv2d(c1, 256, 1, bias=False)

        Norm2d = partial(build_norm_layer, self.norm_cfg)
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)
        self.bot_fine = nn.Conv2d(c2, 48, kernel_size=1, bias=False)
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False),
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False),
        )

        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False),
        )
        self.sigmoid_edge = nn.Sigmoid()

    def cls_seg(self, feat):
        raise NotImplementedError(f"this is implented in forward()")

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        x1 = inputs[self.detail_idx]  # c2 feature
        x = torch.cat(
            [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ],
            dim=1,
        )
        x = self.pre_conv(x)  # change dim from 270 to 256
        seg_body, seg_edge = self.squeeze_body_edge(x)

        fine_size = x1.size()
        dec0_fine = self.bot_fine(x1)

        seg_edge = self.edge_fusion(
            torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1)
        )
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        x = Upsample(x, fine_size[2:])

        seg_out = torch.cat([x, seg_out], dim=1)
        seg_final_out = self.final_seg(seg_out)

        seg_edge_out = self.sigmoid_edge(seg_edge_out)
        seg_body_out = self.dsn_seg_body(seg_body)

        if seg_final_out.isnan().any():
            print("breakpoint")
        return seg_final_out, seg_body_out, seg_edge_out

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, **kargs):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze()
        seg_logit = [
            resize(
                input=logit,
                size=seg_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for logit in seg_logit
        ]
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs,
                )

        loss["acc_seg"] = accuracy(
            seg_logit[0], onehot2label(seg_label), ignore_index=self.ignore_index
        )

        for k, v in loss.copy().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    loss[f"{k}_{k2}"] = v2
                del loss[k]
        return loss

    def forward_test(self, inputs, img_metas, test_cfg):
        final, body, edge = self.forward(inputs)
        return final

@HEADS.register_module()
class TPSFCNDBESV9(FCNDBESV2):
    """only calculate the DBES loss in labeled samples"""

    def __init__(
        self,
        ps_thres,
        unsup_loss,
        add_ps_loss=True,
        add_cross_loss=True,
        cross_thres=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unsup_loss = build_loss(unsup_loss)
        self.ps_thres = ps_thres
        self.add_ps_loss = add_ps_loss
        self.add_cross_loss = add_cross_loss
        if cross_thres is None:
            self.cross_thres = self.ps_thres
        else:
            self.cross_thres = cross_thres

    def _extract_feat(self, labeled, unlabeled, label_size):

        # split two temporals
        unlabel1 = unlabeled["feat"][0]
        unlabel2 = unlabeled["feat"][1]

        logits_labeled = self.forward(labeled["feat"])
        logits_unlabel1 = self.forward(unlabel1)
        logits_unlabel2 = self.forward(unlabel2)

        return logits_labeled, logits_unlabel1, logits_unlabel2

    def forward_train(self, labeled, unlabeled, train_cfg, **kargs):
        logits_labeled, logits_unlabel, logits_unlabe2 = self._extract_feat(
            labeled, unlabeled, labeled["gt_semantic_seg"].shape[-2:]
        )

        invalids = unlabeled["gt_semantic_seg"].squeeze().bool()
        invalids = (
            resize(
                input=invalids.permute(0, 3, 1, 2).float(),
                size=logits_unlabel[0].shape[-2:],
                mode="nearest",
            )
            .bool()
            .squeeze()
        )
        invalids1 = invalids[:, 0, ...].unsqueeze(1)
        invalids2 = invalids[:, 1, ...].unsqueeze(1)

        with torch.no_grad():
            # final map
            prob1 = F.softmax(logits_unlabel[0], dim=1)
            prob2 = F.softmax(logits_unlabe2[0], dim=1)
            val1, ps1 = prob1.max(dim=1, keepdims=True)
            val2, ps2 = prob2.max(dim=1, keepdims=True)

            ps1[invalids1] = self.ignore_index
            ps2[invalids2] = self.ignore_index

            thres_valid_map1 = val1 > self.ps_thres
            thres_valid_map2 = val2 > self.ps_thres
            valid_map = thres_valid_map1 & thres_valid_map2 & ~invalids1 & ~invalids2
            valid_map1 = (thres_valid_map1 & invalids2) | valid_map
            valid_map2 = (thres_valid_map2 & invalids1) | valid_map
            ps1[~valid_map1] = self.ignore_index
            ps2[~valid_map2] = self.ignore_index

            # body map
            prob1 = F.softmax(logits_unlabel[1], dim=1)
            prob2 = F.softmax(logits_unlabe2[1], dim=1)
            val1, ps1_body = prob1.max(dim=1, keepdims=True)
            val2, ps2_body = prob2.max(dim=1, keepdims=True)

            ps1_body[invalids1] = self.ignore_index
            ps2_body[invalids2] = self.ignore_index

            thres_valid_map1 = val1 > self.ps_thres
            thres_valid_map2 = val2 > self.ps_thres
            valid_map = thres_valid_map1 & thres_valid_map2 & ~invalids1 & ~invalids2
            valid_map1 = (thres_valid_map1 & invalids2) | valid_map
            valid_map2 = (thres_valid_map2 & invalids1) | valid_map
            ps1_body[~valid_map1] = self.ignore_index
            ps2_body[~valid_map2] = self.ignore_index

        loss_labeled = self.losses(logits_labeled, labeled["gt_semantic_seg"])
        loss_unlabele1 = self.unsup_lossess(logits_unlabel[0], ps1)
        loss_unlabele2 = self.unsup_lossess(logits_unlabe2[0], ps2)
        loss_unlabele1_body = self.unsup_lossess(logits_unlabel[1], ps1_body)
        loss_unlabele2_body = self.unsup_lossess(logits_unlabe2[1], ps2_body)

        losses = dict()
        losses.update(add_prefix(loss_labeled, "labeled"))
        losses.update(add_prefix(loss_unlabele1, "unlabel1"))
        losses.update(add_prefix(loss_unlabele2, "unlabel2"))
        losses.update(add_prefix(loss_unlabele1_body, "unlabel1_body"))
        losses.update(add_prefix(loss_unlabele2_body, "unlabel2_body"))
        return losses

    @force_fp32(apply_to=("seg_logit",))
    def unsup_lossess(self, seg_logit, seg_label, **kargs):
        """Compute segmentation loss."""
        loss = dict()
        # seg_label = seg_label.squeeze()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.unsup_loss]
        else:
            losses_decode = self.unsup_loss
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                    **kargs,
                )

        loss["acc_seg"] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)

        for k, v in loss.copy().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    loss[f"{k}_{k2}"] = v2
                del loss[k]
        return loss
