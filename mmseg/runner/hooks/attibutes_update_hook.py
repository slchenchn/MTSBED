# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from math import cos, pi

import mmcv
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class AttibuteUpdaterHook(Hook):
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(
        self,
        start_value,
        end_value,
        attr_name,
        by_epoch=True,
    ):
        self.by_epoch = by_epoch
        self.start_value = start_value
        self.end_value = end_value
        self.attr_name = attr_name

    def set_attr(self, runner):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        cur_val = (progress / max_progress) * (self.end_value - self.start_value) + self.start_value

        setattr(runner.model.module, self.attr_name, cur_val)
    
    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        # setattr(runner.model, self.attr_name, self.start_value)
        self.set_attr(runner)

    def before_train_iter(self, runner):
        self.set_attr(runner)
        # cur_iter = runner.iter
        # if not self.by_epoch:
        #     self.regular_lr = self.get_regular_lr(runner)
        #     self._set_lr(runner, self.regular_lr)
        # elif self.by_epoch:
        #     if cur_iter > self.warmup_iters:
        #         return
        #     elif cur_iter == self.warmup_iters:
        #         self._set_lr(runner, self.regular_lr)
        #     else:
        #         warmup_lr = self.get_warmup_lr(cur_iter)
        #         self._set_lr(runner, warmup_lr)