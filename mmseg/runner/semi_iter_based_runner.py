'''
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-08-31
	content: iteration based runner for semi-supervision
'''
import time
import warnings
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner import IterBasedRunner
from mmcv.runner.utils import get_host_info
from mmcv.runner.iter_based_runner import IterLoader


@RUNNERS.register_module()
class SemiIterBasedRunner(IterBasedRunner):
    ''' Iteration based runner for semi-supervision, support multiple dataloader in training
    '''

    def train(self, data_loader:dict, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = list(data_loader.values())[0].epoch
        # print(f'self._epoch: {self._epoch}')
        data_batch = {k: next(v) for k, v in data_loader.items()}
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def run(self, data_loaders:dict, workflow:list, max_iters=None, **kwargs):
        """ Start running.

        Args:
            data_loaders (dict[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """

        # miscellaneous adapted from IterBasedRunner
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        # multiple dataloaders for training, change iter_loaders from `list` to `dict`
        iter_loaders = {}
        for data_loader, (mode, _) in zip(data_loaders, workflow):
            if isinstance(data_loader, dict):
                # tmp = {mode: {k: v for k, v in data_loader.items()}}
                iter_loaders.update({mode: {k: IterLoader(v) for k, v in data_loader.items()}})
            else:
                iter_loaders.update({mode: IterLoader(data_loader)})

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                # print(f'iters: {iters}, iter_runner: {iter_runner}')
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[mode], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
