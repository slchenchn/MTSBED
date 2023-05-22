"""
Author: Shuailin Chen
Created Date: 2021-08-28
Last Modified: 2021-08-31
	content: iteration based runner for semi-supervision
"""
import time
import warnings
import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner.utils import get_host_info
from mmcv.runner.epoch_based_runner import EpochBasedRunner


@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    """Iteration based runner for semi-supervision, support multiple dataloader in training"""

    def train(self, data_loader: dict, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader['labeled'])
        self.call_hook("before_train_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        labeled_data_loader = self.data_loader["labeled"]
        unlabeled_data_loader = self.data_loader["unlabeled"]
        for i, (labeled_batch, unlabeled_batch) in enumerate(
            zip(labeled_data_loader, unlabeled_data_loader)
        ):
            data_batch = {"labeled": labeled_batch, "unlabeled": unlabeled_batch}
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook("after_train_iter")
            del self.data_batch
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run(self, data_loaders: dict, workflow: list, max_epochs=None, **kwargs):
        """Start running.

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
        if max_epochs is not None:
            warnings.warn(
                "setting max_epochs in run is deprecated, "
                "please set max_epochs in runner_config",
                DeprecationWarning,
            )
            self._max_epochs = max_epochs

        assert (
            self._max_epochs is not None
        ), "max_epochs must be specified during instantiation"

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.info(
            "Hooks will be executed in the following order:\n%s", self.get_hook_info()
        )
        self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run")

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an ' "epoch"
                        )
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        "mode in workflow must be a str, but got {}".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_run")
