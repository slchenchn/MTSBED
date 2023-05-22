from mmcv.runner import get_dist_info


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ("auto_scale_lr" not in cfg) or (not cfg.auto_scale_lr.get("enable", False)):
        logger.info("Automatic scaling of learning rate (LR)" " has been disabled.")
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get("base_batch_size", None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(
        f"Training with {num_gpus} GPU(s) with {samples_per_gpu} "
        f"samples per GPU. The total batch size is {batch_size}."
    )

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        ratio = (batch_size / base_batch_size)
        scaled_lr = ratio * cfg.optimizer.lr
        logger.info(
            "LR has been automatically scaled "
            f"from {cfg.optimizer.lr} to {scaled_lr}"
        )
        cfg.optimizer.lr = scaled_lr
        if cfg.lr_config.get('min_lr', False):
            scaled_lr = ratio * cfg.lr_config.min_lr
            logger.info(
                "min LR has been automatically scaled "
                f"from {cfg.lr_config.min_lr} to {scaled_lr}"
            )
            cfg.lr_config.min_lr = scaled_lr
    else:
        logger.info(
            "The batch size match the "
            f"base batch size: {base_batch_size}, "
            f"will not scaling the LR ({cfg.optimizer.lr})."
        )
