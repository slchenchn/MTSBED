''' semi-supervised learning '''
dataset_type = 'SARSuperviseDataset'
unlabeled_dataset_type='SARUnSuperviseTimeDataset'
data_root = 'data/seg_data4'
unlabeled_data_root = 'data/u_seg_data'
city = 'all'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (512, 512)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    
    # put the edge transform at the end, avoiding conflict with the
    # origianl transforms
    dict(type='RelaxedBoundaryLossToTensor', num_classes=2),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

unlabeled_train_pipeline = [
    dict(type='LoadUnlabeledTimeImages'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='NormalizeMultiImages', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        labeled=dict(
            type=dataset_type,
            data_root=data_root,
            split=f'split/{city}_train.txt',
            pipeline=train_pipeline
        ),
        unlabeled=dict(
            type=unlabeled_dataset_type,
            data_root=unlabeled_data_root,
            split=f'split/{city}_v20.txt',
            pipeline=unlabeled_train_pipeline,
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split=f'split/{city}_val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split=f'split/{city}_test.txt',
        pipeline=test_pipeline)
    )