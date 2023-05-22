_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/sar_semi_all_dbes_v2.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100e.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 2
model = dict(
    type='PseudoLabelV2',
    ignore_index=255,
    pretrained=None,
    decode_head=dict(
        _delete_=True,
        type='TPSFCNDBESV9',
        input_transform='multiple_select',
        ps_thres = 0.90,
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=1,
        norm_cfg=norm_cfg,
        num_classes=num_classes,
        loss_decode=dict(
            type='JointEdgeSegLoss', 
            seg_body_weight=0.1,
            num_classes=num_classes,
            loss_weight=0.5,
            ),
        unsup_loss=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    )
)

runner = dict(type='SemiEpochBasedRunner')

find_unused_parameters = True

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)