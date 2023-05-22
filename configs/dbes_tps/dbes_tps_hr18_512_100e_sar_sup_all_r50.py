_base_ = [
    './dbes_tps_hr18_512_100e_sar_sup_all.py'
]


data = dict(
    train=dict(
        labeled=dict(
            ratio=0.5,
        )
    )
)
