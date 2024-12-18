_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    # '../configs/_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='SDPT_tiny',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/SDPT_tiny.pth'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 240, 384],
        out_channels=256,
        num_outs=5)
)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

total_epochs = 12
fp16 = None
find_unused_parameters = True