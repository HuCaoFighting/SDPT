_base_ = [
    '../../_base_/models/sdpt.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
gn_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = False
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[48, 96, 240, 384],
        depths=[2, 2, 6, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/SDPT_small.pth')),
    decode_head=dict(
        type='SdptformerHead',
        in_channels=[96, 240, 384],
        in_index=[1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=gn_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8)
evaluation = dict(interval=8000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=8000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
