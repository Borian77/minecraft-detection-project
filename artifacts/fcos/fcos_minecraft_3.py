classes = (
    'pig',
    'chicken',
    'cow',
    'zombie',
    'skeleton',
    'creeper',
    'spider',
    'turtle',
    'llama',
    'ghast',
    'fox',
    'frog',
    'goat',
    'sheep',
    'bee',
    'enderman',
    'wolf',
    'slime',
)
data_root = 'datasets/minecraft/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
device = 'cuda'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet50_caffe',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    bbox_head=dict(
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='IoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=18,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='FCOSHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            102.9801,
            115.9465,
            122.7717,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    type='FCOS')
num_classes = 18
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/images/'),
        data_root='datasets/minecraft/',
        metainfo=dict(
            classes=(
                'pig',
                'chicken',
                'cow',
                'zombie',
                'skeleton',
                'creeper',
                'spider',
                'turtle',
                'llama',
                'ghast',
                'fox',
                'frog',
                'goat',
                'sheep',
                'bee',
                'enderman',
                'wolf',
                'slime',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='datasets/minecraft/annotations/test.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/images/'),
        data_root='datasets/minecraft/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'pig',
                'chicken',
                'cow',
                'zombie',
                'skeleton',
                'creeper',
                'spider',
                'turtle',
                'llama',
                'ghast',
                'fox',
                'frog',
                'goat',
                'sheep',
                'bee',
                'enderman',
                'wolf',
                'slime',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/valid.json',
        data_prefix=dict(img='valid/images/'),
        data_root='datasets/minecraft/',
        metainfo=dict(
            classes=(
                'pig',
                'chicken',
                'cow',
                'zombie',
                'skeleton',
                'creeper',
                'spider',
                'turtle',
                'llama',
                'ghast',
                'fox',
                'frog',
                'goat',
                'sheep',
                'bee',
                'enderman',
                'wolf',
                'slime',
            )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='datasets/minecraft/annotations/valid.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'artifacts/fcos'
