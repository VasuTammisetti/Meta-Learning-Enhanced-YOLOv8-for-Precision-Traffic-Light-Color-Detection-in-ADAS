Train_annot = 'train/_annotations.coco.json'
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
cudnn_benchmark = True
data_root = '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = 300
interval = 10
launcher = 'none'
load_from = '/content/drive/MyDrive/mmdetection_vishal/mmdetection/weights/epoch_45.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 150
metainfo = dict(classes=(
    'Green',
    'Orange',
    'Red',
))
model = dict(
    backbone=dict(
        ceil_mode=True,
        depth=16,
        init_cfg=dict(
            checkpoint='open-mmlab://vgg16_caffe', type='Pretrained'),
        out_feature_indices=(
            22,
            34,
        ),
        out_indices=(
            3,
            4,
        ),
        type='SSDVGG',
        with_last_pool=False),
    bbox_head=dict(
        anchor_generator=dict(
            basesize_ratio_range=(
                0.15,
                0.9,
            ),
            input_size=300,
            ratios=[
                [
                    2,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                ],
                [
                    2,
                ],
            ],
            scale_major=False,
            strides=[
                8,
                16,
                32,
                64,
                100,
                300,
            ],
            type='SSDAnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        in_channels=(
            512,
            1024,
            512,
            256,
            256,
            256,
        ),
        num_classes=3,
        type='SSDHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            1,
            1,
            1,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=(
            512,
            1024,
        ),
        l2_norm_scale=20,
        level_paddings=(
            1,
            1,
            0,
            0,
        ),
        level_strides=(
            2,
            2,
            1,
            1,
        ),
        out_channels=(
            512,
            1024,
            512,
            256,
            256,
            256,
        ),
        type='SSDNeck'),
    test_cfg=dict(
        max_per_img=200,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.45, type='nms'),
        nms_pre=1000,
        score_thr=0.02),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            gt_max_assign_all=False,
            ignore_iof_thr=-1,
            min_pos_iou=0.0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        neg_pos_ratio=3,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0),
    type='SingleStageDetector')
num_classes = 3
num_last_epochs = 15
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=2000, start_factor=0.001,
        type='LinearLR'),
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
resume = True
seed = 86
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(
            img='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/'
        ),
        data_root='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/',
        metainfo=dict(classes=(
            'Green',
            'Orange',
            'Red',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
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
    ann_file=
    '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/_annotations.coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
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
train_cfg = dict(max_epochs=150, type='EpochBasedTrainLoop', val_interval=1)
train_data_prefix = 'train/'
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file=
        '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/train/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(
            img='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/train/'
        ),
        data_root='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=(
            'Green',
            'Orange',
            'Red',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_annot = 'valid/_annotations.coco.json'
val_cfg = dict(type='ValLoop')
val_data_prefix = 'valid/'
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(
            img='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/'
        ),
        data_root='/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/',
        metainfo=dict(classes=(
            'Green',
            'Orange',
            'Red',
        )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
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
    ann_file=
    '/content/drive/MyDrive/mmdetection_vishal/data1/Traffic_data-3/valid/_annotations.coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/content/drive/MyDrive/mmdetection_vishal/mmdetection/weights/'
