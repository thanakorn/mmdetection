_base_ = '../ssd/ssd512_coco.py'

model=dict(
  backbone=dict(type='MobileNetV2', widen_factor=1., out_indices=(7, )),
  pretrained=None,
  bbox_head=dict(num_classes=1)  
)

# pipeline settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'LUNA/'
classes = ('nodule',)
data = dict(
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/val.json',
        filter_empty_gt=False),
    test=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/test.json',
        filter_empty_gt=False))

# optimizer settings
optimizer = dict(_delete_=True, type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# runtime settings
total_epochs = 20
log_config = dict(interval=500)
evaluation = dict(interval=10)
workflow = [('train', 1)]
