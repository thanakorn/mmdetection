_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

model=dict(
  pretrained=None,
  bbox_head=dict(num_classes=1)
)

# pipeline settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
