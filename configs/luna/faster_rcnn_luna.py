_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
  pretrained=None,
  roi_head=dict(
    bbox_head=dict(
      num_classes=1
    )
  )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'LUNA/'
classes = ('nodule',)

data = dict(
    train=dict(
      type='RepeatDataset',
      times=5,
        dataset=dict(
            type=dataset_type,
            img_prefix='LUNA/',
            classes=classes,
            ann_file='LUNA/train.json',
            pipeline=train_pipeline)
        ),
    val=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json'),
    test=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json'))

optimizer = dict(type='SGD', lr=0.001)
log_config = dict(interval=150)
evaluation = dict(interval=25)
total_epochs = 20
