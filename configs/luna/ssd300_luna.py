_base_ = '../ssd/ssd300_coco.py'
input_size = 512
model = dict(
    pretrained=None,
    backbone=dict(input_size=input_size),
    bbox_head=dict(
      num_classes=1,
      in_channels=(512, 1024, 512, 256, 256, 256, 256),
      anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])
    ))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'LUNA/'
classes = ('nodule',)

data = dict(
    train=dict(
        dataset=dict(
            img_prefix='LUNA/',
            classes=classes,
            ann_file='LUNA/train.json')
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
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
log_config = dict(interval=10)