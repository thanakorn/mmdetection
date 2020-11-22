_base_ = '../ssd/ssd300_coco.py'
input_size = 512
model = dict(
    pretrained=None,
    backbone=dict(input_size=input_size),
    bbox_head=dict(num_classes=1))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'LUNA/'
classes = ('nodule',)

data = dict(
    train=dict(
        dataset=dict(
            img_prefix='LUNA/',
            classes=classes,
            ann_file='LUNA/annotation.json')
        ),
    val=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/annotation.json'),
    test=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/annotation.json'))