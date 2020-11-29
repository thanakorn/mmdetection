_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

model = dict(
  pretrained=None
)

# dataset settings
dataset_type = 'COCODataset'
data_root = 'LUNA/'
classes = ('nodule',)

data = dict(
    train=dict(
          img_prefix='LUNA/',
          classes=classes,
          ann_file='LUNA/train.json'
        ),
    val=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json',),
    test=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json'))
total_epochs = 10