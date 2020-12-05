_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
  pretrained=None,
  roi_head=dict(
    bbox_head=dict(
      num_classes=1
    )
  )
)

# dataset settings
dataset_type = 'CocoDataset'
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
        ann_file='LUNA/train.json'),
    test=dict(
        img_prefix='LUNA/',
        classes=classes,
        ann_file='LUNA/train.json'))

optimizer = dict(type='SGD', lr=0.001)
log_config = dict(interval=50)
evaluation = dict(interval=20)
total_epochs = 10
