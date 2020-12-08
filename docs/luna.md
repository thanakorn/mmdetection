# Training Nodule Detection model

## Clone the project
```
git checkout https://github.com/thanakorn/mmdetection.git
```

## Prepare the dataset

Before training the model, the raw data(**.raw** and **.mhd** files) must be transformed into images. Then, the annotation file containing image list and ground truth in COCO 
format has to be provided(See [here](https://github.com/thanakorn/mmdetection/blob/master/docs/tutorials/customize_dataset.md) for annotation file structure). If there are multiple
image sets(train/test/val), each set needs a dedicated annotation file. The structure of the dataset folder should be as follow:

```plain
dataset
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-0.jpg
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-1.jpg
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-13.jpg
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-15.jpg
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-16.jpg
├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217-18.jpg
...
├── train.json
├── test.json
├── val.json
```

## Install dependencies

Follow OpenMM instructions [here](https://github.com/thanakorn/mmdetection/blob/master/docs/get_started.md) for dependency installations.

## Training the model

After finish installing OpenMM, the object-detection model is ready to be trained. All you need to do is to provide the configuration. There are 3 models which are provided 
at the moment: YOLOv3, Faster-RCNN, and SSD. 

Modify the `dataset_root` and `ann_file` in the [config file](https://github.com/thanakorn/mmdetection/tree/master/configs/luna) to be your dataset directory and annotation file respectively. Then, run the following command:

`python tools/train.py configs/luna/{config_file_name}`

While running, the weights will be constantly(depends on config) save as `.pth` file in `./work_dirs` directory.

## Using the model

Once the training finish, the file `latest.pth` will be generated. This file can be used to do inference as follow:

```python
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config_file = $config_file$
checkpoint_file = $weight_file$
img = $image_file$
model = init_detector(config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
```
