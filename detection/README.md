## [TITS24] SDPT: Semantic-Aware Dimension-Pooling Transformer for Image Segmentation

This folder contains full training and test code for object detection and instance segmentation.

### Requirements

* mmdetection == 2.14

We train each model based on `mmdetection==2.8.0`.
Since new GPU cards (RTX 3000 series) should compile mmcv from source to support this early version,
we reorganize the config and support newer mmdetection version. 
Therefore, you can simply reproduce the result on newer GPUs.

### Data Preparation

Put MS COCO dataset files to `data/coco/`.

### Object Detection

Tested on the coco validation set


|  Base Model    | Variants  | AP | AP@0.5 | AP@0.75 | #Params (M) | # GFLOPS |
| :--: | :-------: | :--: | :--: | :---------: | :------: | :----------------------------------------------------------: |
| RetinaNet    | SDPT-Tiny  | 41.3 | 62.0 |    44.1    |    21.1    |   206   |
| RetinaNet  | SDPT-Small | 44.4 | 65.3 |    47.6    |    33.8    |   260   |
| RetinaNet  | SDPT-Base  | 46.1 | 67.5 |    49.6    |    45.8    |   344    |

Use this address to access all logs: [[Google Drive]](https://drive.google.com/drive/folders/1fcg7n3Ga8cYoT-3Ar0PeQXjAC3AnQYyY?usp=sharing)

### Instance Segmentation 

Tested on the coco val set


|  Base Model    | Variants  | APb | APb@0.5 | APm  | APm@0.5 | #Params (M) | # GFLOPS |
| :--: | :-------: | :--: | :--: | :---------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Mask R-CNN | SDPT-Tiny  | 43.3 | 65.7 |    39.6    |    62.5    |    31.3     |   225   |
| Mask R-CNN | SDPT-Small | 45.5 | 67.7 |    41.4    |    64.6    |    43.7     |   279   |
| Mask R-CNN | SDPT-Base  | 47.2 | 69.3 |    42.7    |    66.1    |    55.7    |   363   |

`APb` denotes AP box metric, and `APm` is the AP mask metric.

Use this address to access all logs: [[Google Drive]](https://drive.google.com/drive/folders/1fcg7n3Ga8cYoT-3Ar0PeQXjAC3AnQYyY?usp=sharing)


### Train

Before training, please make sure you have `mmdetection==2.14` and the ImageNet-pretrained SDPT weights
Put them to `pretrained/` folder.

Use the following commands to train `Mask R-CNN` with `SDPT-Tiny` backbone for distributed learning with 8 GPUs:

````
bash dist_train.sh configs/mask_rcnn_sdpt_t_fpn_1x_coco.py 8
````

Other configs are on the `configs` directory.

### Validate

Put the pretrained model to `pretrained` folder.
Then, use the following commands to validate the model in a single GPU:

````
bash dist_test.sh configs/mask_rcnn_sdpt_t_fpn_1x_coco.py pretrained/mask_rcnn_sdpt_t_fpn_1x_coco-d875fa68.pth 1
````


### Other Notes

If you meet any problems, please do not hesitate to contact us.
Issues and discussions are welcome in the repository!
You can also contact us via sending messages to this email: hu.cao@tum.de



### Citation

If you are using the code/model/data provided here in a publication, please consider citing our works:

````
@ARTICLE{10584449,
  author={Cao, Hu and Chen, Guang and Zhao, Hengshuang and Jiang, Dongsheng and Zhang, Xiaopeng and Tian, Qi and Knoll, Alois},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={SDPT: Semantic-Aware Dimension-Pooling Transformer for Image Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Transformers;Image segmentation;Decoding;Task analysis;Semantics;Image edge detection;Computational efficiency;Image segmentation;vision transformer;dimension-pooling attention;semantic-balanced decoder;scene understanding},
  doi={10.1109/TITS.2024.3417813}}
````

### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Non-Commercial use only. Any commercial use should get formal permission first.

