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

