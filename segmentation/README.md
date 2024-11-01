# SDPT: Semantic-Aware Dimension-Pooling Transformer for Image Segmentation (TITS 2024)


The repository contains official Pytorch implementations of training and evaluation codes and pre-trained models for **SDPT**. 

The code is based on [MMSegmentaion v0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1).


## Citation
If you find our repo useful for your research, please consider citing our paper:

```
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

```


**Notes**: In this scheme, The number of FLOPs (G) is calculated on the input size of 512 $\times$ 512 for ADE20K, 2048 $\times$ 1024 for Cityscapes by [torchprofile](https://github.com/zhijian-liu/torchprofile) (recommended, highly accurate and automatic MACs/FLOPs statistics).



## Installation
Install the dependencies and download ADE20K according to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/v0.24.1/docs/en/get_started.md#installation).


```
pip install timm
cd segmentation
python setup.py develop
```

## Training

We use 8 GPUs for training by default. Run:

```bash
./tools/dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the model, run:

```bash
./tools/dist_test.sh /path/to/config /path/to/checkpoint_file 8 --eval mIoU
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/get_flops.py /path/to/config --shape 512 512
```

## Contact

For technical problem, please create an issue.

If you have any private question, please feel free to contact me via hu.cao@tum.de.


## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1) and [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
