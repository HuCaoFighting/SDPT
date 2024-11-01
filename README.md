## [TITS2024] SDPT: Semantic-Aware Dimension-Pooling Transformer for Image Segmentation

This is the official repository for SDPT. Our trained model is stored on the Huawei internal source. We do not have the right to send any files out from the internal system, so we can't share our trained model weights. I rewrote the codes and provided our log for your reference.

Related links:
[[Official PDF Download]](https://ieeexplore.ieee.org/document/10584449)

### Requirements:

* torch>=1.7+
* torchvision>=0.7.0+
* timm>=0.3.2

### Introduction

In this work, we present the semantic-aware dimension-pooling transformer (SDPT) to mitigate the conflict between accuracy and efficiency. The proposed model comprises an efficient transformer encoder for generating hierarchical
features and a semantic-balanced decoder for predicting semantic masks. In the encoder, a dimension-pooling mechanism is used in the multi-head self-attention (MHSA) to reduce the computational cost, and a parallel depth-wise convolution is used
to capture local semantics. Simultaneously, we further apply this dimension-pooling attention (DPA) to the decoder as a refinement module to integrate multi-level features. With such a simple yet powerful encoder-decoder framework, we empirically
demonstrate that the proposed SDPT achieves excellent performance and efficiency on various popular benchmarks.


### Train

Use the following commands to train `SDPT-Small` for distributed learning with 8 GPUs:

````bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=$((RANDOM+10000)) --use_env main.py --data-path ${YOUR_DATA_PATH} --batch-size 128 --model SDPT_small --drop-path 0.1
# model names: --model SDPT_tiny/SDPT_small/SDPT_base
# with --drop-path 0.1/0.1/0.3/0.3
# replace ${YOUR_DATA_PATH} with your data path that contains train/ val/ directory
````

### Validate the performance

Download the pretrained weights to `pretrained` directory first. Then use the following commands to validate the performance:

````bash
python main.py --eval --resume pretrained/SDPT_small.pth --model SDPT_small
````

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

### Other Notes

If you meet any problems, please do not hesitate to contact us.
Issues and discussions are welcome in the repository!
You can also contact us via sending messages to this email: hu.cao@tum.de


### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Non-Commercial use only. Any commercial use should get formal permission first.

