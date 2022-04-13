# [SCAN: Cross-domain Object Detection with Semantic Conditioned Adaptation (AAAI22 ORAL)](https://www.aaai.org/AAAI22Papers/AAAI-902.LiW.pdf)

By [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)


[2022/03/08/] Welcome to follow our new work [SIGMA](https://github.com/CityU-AIM-Group/SIGMA), which is a comprehensive upgrade of this work (SCAN).


## Installation

Check [INSTALL.md](https://github.com/CityU-AIM-Group/SCAN/blob/main/INSTALL.md) for installation instructions.

## Data preparation

Step 1: Format three benchmark datasets.

```
[DATASET_PATH]
└─ Cityscapes
   └─ cocoAnnotations
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
└─ KITTI
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ Sim10k
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
```


Step 2: change the data root for your dataset at [paths_catalog.py](https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/config/paths_catalog.py).

```
DATA_DIR = [$Your dataset root]
```

More detailed dataset preparation can be found at [EPM](https://github.com/chengchunhsu/EveryPixelMatters).


## Tutorials for this project
We present basic instructions about our main modification to understand our codes better.
1. Middle_head: [congraph](https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/condgraph.py)
    - We design a "middle head" between the feature extractor and detection head for different DA operations on feature maps.
    - We give lots of APIs for further research, including different kinds of graphs, manifestation modules, paradigms, and semantic transfer settings, and you can use them by changing the config file directly, (more details are shown in 'fcos_core/config/default.py')

2. Node generation: [here](https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/loss.py)
    - We sample graph nodes with ground-truth in the source domain and use DBSCAN to sample target domain nodes.
    - We have tried different clustering algorithms for target node sampling and preserving the APIs.

3. An interesting inference strategy [here](https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/fcos.py)
    - We find that ensembling the semantic maps (the outputs of semantic conditioned kernels) and the classification maps can achieve a higher result (C2F: 42.3 to 42.8). You can have a try by changing the TEST.MODE from 'common' to 'precision' in the config file. 
    - Besides, only using the semantic maps can achieve a comparable result with the standard 4-Conv detection head and reduce computation costs (TEST.MODE =' light'). Kindly note that we still use the 'common' mode for a fair comparison with other methods.

4. CKA module is implemented [here](https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/discriminator/fcos_head_discriminator_con.py)

5. DEBUGGGG
      - We also preserve may debug APIs to save different maps for a better understanding of our works.

## Well-trained models 
We provide the experimental results and model weights in this section ([onedrive line](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/Eso9N-h_saNOt35J7taAEokB23_M6VjXn4xFW9wMP3kR0A?e=Bblcnh)). Kindly note that it is easy to get higher results than the reported ones with tailor-tuned hyperparameters.

| dataset | backbone | mAP	 | mAP@50 |  mAP@75 |	 
| :-----| :----: | :----: |:-----:| :----: | 
| Cityscapes -> Foggy Cityscapes | VGG16 | 23.0 |42.3|21.2|
| Sim10k -> Cityscapes | VGG16 | 27.4 |53.0 |27.4 |
| KITTI -> Cityscapes | VGG16 | 23.0 |46.3 |20.9 |


## Get start 

Train from the scratch:
(Use VGG-16 as the backbone with 1 GPU. Our code doesn't support distributed training now and only supports single-GPU training.)

```
python tools/train_net_da.py \
        --config-file configs/scan/xxx.yaml

```

Test with the well trained models:

```
python tools/test_net.py \
        --config-file configs/scan/xxx.yaml \
        MODEL.WEIGHT xxx.pth

```

 
## Citation 

If you think this work is helpful for your project, please give it a star and citation:
```
@inproceedings{li2022scan,
  title={SCAN: Cross Domain Object Detection with Semantic Conditioned Adaptation},
  author={Li, Wuyang and Liu, Xinyu and Yao, Xiwen and Yuan, Yixuan},
  booktitle={36th AAAI Conference on Artificial Intelligence (AAAI-22)},
  year={2022}
}
```

## Acknowledgements

This work is based on the EveryPixelMatter (ECCV20) [EPM](https://github.com/chengchunhsu/EveryPixelMatters). 

The implementation of the detector is heavily based on [FCOS](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f).

 
## Abstract

The domain gap severely limits the transferability and scalability of object detectors trained in a specific domain when applied to a novel one. Most existing works bridge the domain gap through minimizing the domain discrepancy in the category space and aligning category-agnostic global features. Though great success, these methods model domain discrepancy with prototypes within a batch, yielding a biased estimation of domain-level distribution. Besides, the category-agnostic alignment leads to the disagreement of class-specific distributions in the two domains, further causing inevitable classification errors. To overcome these two challenges, we propose a novel Semantic Conditioned AdaptatioN (SCAN) framework such that well-modeled unbiased semantics can support semantic conditioned adaptation for precise domain adaptive object detection. Specifically, class-specific semantics crossing different images in the source domain are graphically aggregated as the input to learn an unbiased semantic paradigm incrementally. The paradigm is then sent to a lightweight manifestation module to obtain conditional kernels to serve as the role of extracting semantics from the target domain for better adaptation. Subsequently, conditional kernels are integrated into global alignment to support the class-specific adaptation in a designed Conditional Kernel guided Alignment (CKA) module. Meanwhile, rich knowledge of the unbiased paradigm is transferred to the target domain with a novel Graph-based Semantic Transfer (GST) mechanism, yielding the adaptation in the category-based feature space. Comprehensive experiments conducted on three adaptation benchmarks demonstrate that SCAN outperforms existing works by a large margin.

![image](https://github.com/CityU-AIM-Group/SCAN/blob/main/overall.png)
## Contact 

If you have any problems, please feel free to contact me at wuyangli2-c@my.cityu.edu.hk. Thanks.

