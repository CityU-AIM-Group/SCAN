# [SCAN++: Enhanced Semantic Conditioned Adaptation for Domain Adaptive Object Detection (TMM)](https://ieeexplore.ieee.org/document/9931144)

By [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)


[2022/03/08/] Welcome to follow our new work [SIGMA](https://github.com/CityU-AIM-Group/SIGMA), which is a comprehensive upgrade of this work (SCAN).

[2022/10/21/] The journal-extended version SCAN++ has been released.
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
      - We also preserve may debug APIs to save different maps.

<!-- ## Well-trained models 
We provide the experimental results and model weights in this section ([onedrive line](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/Eso9N-h_saNOt35J7taAEokB23_M6VjXn4xFW9wMP3kR0A?e=Bblcnh)). Kindly note that it is easy to get higher results than the reported ones with tailor-tuned hyperparameters.

| dataset | backbone | mAP	 | mAP@50 |  mAP@75 |	 
| :-----| :----: | :----: |:-----:| :----: | 
| Cityscapes -> Foggy Cityscapes | VGG16 | 23.0 |42.3|21.2|
| Sim10k -> Cityscapes | VGG16 | 27.4 |53.0 |27.4 |
| KITTI -> Cityscapes | VGG16 | 23.0 |46.3 |20.9 | -->


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

@ARTICLE{9931144,
  author={Li, Wuyang and Liu, Xinyu and Yuan, Yixuan},
  journal={IEEE Transactions on Multimedia}, 
  title={SCAN++: Enhanced Semantic Conditioned Adaptation for Domain Adaptive Object Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TMM.2022.3217388}}


```

## Acknowledgements

This work is based on the EveryPixelMatter (ECCV20) [EPM](https://github.com/chengchunhsu/EveryPixelMatters). 

The implementation of the detector is heavily based on [FCOS](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f).

 
## Abstract
Domain Adaptive Object Detection (DAOD) transfers an object detector from the labeled source domain to a novel unlabelled target domain. Recent advances bridge the domain gap by aligning category-agnostic feature distribution and minimizing the domain discrepancy for adapting semantic distribution. Though great success, these methods model domain discrepancy with prototypes within a batch, yielding a biased estimation of domain-level statistics. Moreover, the category-agnostic alignment leads to the disagreement of the cross-domain semantic distribution with inevitable classification errors. To address these two issues, we propose an enhanced Semantic Conditioned AdaptatioN (SCAN++) framework, which leverages unbiased semantics for DAOD. Specifically, in the source domain, we design the conditional kernel to sample Pixel of Interests (PoIs), and aggregate PoIs with a cross-image graph to estimate an unbiased semantic sequence. Conditioned on the semantic sequence, we further update the parameter of the conditional kernel in a semantic conditioned manifestation module, and establish a novel conditional graph in the target domain to model unlabeled semantics. After modeling the semantic distribution in both domains, we integrate the conditional kernel into adversarial alignment to achieve semantic-aware adaptation in a Conditional Kernel guided Alignment (CKA) module. Meanwhile, the Semantic Sequence guided Transport (SST) module is proposed to transfer reliable semantic knowledge to the target domain through solving the cross-domain Optimal Transport (OT) assignment, achieving unbiased adaptation at the semantic level. Comprehensive experiments on four adaptation scenarios demonstrate that SCAN++ achieves state-of-the-art results. 

![image](https://github.com/CityU-AIM-Group/SCAN/blob/SCAN%2B%2B/SCANv2.png)

## Contact 

If you have any problems, please feel free to contact me at wuyangli2-c@my.cityu.edu.hk. Thanks.

