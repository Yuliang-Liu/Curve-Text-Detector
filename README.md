# Curve-Text-Detector

(Released on December 06, 2017)

# News

* We annotate point for each English character and correct some recognition annotations. New annotaions can be found [here](https://github.com/Yuliang-Liu/Curve-Text-Detector/tree/master/data).

* End-to-end annotations of SCUT-CTW1500 have been updated (see data/README.md). Note that the new annotations have a little difference from original detection annotations, and thus the end-to-end annotations are only used for end-to-end evaluation purpose. 

* SCUT-CTW1500 is a text-line based dataset with both English and Chinese instances. If you are insterested in word-level based Engish curve text, we highly recommend you refer to [Total-text](https://github.com/cs-chan/Total-Text-Dataset). In addition, a recent ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text ([ArT](http://rrc.cvc.uab.es/?ch=14)), which is extended from SCUT-CTW1500 and Total-text, is held for stimulating more improvements on the arbitrary-shaped text reading task. The competition results of ArT can be found on [ICDAR2019-ArT](https://rrc.cvc.uab.es/files/ICDAR2019-ArT1.pdf).

* Total-Text and SCUT-CTW1500 are now part of the training set of the largest curved text dataset - [ArT](http://rrc.cvc.uab.es/?ch=14) (Arbitrary-Shaped Text dataset). In order to retain the validity of future benchmarking on both mentioned datasets, the **test-set images** of CTW1500 should be removed (with the corresponding ID provided in CTW1500_ID_vs_ArT_ID.txt) from the ArT dataset shall one intend to leverage the extra training data from the ArT dataset. We count on the trust of the research community to perform such removal operation to attain the fairness of the benchmarking.

# SCUT-CTW1500 Detection Leaderboard

| Method     |  Recall (%)  |  Precision (%)  |   Hmean (%)     |    Publication   | [TIoU-Hmean](https://github.com/Yuliang-Liu/TIoU-metric/tree/master/curved-tiou) (%) | FPS |
|:--------:  | :-----:   | :----:      |  :-----:     | :-----: | :-----: | :-----: |
|Proposed CTD [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664)     | 65.2     |  74.3       |    69.5      |   PR 2019 |   - |
|Proposed CTD+TLOC [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320319300664) | 69.8     |  74.3       |    73.4      |   PR 2019 |  47.5 | 13.3 |
|SLPR [[paper]](https://arxiv.org/abs/1801.09969)             | 70.1     |  80.1       |    74.8      |   arXiv 1801 |   - |
|TextSnake [[paper]](https://arxiv.org/abs/1807.01544)[[code]](https://github.com/princewang1994/TextSnake.pytorch)        | 85.3     |  67.9       |    75.6     |   ECCV 2018   |-|
|Qin et al. [[paper]](https://arxiv.org/pdf/1908.09990.pdf)| 78.2     |  73.8       |    76.0      |   ICDAR 2019 | -|
|CSE [[paper]](https://arxiv.org/abs/1903.08836?context=cs.CV)| 76.1     |  78.7       |    77.4      |   CVPR 2019 | -|
|LOMO [[paper]](https://arxiv.org/abs/1904.06535) <br > LOMO MS| 69.6<br >76.5     |  89.2<br >85.7       |    78.4<br >80.8      |   CVPR 2019 |-|
|SAE [[paper]](http://jiaya.me/papers/textdetection_cvpr19.pdf)| 77.8     |  82.7       |    80.1      |   CVPR 2019 | -|
|ATRR [[paper]](https://arxiv.org/pdf/1905.05980.pdf)| 80.2     |  80.1       |    80.1      |   CVPR 2019 | 58.0 |
|AGBL [[paper]](https://link.springer.com/article/10.1007/s11432-019-2673-8)| 76.6    |  83.9      |    80.1      |   SCIS 1912 | - |
|NASK [[paper]](https://arxiv.org/pdf/2004.12436.pdf)| 78.3    |  82.8      |    80.5      |   ICASSP 2020 | - | 12|
|LSN+CC [[paper]](https://arxiv.org/abs/1903.09837)| 78.8     |  83.2       |    80.8      |   arXiv 1903 |60.0|
|SAST [[paper]](http://delivery.acm.org/10.1145/3360000/3350988/p1277-wang.pdf?ip=129.127.145.232&id=3350988&acc=OA&key=65D80644F295BC0D%2E001A23AA3BABC648%2E4D4702B0C3E38B35%2E166E25130A976506&__acm__=1573107294_6147cbbe22189ecb67c0d1f1be453062)| 77.1     |  85.3       |    81.0      |   ACM MM 2019 |- | 27.6|
|ICG [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302511)| 79.8     |  82.8       |    81.3      |   PR 2019 | - |
|TextField [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8648420)[[code]](https://github.com/YukangWang/TextField)| 79.8     |  83.0      |    81.4      |   TIP 2019 | 61.4 |
|ABCNet [[paper]](https://arxiv.org/abs/2002.10200)[[code]](https://github.com/aim-uofa/AdelaiDet) |  79.1  |   83.8     |    81.4      |  CVPR2020 | - | 9.5 |
|MSR [[paper]](https://arxiv.org/abs/1901.02596)| 79.0    |  84.1      |    81.5      |   IJCAI 2019 | 61.3 |
|PSENet-1s [[paper]](https://arxiv.org/abs/1903.12473)[[code]](https://github.com/whai362/PSENet)  | 79.7    |  84.8      |    82.2     |  CVPR 2019 |60.6| 3.9 |
|TextMountain [[paper]](https://arxiv.org/pdf/1811.12786.pdf)  | 83.4  |  82.9       |    83.2     |  arXiv 1811 |64.2|
|Relation [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8977995)  | 80.9  |  85.8       |    83.3     |  ICDAR 2019 | |
|DB-ResNet-50 [[paper]](https://arxiv.org/pdf/1911.08947.pdf)[[code]](https://github.com/MhLiao/DB) | 80.2 | 86.9 | 83.4 | AAAI 2020 | - | 22 |
|CRAFT [[paper]](https://arxiv.org/abs/1904.01941)[[code]](https://github.com/clovaai/CRAFT-pytorch) | 81.1 | 86.0 | 83.5 | CVPR 2019 |61.0|
|TextDragon [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf) | 82.8 | 84.5 | 83.6 | ICCV 2019 | - |
|TextTubes [[paper]](https://arxiv.org/pdf/1912.08990.pdf) | 80.0 | 87.65 | 83.65 | arXiv 1912 | - |
|PSENet_v2 [[paper]](https://arxiv.org/pdf/1908.05900v1.pdf)[[unofficial code]](https://github.com/WenmuZhou/PAN.pytorch) | 81.2 | 86.4 | 83.7 | ICCV 2019 | - | 39.8 |
|ContourNet [[paper]](https://arxiv.org/abs/2004.04940)[[code]](https://github.com/wangyuxin87/ContourNet) | 84.1 | 83.7 | 83.9 | CVPR 2020 | - | 4.5 |
|SA-TEXT MS [[paper]](https://arxiv.org/pdf/1911.07046.pdf) | 85.4 | 83.3 | 84.4 | arXiv 1911 | - | - |
|PuzzleNet [[paper]](https://arxiv.org/abs/2002.11371) | 84.7 | 84.1 | 84.4 | arXiv 2002 | - | - |
|PAN Mask R-CNN [[paper]](https://arxiv.org/pdf/1811.09058.pdf)| 83.2    |  86.8       |    85.0     |  WACV 2019 |65.2|
|TextPerception [[paper]](https://arxiv.org/pdf/2002.06820.pdf)| 81.8    |  88.8       |    85.2     |  AAAI 2020 |-|
|TextCohesion [[paper]](https://arxiv.org/pdf/1904.12640.pdf)| 84.7    |  88.0       |    86.3     |  arXiv 1904 |-|


*Note that training data and backbone of different methods may not be the same, and thus the comparison is not strictly fair. 

# SCUT-CTW1500 End-to-end Leaderboard

| Method     | Dataset | E2E-Hmean (%) | Wordspotting-Hmean (%)  |   Publication   |  
|:--------:  | :-----: | :-----:      |  :-----:        | :-----:          | 
|TextDragon [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf) | SynText800k + CTW1500     | 39.7     |  -   |    ICCV 2019      | 
|TextPerception [[paper]](https://arxiv.org/pdf/2002.06820.pdf) |SynText800k + CTW1500    | -  |  57.0   |    AAAI 2020      |
|ABCNet [[paper]](https://arxiv.org/abs/2002.10200)[[code]](https://github.com/aim-uofa/AdelaiDet) |SynText150k + CTW1500   | 45.2  |  -  |  CVPR 2020  |

*[e2e evaluation script](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BAText/README.md)

# Evaluation Script

We provide a brief evaluation script for researches to evaluate their own methods on the CTW1500 dataset. The instruction and details are given in tools/ctw1500_evaluation/Readme.md. An easier way is to use [TIoU curved text evaluation script](https://github.com/Yuliang-Liu/TIoU-metric/tree/master/curved-tiou) (The origin result of TIoU script is the same as the result from this evaluation scipt).


# Clone the Curve-Text-Detector repository

Clone the Curve-Text-Detector repository
  ```Shell
  git clone https://github.com/Yuliang-Liu/Curve-Text-Detector.git --recursive
  ```

# Getting Started
## Dataset

<img src="images/annotation.jpg" width="100%">

The SCUT-CTW1500 dataset can be downloaded through the following link:

(https://pan.baidu.com/s/1eSvpq7o PASSWORD: fatf) (BaiduYun. Size = 842Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk) (OneDrive)

unzip the file in ROOT/data/ 


### Dataset Information

a) Train/ - 1000 images.

b) Test/ - 500 images.

c) Each image contains at least 1 curved text.

The visualization of the annotated images can be downloaded through the following link:

(https://pan.baidu.com/s/1eR641zG PASSWORD: 5xei) (BaiduYun. Size = 696 Mb).

## Pre-trained model and our trained model

We use resnet-50 model as our pre-trained model, which can be download through the following link:

(https://pan.baidu.com/s/1eSJBL5K PASSWORD: mcic) (Baidu Yun. Size = 102Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilHwMsW2N_bfnb0Bx) (OneDrive)

put model in ROOT/data/imagenet_models/

Our model trained with SCUT-CTW1500 training set can be download through the following link:

(https://pan.baidu.com/s/1gfs5vH5 PASSWORD: 1700) (BaiduYun. Size = 114Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilH0rLDFrRof8qmRD) (OneDrive)

put model in ROOT/output/

* [test.sh](./test.py) Downloading the dataset and our ctd_tloc.caffemodel, and running this file to evaluate our method on the SCUT-CTW1500 test set. Uncommend --vis to visualize the detecting results.

* [my_train.sh](./my_train.sh) This file shows how to train on the SCUT-CTW1500 dataset. Downloading the dataset and resnet-50 pre-trained model, and running my_train.sh to start training. 

Both train and test require less than 4 GB video memory.

* [demo.py](./tools/demo.py) (cd tools/) then (python demo.py). This file easily shows how to test other images. With provided model, it can produce like

<div align=center><img src="images/demo_result.png" width="50%" ></div>

# Comparing smooth effect by TLOC 
Train and test files are put under (model/ctd/smooth_effect/), and both the training and testing procedures are the same as above.

To visulize the ctd+tloc, simply uncomment ctd in the last of the test.prototxt, vice versa. Below are the first three images in our test set: 

<table><tr>
    <td><img src="images/s1.png" width="240" height="180" border=0></td>
    <td><img src="images/s2.png" width="240" height="180" border=0></td>
    <td><img src="images/s3.png" width="240" height="180" border=0></td>
</tr></table>

If you are insterested in it, you can train your own model to test. Because training doesn't require so much time, we don't upload our own model (Of course, you can email me for it). 


# Labeling tool 
  For the labeling tool and specific details of the gound truths, please refer to data/README.md. 

# Citation
If you find our method or the dataset useful for your research, please cite 
```
@article{liu2019curved,
  title={Curved scene text detection via transverse and longitudinal sequence connection},
  author={Liu, Yuliang and Jin, Lianwen and Zhang, Shuaitao and Luo, Canjie and Zhang, Sheng},
  journal={Pattern Recognition},
  volume={90},
  pages={337--345},
  year={2019},
  publisher={Elsevier}
}
```

# Requirement 
1. Clone this repository. ROOT is the directory where you clone.
2. cd ROOT/caffe/  and use your own Makefile.config to compile (make all && make pycaffe). If you are using ubuntu 14.04, you may need to modify Makefile line 181 (hdf5_serial_hl hdf5_serial) to (hdf5 hdf5_hl).
3. cd ROOT/lib make (based on python2)
4. pip install shapely. (Enable computing polygon intersection.)

## Feedback
Suggestions and opinions of this dataset (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`liu.yuliang@mail.scut.edu.cn`.

## Copyright
The SCUT-CTW1500 database is free to the academic community for research purpose usage only.

For commercial purpose usage, please contact Dr. Lianwen Jin: [eelwjin@scut.edu.cn](eelwjin@scut.edu.cn)

Copyright 2017, Deep Learning and Vision Computing Lab, South China China University of Technology. [http://www.dlvc-lab.net](http://www.dlvc-lab.net)
