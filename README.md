# Curve-Text-Detector

Released on December 06, 2017

# Description

Curve text are very common in our real-world. For examples, text in most kinds of columnar objects (bottles, stone piles, etc.), spherical objects, plicated plane (clothes, streamer, etc.), coins, logos, signboard and so on. Current datasets have very little curve text, and it is defective to label such text with quadrangle let alone rectangle. Curve bounding box has three remarkable advantages:
<div align=center><img src="images/1-1.jpg" width="50%" ></div>
<div align=center><img src="images/1-2.jpg" width="50%" ></div>
<div align=center><img src="images/1-3.jpg" width="50%"></div>

* Avoid needless overlap

* Less background noise

* Avoid multiple text lines

We introduce our dataset and our CTD method at our paper [(https://arxiv.org/abs/1712.02170)](https://arxiv.org/abs/1712.02170).

# Clone the Curve-Text-Detector repository

Clone the Curve-Text-Detector repository
  ```Shell
  git clone https://github.com/Yuliang-Liu/Curve-Text-Detector.git --recursive
  ```

# Getting Started
## Dataset

<img src="images/annotation.jpg" width="100%">

The SCUT-CTW1500 dataset can be downloaded through the following link:

(https://pan.baidu.com/s/1eSvpq7o PASSWORD: fatf) (Size = 842Mb).

unzip the file in ROOT/data/ 

### Dataset Information

a) Train - It contains 1000 images.

b) Test - It contains 500 images.

c) Each image contains at least 1 curve text.

## Pre-trained model and our trained model

We use resnet-50 model as our pre-trained model, which can be download through the following link:

(https://pan.baidu.com/s/1eSJBL5K PASSWORD: mcic) (Size = 102Mb).

put model in ROOT/data/imagenet_models/

Our model trained with SCUT-CTW1500 training set can be download through the following link:

(https://pan.baidu.com/s/1gfs5vH5 PASSWORD: 1700) (Size = 114Mb).

put model in ROOT/output/

* [test.sh](./test.py) After downloading the dataset and our ctd_tloc.caffemodel, simply run the this file can evaluate our method on the SCUT-CTW1500 test set. Uncommend --vis to visualize the detecting results.

* [my_train.sh](./my_train.sh) This file shows how to train on the SCUT-CTW1500 dataset. After downloading the dataset and resnet-50 pre-trained model, simply running my_train.sh to start. 

Both train and test require less than 4GB video memory.

# Detecting Results 
<img src="images/table.png" width="100%">
<img src="images/detect_results.png" width="100%">

More details can be found on our paper [(https://arxiv.org/abs/1712.02170)](https://arxiv.org/abs/1712.02170).

# Citation
If you find our method or the dataset useful for your research, please cite 
```
@article{liu2017curve,
  title     = {Detecting Curve Text in the Wild: New Dataset and New Solution},
  author    = {Liu, Yuliang and Jin, Lianwen and Zhang Shuaitao and Zhang Sheng},
  jurnal    = {arXiv preprint arXiv:1712.02170},
  year      = {2017}
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