# Curve-Text-Detector

Released on December 06, 2017

# Description

This method is aimed for curve text detection. 

# Clone the caffe-model repository

Clone the caffe-model repository
  ```Shell
  git clone https://github.com/Yuliang-Liu/Curve-Text-Detector.git --recursive
  ```

# Getting Started
## Dataset

The CTW1500 dataset can be downloaded through the following link:

(https://pan.baidu.com/s/1eSvpq7o PASSWORD: fatf) (Size = 836Mb).

unzip the file in ROOT/data/ 

### Dataset Information

a) Train - It contains 1000 images.

b) Test - It contains 500 images.

Note:
1. The CTW1500 dataset can be used only for non-commercial research purpose.

## Pre-trained model and our trained model

We use resnet-50 model as our pre-trained model, which can be download through the following link:

(https://pan.baidu.com/s/1eSJBL5K PASSWORD: mcic) (Size = 102Mb).

put model in ROOT/data/imagenet_models/

Our model trained with CTW1500 training set can be download through the following link:

(https://pan.baidu.com/s/1gfs5vH5 PASSWORD: 1700) (Size = 114Mb).

put model in ROOT/output/

* [test.sh](./test.py) After downloading the dataset and our ctd_tloc.caffemodel, simply run the this file can evaluate our method on the CTW1500 test set. Uncommend --vis to visualize the detecting results.

* [my_train.sh](./my_train.sh) This file shows how to train on the CTW1500 dataset. After downloading the dataset and resnet-50 pre-trained model, simply running my_train.sh to start.

## Citation
If you find our method or the dataset useful for your research, please cite

## Requirement 
1. Clone this repository. ROOT is the directory where you clone.
2. cd ROOT/caffe/  and use your own Makefile.config to compile. 
3. cd ROOT/lib make (based on python2)
4. pip install shapely. (Enable computing polygon intersection.)

    
## Feedback
Suggestions and opinions of this dataset (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`liu.yuliang@mail.scut.edu.cn`.

## acknowledgement