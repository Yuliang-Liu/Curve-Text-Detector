
# SCUT-CTW1500 Datasets 
**We have updated annotations for both train and test set.** 

Train: 1000 images [[images]](https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip)[[annos]](https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip) 

Additional point annotation for each character is included. Example can be referred to [here](https://github.com/Yuliang-Liu/Curve-Text-Detector/tree/master/data).

```
wget -O train_images.zip https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip
wget -O train_labels.zip https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip
```

Test: 500 images [[images]](https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip)[[annos]](https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5) 
```
wget -O test_images.zip https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip
wget -O test_labels.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download
```

*Note all Chinese texts are annotated with '###' (ignore) in this updated version, because the number of Chinese is too small for both training and testing purpose. [ArT](http://rrc.cvc.uab.es/?ch=14) and [LSVT](https://rrc.cvc.uab.es/?ch=16) two optional large-scale arbitrarily-shaped text benchmarks for both Chinese and English.* 

# SCUT-CTW1500 Evaluation
Original detection only [evaluation script](https://github.com/Yuliang-Liu/TIoU-metric/tree/master/curved-tiou). 

For both detection and end-to-end evaluation in the updated version, please refer to [here](https://universityofadelaide.box.com/shared/static/ys234cg1rtgke051hu33lbm5ri0bvxr0.zip). This scipt also includes evaluation example for [Total-text](https://github.com/cs-chan/Total-Text-Dataset).


# Info
The project is outdated and will not be maintained anymore. Original info is kept in [OLD_README.md](https://github.com/Yuliang-Liu/Curve-Text-Detector/tree/master/OLD_README.md).


## Copyright
The SCUT-CTW1500 database is free to the academic community for research only.

For other purpose, please contact Dr. Lianwen Jin: [eelwjin@scut.edu.cn](eelwjin@scut.edu.cn)
