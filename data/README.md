# Description

The SCUT-CTW1500 dataset can be downloaded through the following link:

(https://pan.baidu.com/s/1eSvpq7o PASSWORD: fatf)

or (https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk) (OneDrive)

unzip the file in ROOT/data/ 

## Dataset Information

There are two file (train) and (test) inside ROOT/data/ctw1500/. Each one has four files.

a) **text_image** contains images files.

b) **text_label_curve** cotains annotation files with suffix ".txt"
* The file name is correspoinding to image file.
* Each line of the each file represents one text region.
* Each line has 32 values, representing xmin, ymin, xmax, ymax (of the circumsribed rectangle), pw1, ph1, ..., pw14, ph14.
* For Rectangle or Quadrangle bounding box, the extra 20 offsets are automatically created, so every bounding box has 32 values.

c) **trainval.txt** or **test.txt** are the list of the image file. 

d) **trainval_label_curve.txt** or **test_label_curve.txt** are the list of the label file.

## Labeling tool

Below shows how we label the data.

<img src="labeling.gif" width="50%">

The labeling tool and manual can be downloaded through the following links:

Ubuntu label tool: https://1drv.ms/u/s!AradqGvJ8Ebta94HFxxGrtavTUo

Windows label tool: Coming soon.

Manual: https://1drv.ms/b/s!AradqGvJ8EbtahczW759VekS4lg 

Note:
1. The SCUT-CTW1500 dataset can be used only for non-commercial research purpose.