#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, shutil, sys
# import voc_eval_polygon_sympy

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root = './'
lib_path = os.path.join(root, 'lib')
add_path(lib_path)

from voc_eval_polygon import voc_eval_polygon

detpath = 'tools/ctw1500_evaluation/detections_{:s}.txt'

annopath  = 'data/ctw1500/test/test_label_curve.txt'
imagesetfile = 'data/ctw1500/test/test.txt'

score_thresh_list=[0.5]
for isocre in score_thresh_list:
    rec, prec, ap = voc_eval_polygon(detpath[:-4]+str(isocre)+'.txt', annopath, imagesetfile, 'text', ovthresh=0.5)

file = 'tools/ctw1500_evaluation/txt_pr.txt'
_ = lambda x,y: 2*x*y*1.0/(x+y)
with open(file, 'w') as f:
    f.write('ap     rec    prec   f-measure\n')
    for i in range(len(rec)):
        f.write('{:.4f} {:.4f} {:.4f} {:.4f}\n'.format(ap, rec[i], prec[i], _(rec[i], prec[i])))
    print('ap: {:.4f}, recall: {:.4f}, pred: {:.4f}, FM: {:.4f}\n'.format(ap, rec[-1], prec[-1], _(rec[-1], prec[-1])))