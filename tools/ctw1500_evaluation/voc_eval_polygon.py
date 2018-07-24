# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# Modified by yl
# --------------------------------------------------------

import os
import cPickle
import numpy as np

from shapely.geometry import *

def parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[0]),
                                  int(cors[1]),
                                  int(cors[2]),
                                  int(cors[3])]
            objects.append(obj_struct)
    return objects

def curve_parse_rec_txt(filename):
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            obj_struct = {}
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[0]), int(cors[1]),int(cors[2]),int(cors[3]),
                                  int(cors[4]), int(cors[5]),int(cors[6]),int(cors[7]),
                                  int(cors[8]), int(cors[9]),int(cors[10]),int(cors[11]),
                                  int(cors[12]), int(cors[13]),int(cors[14]),int(cors[15]),int(cors[16]), int(cors[17]),int(cors[18]),int(cors[19]),int(cors[20]), int(cors[21]),
                                  int(cors[22]), int(cors[23]),int(cors[24]),int(cors[25]),int(cors[26]), int(cors[27]),int(cors[28]),int(cors[29]),int(cors[30]), int(cors[31])]
            objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_polygon(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):

    # first load gt
    cachefile = 'tools/ctw1500_evaluation/annots.pkl'
    # read list of images
    with open(imagesetfile, 'r') as f, open(annopath, 'r') as fa:
        lines = f.readlines()
        anno_lines = fa.readlines()
    imagenames = [x.strip() for x in lines]
    anno_names = [y.strip() for y in anno_lines]
    assert(len(imagenames) == len(anno_names)), 'each image should correspond to one label file'

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            print(anno_names[i].strip())
            recs[imagename] = curve_parse_rec_txt(anno_names[i])
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    class_recs = {}
    npos = 0
    for ix, imagename in enumerate(imagenames):
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # text 
        # assert(R), 'Can not find any object in '+ classname+' class.'
        if not R: continue
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[str(ix)] = {'bbox': bbox,
                                'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d] # mask rcnn
        det_bbox = bb[:]
        pts = [(det_bbox[j], det_bbox[j+1]) for j in xrange(0,len(bb),2)]
        try:
            pdet = Polygon(pts)
        except Exception as e:
            print(e)
            continue
        if not pdet.is_valid: 
            print('predicted polygon has intersecting sides.')
            # print(pts, image_ids[d])
            continue

        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        gt_bbox = BBGT[:, :4]
        info_bbox_gt = BBGT[:, 4:32]
        ls_pgt = [] 
        overlaps = np.zeros(BBGT.shape[0])
        for iix in xrange(BBGT.shape[0]):
            pts = [(int(gt_bbox[iix, 0]) + info_bbox_gt[iix, j], int(gt_bbox[iix, 1]) + info_bbox_gt[iix, j+1]) for j in xrange(0,28,2)]
            pgt = Polygon(pts)
            if not pgt.is_valid: 
                print('GT polygon has intersecting sides.')
                continue
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                print('intersect invalid',e)
                continue
            try:
                assert(sec.is_valid), 'polygon has intersection sides.' # for mask rcnn
            except Exception as e:
                print(e)
                continue
            inters = sec.area
            uni = pgt.area + pdet.area - inters
            if uni <= 0.00001: uni = 0.00001
            overlaps[iix] = inters*1.0 / uni
            
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap