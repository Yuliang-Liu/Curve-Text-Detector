import numpy as np
import cv2
import os, glob

import _init_paths
import caffe
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, info_syn_transform_inv_h, info_syn_transform_inv_w
from fast_rcnn.nms_wrapper import nms, pnms
from utils.blob import im_list_to_blob

from shapely.geometry import *

caffe.set_mode_gpu()
caffe.set_device(0)

net_prototxt = "../models/ctd/test_ctd_tloc.prototxt" 
model = "../output/ctd_tloc.caffemodel" 
cofig_file = "../experiments/cfgs/rfcn_ctd.yml"
images = glob.glob("../images/demo/*.jpg")


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    blobs, im_scales = _get_blobs(im, boxes)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]

    scores = blobs_out['cls_prob']

    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    ############################################### curve
    info_deltas_h = blobs_out['info_pred_h']
    pred_infos_h = info_syn_transform_inv_h(boxes, info_deltas_h)
    info_deltas_w = blobs_out['info_pred_w']
    pred_infos_w = info_syn_transform_inv_w(boxes, info_deltas_w)
    assert len(boxes) == len(pred_infos_h) == len(pred_infos_w)
    ###############################################

    return scores, pred_boxes, pred_infos_h, pred_infos_w

def vis(im, dets, thresh=0.3):
    for i in xrange(np.minimum(100, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, 4]
        info_bbox = dets[i, 5:33] # syn
        pts = [info_bbox[i] for i in xrange(28)]
        assert(len(pts) == 28), 'wrong length.'
        if score > thresh:
            for p in xrange(0,28,2):
                cv2.line(im,(int(bbox[0]) + int(pts[p%28]), int(bbox[1]) + int(pts[(p+1)%28])), 
                             (int(bbox[0]) + int(pts[(p+2)%28]), int(bbox[1]) + int(pts[(p+3)%28])),(0,0,255),2) 

    im = cv2.resize(im, (1280, 720)) # visualization
    cv2.imshow('Dectecting results syn.', im)
    cv2.waitKey(0)

def nps(dets, cdets):
    delete_inds = []
    for i in xrange(cdets.shape[0]):
        bbox = cdets[i, :4]
        score = cdets[i, 4]
        info_bbox = cdets[i, 5:33]
        pts = [(int(bbox[0]) + info_bbox[j], int(bbox[1]) + info_bbox[j+1]) for j in xrange(0,28,2)]

        ploygon_test = Polygon(pts)
        if not ploygon_test.is_valid:
            print('non-ploygon detected')
            delete_inds.append(i)
        if int(ploygon_test.area) < 10:
            print('neg-ploygon')
            delete_inds.append(i)
    dets = np.delete(dets, delete_inds, 0)
    cdets = np.delete(cdets, delete_inds, 0)
    return dets, cdets

if __name__ == "__main__":
    cfg_from_file(cofig_file)
    net = caffe.Net(net_prototxt, model, caffe.TEST)

    for image in images:
        im = cv2.imread(image)
        scores, boxes, infos_h, infos_w = im_detect(net, im, None) 

    
        assert(scores.shape[0] == infos_h.shape[0] == infos_w.shape[0]) , 'length mismatch'
        inds = np.where(scores[:, 1] > 0.5)[0]

        cls_scores = scores[inds, 1]
        
        cls_boxes = boxes[inds, 4:8]
        ## curve
        cls_infos_h = infos_h[inds, :14]
        cls_infos_w = infos_w[inds, :14]

        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)

        # stack h and w pred.
        cls_infos = np.zeros((cls_infos_h.shape[0], 28))
        wh_stack_temp = np.dstack((cls_infos_w, cls_infos_h))  
        assert(wh_stack_temp.shape[0] == cls_infos.shape[0]), 'wh stack length mismatch.'
        for ixstack, row_cls_infos in enumerate(cls_infos):
            cls_infos[ixstack] = wh_stack_temp[ixstack].ravel()

        cls_dets_withInfo = np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_infos)) \
            .astype(np.float32, copy=False)
        
        cls_dets, cls_dets_withInfo = nps(cls_dets, cls_dets_withInfo)
        if cfg.TEST.USE_PNMS:
            keep = pnms(cls_dets_withInfo, cfg.TEST.PNMS)
        else:
            keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        cls_dets_withInfo = cls_dets_withInfo[keep, :]
        
        vis(im, cls_dets_withInfo, 0.1)
                
