from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, info_syn_transform_inv_h, info_syn_transform_inv_w
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms, pnms
import cPickle
from utils.blob import im_list_to_blob
import os

import re 
from shapely.geometry import *
import matplotlib.pyplot as plts

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
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

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None, info=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    ############################################### curve
    if info:
        info_deltas_h = blobs_out['info_pred_h']
        pred_infos_h = info_syn_transform_inv_h(boxes, info_deltas_h)
        info_deltas_w = blobs_out['info_pred_w']
        pred_infos_w = info_syn_transform_inv_w(boxes, info_deltas_w)
        assert len(boxes) == len(pred_infos_h) == len(pred_infos_w)
    ###############################################

    if info:
        return scores, pred_boxes, pred_infos_h, pred_infos_w
    else:
        return scores, pred_boxes, None

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

# use opencv
def vis_detections_opencv(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    for i in xrange(np.minimum(100, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
    im = cv2.resize(im, (1280, 720))            
    cv2.imshow('Dectecting results.', im)
    cv2.waitKey(0)

def syn_vis_detections_opencv(im, class_name, dets, out_filename, thresh=0.3, ):
    """Visual debugging of detections."""
    for i in xrange(np.minimum(100, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, 4]
        info_bbox = dets[i, 5:33] # syn
        pts = [info_bbox[i] for i in xrange(28)]
        # print pts
        # a = input('stop check')
        assert(len(pts) == 28), 'wrong length.'
        if score > thresh:
            for p in xrange(0,28,2):
                # if p == 0:
                #     cv2.line(im,(int(bbox[0]) - int(pts[p%28]), int(bbox[1]) - int(pts[(p+1)%28])), 
                #              (int(bbox[0]) - int(pts[(p+2)%28]), int(bbox[1]) - int(pts[(p+3)%28])),(0,0,255),2) 
                # else:
                cv2.line(im,(int(bbox[0]) + int(pts[p%28]), int(bbox[1]) + int(pts[(p+1)%28])), 
                             (int(bbox[0]) + int(pts[(p+2)%28]), int(bbox[1]) + int(pts[(p+3)%28])),(0,0,255),2) 

    imk = cv2.resize(im, (1280, 720)) # visualization
    cv2.imshow('Dectecting results syn.', imk)
    cv2.waitKey(0)

def nps(dets, cdets):
    delete_inds = []
    for i in xrange(cdets.shape[0]):
        bbox = cdets[i, :4]
        score = cdets[i, 4]
        info_bbox = cdets[i, 5:33]
        pts = [(int(bbox[0]) + info_bbox[j], int(bbox[1]) + info_bbox[j+1]) for j in xrange(0,28,2)]

        # print('try ploygon test')
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

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb._image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_boxes_info = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    cnt=0
    cnt1=0
    cnt2=2

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
    cnt=0
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truths.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        print imdb.image_path_at(i)
        _t['im_detect'].tic()
        scores, boxes, infos_h, infos_w = im_detect(net, im, box_proposals, info = True) 
        _t['im_detect'].toc()
        _t['misc'].tic()

        for j in xrange(1, imdb.num_classes):
            assert(scores.shape[0] == infos_h.shape[0] == infos_w.shape[0]) , 'length mismatch'
            inds = np.where(scores[:, j] > 0.5)[0]

            ind_35 = np.where((scores[:, j] > 0.3))[0]
            
            print "thresh>0.5:   ",len(inds)
            print "0.5>thresh>0.3:   ",len(ind_35)
            print "all:   ",len(scores[:,j])

            cnt+=len(inds)
            cnt1+=len(ind_35)
            cnt2+=len(scores[:,j])
            cls_scores = scores[inds, j]
            
            if cfg.TEST.AGNOSTIC:
                cls_boxes = boxes[inds, 4:8]
                ## SYN
                cls_infos_h = infos_h[inds, :14]
                cls_infos_w = infos_w[inds, :14]
            else:
                pass
                # cls_boxes = boxes[inds, j*4:(j+1)*4]
                # cls_infos_h = infos_h[inds, :j*14]
                # cls_infos_w = infos_w[inds, :j*14]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            # stack h and w pred.
            cls_infos = np.zeros((cls_infos_h.shape[0], 28))
            wh_stack_temp = np.dstack((cls_infos_w, cls_infos_h))  
            assert(wh_stack_temp.shape[0] == cls_infos.shape[0]), 'wh stack length mismatch.'
            for ixstack, row_cls_infos in enumerate(cls_infos):
                cls_infos[ixstack] = wh_stack_temp[ixstack].ravel()

            # if 1:
                # print(cls_infos)
                # debug = input('debug test.py check cls_infos stack.')

            cls_dets_withInfo = np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_infos)) \
                .astype(np.float32, copy=False)
            
            cls_dets, cls_dets_withInfo = nps(cls_dets, cls_dets_withInfo)
            if cfg.TEST.USE_PNMS:
                keep = pnms(cls_dets_withInfo, cfg.TEST.PNMS)
            else:
                keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            cls_dets_withInfo = cls_dets_withInfo[keep, :]
            
            if vis: 
                # vis_detections_opencv(im, imdb.classes[j], cls_dets)
                syn_vis_detections_opencv(im, imdb.classes[j], cls_dets_withInfo, imdb.image_path_at(i), 0.1)
                
            all_boxes[j][i] = cls_dets
            all_boxes_info[j][i] = cls_dets_withInfo

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])

            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    all_boxes_info[j][i] = all_boxes_info[j][i][keep, :]

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes_info, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes_info, output_dir) 
    print "avg proposals>0.5:  ",cnt/num_images," 0.3--0.5:  ",cnt1/num_images,"all:  ",cnt2/num_images