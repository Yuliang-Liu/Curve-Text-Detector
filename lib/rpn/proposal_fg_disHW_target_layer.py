import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.bbox_transform import info_syn_transform_hw
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalFgDisHWTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        assert(len(top) == 10), 'for gt_info SYN LSTM distriHW'
        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        # labels
        top[1].reshape(1, 1, 1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4, 1, 1)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4, 1, 1)
        ############################################## curve
        # info_targets hw
        top[5].reshape(1, (self._num_classes - 1) * 16, 1, 1)
        top[6].reshape(1, (self._num_classes - 1) * 16, 1, 1)
        # info_inside_weights hw
        top[7].reshape(1, (self._num_classes - 1) * 16, 1, 1)
        top[8].reshape(1, (self._num_classes - 1) * 16, 1, 1)
        # ##############################################
        # ############################################## Fg slice
        top[9].reshape(1, 5, 1, 1)
        # ##############################################

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        ############################################## curve
        gt_info = bottom[2].data
        ##############################################

        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        gt_info = gt_info.reshape(gt_info.shape[0], gt_info.shape[1]) # curve
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # add the gt_boxes to the rois
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only, all index_id are 0?
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE # 128
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) # 0.25 * 128

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, info_targets_h, info_targets_w, bbox_inside_weights, info_inside_weights_h, info_inside_weights_w, rois_fg = syn_sample_rois(
            all_rois, gt_boxes, gt_info, fg_rois_per_image,
            rois_per_image, self._num_classes) # syn

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        labels = labels.reshape((labels.shape[0], 1, 1, 1))
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets.reshape((bbox_targets.shape[0], bbox_targets.shape[1], 1, 1))
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        ########## creating layers in the setup process for testing. In the forward process, blobs often need to be reshaped again.

        # info_targets quad
        # modified by yl
        info_targets_h = info_targets_h.reshape((info_targets_h.shape[0], info_targets_h.shape[1], 1, 1))
        top[5].reshape(*info_targets_h.shape)
        top[5].data[...] = info_targets_h
        info_targets_w = info_targets_w.reshape((info_targets_w.shape[0], info_targets_w.shape[1], 1, 1))
        top[6].reshape(*info_targets_w.shape)
        top[6].data[...] = info_targets_w
        # info_inside_weights_h
        info_inside_weights_h = info_inside_weights_h.reshape((info_inside_weights_h.shape[0], info_inside_weights_h.shape[1], 1, 1))
        top[7].reshape(*info_inside_weights_h.shape)
        top[7].data[...] = info_inside_weights_h
        # info_inside_weights_w
        info_inside_weights_w = info_inside_weights_w.reshape((info_inside_weights_w.shape[0], info_inside_weights_w.shape[1], 1, 1))
        top[8].reshape(*info_inside_weights_w.shape)
        top[8].data[...] = info_inside_weights_w

        ##########  rois_fg & rois_bg
        rois_fg = rois_fg.reshape((rois_fg.shape[0], rois_fg.shape[1], 1, 1))
        top[9].reshape(*rois_fg.shape)
        top[9].data[...] = rois_fg


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def syn_get_bbox_regression_labels(bbox_target_data, info_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    # compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    ######## curve
    num_fg = info_target_data.shape[0]
    info_targets_fg = np.zeros((num_fg, 32 * (num_classes-1)), dtype = np.float32) 
    info_inside_weights_fg = np.zeros(info_targets_fg.shape, dtype = np.float32)
    ########

    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * (1 if cls > 0 else 0)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        start2 = 32 * (1 if (cls-1) > 0 else 0) # curve 32
        assert(start2 == 0), 'fg should start from very beginning.'
        end2 = start2 + 32 
        info_targets_fg[ind, start2:end2] = info_target_data[ind, :]
        info_inside_weights_fg[ind, start2:end2] = cfg.TRAIN.INFO_INSIDE_WEIGHTS

    return bbox_targets, info_targets_fg, bbox_inside_weights, info_inside_weights_fg


def syn_compute_targets(ex_rois, gt_rois, gt_info, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    ########################################### curve
    assert gt_rois.shape[0] == gt_info.shape[0]
    assert gt_info.shape[1] == 28
    ###########################################

    targets = bbox_transform(ex_rois, gt_rois)
    # curve
    targets_2 = info_syn_transform_hw(ex_rois, gt_info)
    
    if DEBUG:
        print 'targets after bbox_transform:'
        print targets
        print 'targets_info after bbox_transform:'
        print targets_2
        print 'targets_info_curve after bbox_transform:'

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

        targets_2 = ((targets_2 - np.array(cfg.TRAIN.INFO_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.INFO_NORMALIZE_STDS))

    if DEBUG:
        print 'targets after normalize:'
        print targets
        print 'targets_info after normalize:'
        print targets_2

    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False), targets_2

def syn_sample_rois(all_rois, gt_boxes, gt_info, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & # 0.5
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0] # 0.1
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image # 128 - 32
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds
    
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    rois_fg = all_rois[fg_inds]
    
    # print 'proposal_target_layer:', rois ## curve
    bbox_target_data, info_target_data = syn_compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], gt_info[gt_assignment[keep_inds], :28], labels)
    assert(info_target_data.shape[1] == 32), 'info_targets num_columns ' + str(info_target_data.shape[1]) + ' should be 32'
    
    info_target_data_fg = info_target_data[:fg_rois_per_this_image]     
    if DEBUG:
        print 'bbox_target_data after _compute_targets:'
        print bbox_target_data
        print 'info_target_data after _compute_targets:'
        print info_target_data

    bbox_targets, info_targets, bbox_inside_weights, info_inside_weights = \
        syn_get_bbox_regression_labels(bbox_target_data, info_target_data_fg, num_classes) # syn
    info_target_data_fg_h = info_targets[:, 0:16]
    info_target_data_fg_w = info_targets[:, 16:32]
    info_inside_weights_h = info_inside_weights[:, 0:16]
    info_inside_weights_w = info_inside_weights[:, 16:32]          

    return labels, rois, bbox_targets, info_target_data_fg_h, info_target_data_fg_w, bbox_inside_weights, info_inside_weights_h, info_inside_weights_w, rois_fg