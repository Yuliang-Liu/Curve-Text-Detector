# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    # print (targets_dx,targets_dy,targets_dw,targets_dh)
    # debug = input('stop here bbox_transform.py')
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def info_syn_transform_hw(ex_rois, gt_info):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert gt_info.shape[1] == 28, 'length does not match gt_info'

    gt_p1h = gt_info[:, 1]
    gt_p2h = gt_info[:, 3]
    gt_p3h = gt_info[:, 5]
    gt_p4h = gt_info[:, 7]
    gt_p5h = gt_info[:, 9]
    gt_p6h = gt_info[:, 11]
    gt_p7h = gt_info[:, 13]
    gt_p8h = gt_info[:, 15]
    gt_p9h = gt_info[:, 17]
    gt_p10h = gt_info[:, 19]
    gt_p11h = gt_info[:, 21]
    gt_p12h = gt_info[:, 23]
    gt_p13h = gt_info[:, 25]
    gt_p14h = gt_info[:, 27]

    gt_p1w = gt_info[:, 0]
    gt_p2w = gt_info[:, 2]
    gt_p3w = gt_info[:, 4]
    gt_p4w = gt_info[:, 6]
    gt_p5w = gt_info[:, 8]
    gt_p6w = gt_info[:, 10]
    gt_p7w = gt_info[:, 12]
    gt_p8w = gt_info[:, 14]
    gt_p9w = gt_info[:, 16]
    gt_p10w = gt_info[:, 18]
    gt_p11w = gt_info[:, 20]
    gt_p12w = gt_info[:, 22]
    gt_p13w = gt_info[:, 24]
    gt_p14w = gt_info[:, 26]

    targets_dp1h = ( gt_p1h - ex_heights) * 0.5 / ex_heights
    targets_dp2h = ( gt_p2h - ex_heights) * 0.5 / ex_heights
    targets_dp3h = ( gt_p3h - ex_heights) * 0.5 / ex_heights
    targets_dp4h = ( gt_p4h - ex_heights) * 0.5 / ex_heights
    targets_dp5h = ( gt_p5h - ex_heights) * 0.5 / ex_heights
    targets_dp6h = ( gt_p6h - ex_heights) * 0.5 / ex_heights
    targets_dp7h = ( gt_p7h - ex_heights) * 0.5 / ex_heights
    targets_dp8h = ( gt_p8h - ex_heights) * 0.5 / ex_heights
    targets_dp9h = ( gt_p9h - ex_heights) * 0.5 / ex_heights
    targets_dp10h = ( gt_p10h - ex_heights) * 0.5 / ex_heights
    targets_dp11h = ( gt_p11h - ex_heights) * 0.5 / ex_heights
    targets_dp12h = ( gt_p12h - ex_heights) * 0.5 / ex_heights
    targets_dp13h = ( gt_p13h - ex_heights) * 0.5 / ex_heights
    targets_dp14h = ( gt_p14h - ex_heights) * 0.5 / ex_heights

    targets_dp1w = ( gt_p1w - ex_widths) * 0.5 / ex_widths
    targets_dp2w = ( gt_p2w - ex_widths) * 0.5 / ex_widths
    targets_dp3w = ( gt_p3w - ex_widths) * 0.5 / ex_widths
    targets_dp4w = ( gt_p4w - ex_widths) * 0.5 / ex_widths
    targets_dp5w = ( gt_p5w - ex_widths) * 0.5 / ex_widths
    targets_dp6w = ( gt_p6w - ex_widths) * 0.5 / ex_widths
    targets_dp7w = ( gt_p7w - ex_widths) * 0.5 / ex_widths
    targets_dp8w = ( gt_p8w - ex_widths) * 0.5 / ex_widths
    targets_dp9w = ( gt_p9w - ex_widths) * 0.5 / ex_widths
    targets_dp10w = ( gt_p10w - ex_widths) * 0.5 / ex_widths
    targets_dp11w = ( gt_p11w - ex_widths) * 0.5 / ex_widths
    targets_dp12w = ( gt_p12w - ex_widths) * 0.5 / ex_widths
    targets_dp13w = ( gt_p13w - ex_widths) * 0.5 / ex_widths
    targets_dp14w = ( gt_p14w - ex_widths) * 0.5 / ex_widths

    encode_0 = np.zeros_like(targets_dp1w) 
    targets = np.vstack((encode_0, encode_0, targets_dp1h, targets_dp2h, targets_dp3h, targets_dp4h, targets_dp5h, targets_dp6h, targets_dp7h, targets_dp8h, targets_dp9h, targets_dp10h, targets_dp11h, targets_dp12h, targets_dp13h, targets_dp14h, 
        encode_0, encode_0, targets_dp1w, targets_dp2w, targets_dp3w, targets_dp4w, targets_dp5w, targets_dp6w, targets_dp7w, targets_dp8w, targets_dp9w, targets_dp10w, targets_dp11w, targets_dp12w, targets_dp13w, targets_dp14w)).transpose() # 44

    return targets

def info_syn_transform_inv_h(boxes, deltas):
    ''' Return the offest of 14 cors '''
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    assert len(deltas[0,:]) == 16, 'info_inv length wrong'

    dp1h = deltas[:, 2::16]
    dp2h = deltas[:, 3::16]
    dp3h = deltas[:, 4::16]
    dp4h = deltas[:, 5::16]
    dp5h = deltas[:, 6::16]
    dp6h = deltas[:, 7::16]
    dp7h = deltas[:, 8::16]
    dp8h = deltas[:, 9::16]
    dp9h = deltas[:, 10::16]
    dp10h = deltas[:, 11::16]
    dp11h = deltas[:, 12::16]
    dp12h = deltas[:, 13::16]
    dp13h = deltas[:, 14::16]
    dp14h = deltas[:, 15::16]

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] -boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1]+ 1.0
    # ctr_x = boxes[:, 0] + 0.5 * widths
    # ctr_y = boxes[:, 1] + 0.5 * heights

    pred_dp1h = dp1h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp2h = dp2h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp3h = dp3h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp4h = dp4h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp5h = dp5h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp6h = dp6h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp7h = dp7h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp8h = dp8h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp9h = dp9h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp10h = dp10h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp11h = dp11h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp12h = dp12h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp13h = dp13h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]
    pred_dp14h = dp14h * heights[:, np.newaxis] / 0.5 + heights[:, np.newaxis]

    pred = np.zeros((deltas.shape[0], deltas.shape[1]-2), dtype = deltas.dtype)
    
    pred[:, 0::14] = pred_dp1h 
    pred[:, 1::14] = pred_dp2h 
    pred[:, 2::14] = pred_dp3h 
    pred[:, 3::14] = pred_dp4h 
    pred[:, 4::14] = pred_dp5h 
    pred[:, 5::14] = pred_dp6h 
    pred[:, 6::14] = pred_dp7h 
    pred[:, 7::14] = pred_dp8h 
    pred[:, 8::14] = pred_dp9h 
    pred[:, 9:14] = pred_dp10h
    pred[:, 10::14] = pred_dp11h
    pred[:, 11::14] = pred_dp12h
    pred[:, 12::14] = pred_dp13h
    pred[:, 13::14] = pred_dp14h
    
    return pred

def info_syn_transform_inv_w(boxes, deltas):
    ''' Return the offest of 14 cors '''
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    assert len(deltas[0,:]) == 16, 'info_inv length wrong'

    dp1w =  deltas[:, 2::16]
    dp2w =  deltas[:, 3::16]
    dp3w =  deltas[:, 4::16]
    dp4w =  deltas[:, 5::16]
    dp5w =  deltas[:, 6::16]
    dp6w =  deltas[:, 7::16]
    dp7w =  deltas[:, 8::16]
    dp8w =  deltas[:, 9::16]
    dp9w =  deltas[:, 10::16]
    dp10w = deltas[:, 11::16]
    dp11w = deltas[:, 12::16]
    dp12w = deltas[:, 13::16]
    dp13w = deltas[:, 14::16]
    dp14w = deltas[:, 15::16]

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] -boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1]+ 1.0

    pred_dp1w = dp1w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp2w = dp2w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp3w = dp3w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp4w = dp4w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp5w = dp5w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp6w = dp6w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp7w = dp7w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp8w = dp8w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp9w = dp9w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp10w = dp10w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp11w = dp11w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp12w = dp12w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp13w = dp13w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]
    pred_dp14w = dp14w * widths[:, np.newaxis] / 0.5 + widths[:, np.newaxis]

    pred = np.zeros((deltas.shape[0], deltas.shape[1]-2), dtype = deltas.dtype)
    
    pred[:, 0::14] =  pred_dp1w 
    pred[:, 1::14] =  pred_dp2w 
    pred[:, 2::14] =  pred_dp3w 
    pred[:, 3::14] =  pred_dp4w 
    pred[:, 4::14] =  pred_dp5w 
    pred[:, 5::14] =  pred_dp6w 
    pred[:, 6::14] =  pred_dp7w 
    pred[:, 7::14] =  pred_dp8w 
    pred[:, 8::14] =  pred_dp9w 
    pred[:, 9::14] =  pred_dp10w 
    pred[:, 10::14] =  pred_dp11w 
    pred[:, 11::14] =  pred_dp12w 
    pred[:, 12::14] =  pred_dp13w 
    pred[:, 13::14] =  pred_dp14w 
    
    return pred
    
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
