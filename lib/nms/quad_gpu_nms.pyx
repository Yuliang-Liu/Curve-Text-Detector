# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "quad_gpu_nms.hpp":
    void _nms_quad(np.int32_t*, int*, np.float32_t*, np.float32_t*, np.float32_t*, int, int, float, int)

def quad_gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = (dets.shape[1]-5) / 2 
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] \
        keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] \
        scores = dets[:, 4]
    cdef np.ndarray[np.int_t, ndim=1] \
        order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets = dets[order, :]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_bound = sorted_dets[:, :14]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets_x = sorted_dets[:, 0:1] + sorted_dets[:, 5:33:2]
    cdef np.ndarray[np.float32_t, ndim=2] \
        sorted_dets_y = sorted_dets[:, 1:2] + sorted_dets[:, 6:33:2]

    _nms_quad(&keep[0], &num_out, &sorted_bound[0, 0], &sorted_dets_x[0, 0], &sorted_dets_y[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    # print(keep)
    return list(order[keep])