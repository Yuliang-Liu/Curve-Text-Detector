import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

import numpy as np

mod = SourceModule("""
__global__ void dbnumber(float *dest, float *a)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  dest[i] = a[i]*2;
}
""")
blocks = 64
block_size = 128
nbr_values = blocks * block_size

dbnumber = mod.get_function("dbnumber")
a = np.random.randn(16000,16000).astype(np.float32)
dest = np.empty_like(a)
dbnumber(cuda.Out(dest), cuda.In(a), grid=(blocks,1), block=(block_size,1,1))

mod = SourceModule("""
__global__ void pnms_kernel(float n_box, float pnms_overlap_thresh, float *dev_boxes, float *dev_mask)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  dest[i] = a[i]*2;

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int threadsPerBlock = 4*28;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 28];
  if (threadIdx.x < col_size) {
  	for (int j = 0; j<28; j++){
  	  block_boxes[threadIdx.x * 28 + j] = 
  	    dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 28 + j];   
  	}
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 33;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 33) > nms_overlap_thresh) {
        t |= 1 << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
""")

def quad_gpu_nms(dets, thresh, device_id):
	bbox = dets[:, :4]
	scores = dets[:, 4]
	info_bbox = dets[:, 5:33] # syn

	pts = bbox[:, 0:1] + info_bbox
	# DIVUP = lambda m,n :(m) / (n) + ((m) % (n) > 0)
	boxes_num = pts.shape[0]
	boxes_dim = pts.shape[1]
	# threadsPerBlock = 4 * 28 
	blocks = 256
	# boxes_dev = 
	mask_dev = np.empty_like(pts).astype(np.float32)
	# unsigned long long* mask_de
	# col_blocks = DIVUP(boxes_num, threadsPerBlock)
	pnms_kernel = mod.get_function("pnms_kernel")
	pnms_kernel(cuda.In(boxes_num), cuda.In(thresh), cuda.In(pts), cuda.Out(mask_dev), grid=(blocks,1), block=(boxes_num, boxes_dim, 1))


