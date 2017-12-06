// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include "quad_gpu_nms.hpp"
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


// float a_p1x = a[0]+a[5], a_p2x = a[0]+a[7], a_p3x = a[0]+a[9] , a_p4x = a[0]+a[11];
// float a_p5x = a[0]+a[13], a_p6x = a[0]+a[15], a_p7x = a[0]+a[17] , a_p8x = a[0]+a[19];
// float a_p9x = a[0]+a[21], a_p10x = a[0]+a[23], a_p11x = a[0]+a[25] , a_p12x = a[0]+a[27];
// float a_p13x = a[0]+a[29], a_p14x = a[0]+a[31]; // ax
// float a_p1y = a[1]+a[6], a_p2y = a[1]+a[8], a_p3y = a[1]+a[10], a_p4y = a[1]+a[12];
// float a_p5y = a[1]+a[14], a_p6y = a[1]+a[16], a_p7y = a[1]+a[18] , a_p8y = a[1]+a[20];
// float a_p9y = a[1]+a[22], a_p10y = a[1]+a[24], a_p11y = a[1]+a[26] , a_p12y = a[1]+a[28];
// float a_p13y = a[1]+a[30], a_p14y = a[1]+a[32]; // ay

// float b_p1x = b[0]+b[5], b_p2x = b[0]+b[7], b_p3x = b[0]+b[9] , b_p4x = b[0]+b[11];
// float b_p5x = b[0]+b[13], b_p6x = b[0]+b[15], b_p7x = b[0]+b[17] , b_p8x = b[0]+b[19];
// float b_p9x = b[0]+b[21], b_p10x = b[0]+b[23], b_p11x = b[0]+b[25] , b_p12x = b[0]+b[27];
// float b_p13x = b[0]+b[29], b_p14x = b[0]+b[31]; // bx
// float b_p1y = b[1]+b[6], b_p2y = b[1]+b[8], b_p3y = b[1]+b[10], b_p4y = b[1]+b[12];
// float b_p5y = b[1]+b[14], b_p6y = b[1]+b[16], b_p7y = b[1]+b[18] , b_p8y = b[1]+b[20];
// float b_p9y = b[1]+b[22], b_p10y = b[1]+b[24], b_p11y = b[1]+b[26] , b_p12y = b[1]+b[28];
// float b_p13y = b[1]+b[30], b_p14y = b[1]+b[32]; // by

__device__ inline float quad_devIoU(float const * const ba, float const * const ax, float const * const ay, float const * const bb, float const * const bx, float const * const by) {
  int xmin,ymin,xmax,ymax;
  if (ba[0]<=bb[0]) xmin = (int)ba[0]; else xmin = (int)bb[0];
  if (ba[2]>=bb[2]) xmax = (int)ba[2]; else xmax = (int)bb[2];
  if (ba[1]<=bb[1]) ymin = (int)ba[1]; else ymin = (int)bb[1];
  if (ba[3]>=bb[3]) ymax = (int)ba[3]; else ymax = (int)bb[3];

  float b_width = bb[2]-bb[0], b_height = bb[3]-bb[1]; 
  float a_width = aa[2]-aa[0], a_height = aa[3]-aa[1];
  if(((xmax-xmin)>=(a_width+b_width)) || ((ymax-ymin)>=(a_height+b_height)))
    return 0;
  
  bool oddNodes = false; //
  int countOverlap=0; int countGT=0; int countPB=0;
  for(float i=xmin; i<=xmax; i=i+(xmax-xmin)*0.04) 
  {
    for(float j=ymin; j<=ymax; j=j+(ymax-ymin)*0.04)
    { 
      int k,l = 14;
      oddNodes=false;
      for (k =0; k<(14+1);k++){
        if((ay[k] < j && ay[l] >= j || ay[l] < j && ay[k] >= j) && (ax[k] <= i || ax[l] <= i)){
          oddNodes^=(ax[k]+(j-ay[k])/(ay[l]-ay[k])*(ax[l]-ax[k])<i)
        }
      }
      countPB+=int(oddNodes);
      if (oddNodes==true)
      {
        oddNodes=false;
        int k,l = 14;
        for (k =0; k<(14+1);k++){
        if((by[k] < j && by[l] >= j || by[l] < j && by[k] >= j) && (bx[k] <= i || bx[l] <= i)){
          oddNodes^=(bx[k]+(j-by[k])/(by[l]-by[k])*(bx[l]-bx[k])<i)
          }
        }
        countGT+=int(oddNodes);
        countOverlap+=int(oddNodes);
      }
      else
      {
        oddNodes=false;
        int k,l = 14;
        for (k =0; k<(14+1);k++){
        if((by[k] < j && by[l] >= j || by[l] < j && by[k] >= j) && (bx[k] <= i || bx[l] <= i)){
          oddNodes^=(bx[k]+(j-by[k])/(by[l]-by[k])*(bx[l]-bx[k])<i)
          }
        }
        countGT+=int(oddNodes);
      }
    } 
  }
  return (countOverlap)*1.0/(countPB+countGT-countOverlap);  
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *bdev_boxes, const float *xdev_boxes, const float *ydev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;


  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float xblock_boxes[threadsPerBlock * 14]; 
  __shared__ float yblock_boxes[threadsPerBlock * 14]; 
  __shared__ float bblock_boxes[threadsPerBlock * 14]; 

  if (threadIdx.x < col_size) {
    for (int j = 0; j<14; j++){
      bblock_boxes[threadIdx.x * 14 + j] = 
        bdev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 14 + j];   
      xblock_boxes[threadIdx.x * 14 + j] = 
        xdev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 14 + j];   
      yblock_boxes[threadIdx.x * 14 + j] = 
        ydev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 14 + j];   
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *bcur_box = bxdev_boxes + cur_box_idx * 14; 
    const float *xcur_box = xdev_boxes + cur_box_idx * 14; 
    const float *ycur_box = ydev_boxes + cur_box_idx * 14; 
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (quad_devIoU(bcur_box, xcur_box, ycur_box, bblock_boxes + i*14, xblock_boxes + i * 14, yblock_boxes + i * 14) > nms_overlap_thresh) { 
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _nms_quad(int* keep_out, int* num_out, const float* boxes_bound, const float* boxes_hostx, const float* boxes_hosty, int boxes_num, // 4 13
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_devx = NULL;
  float* boxes_devy = NULL;
  float* boxes_devb = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  // std::cout<<"col_blocks: "<<col_blocks<<std::endl;  // 1
  // std::cout<<"sizeof(unsigned long long): "<<sizeof(unsigned long long)<<std::endl;  // 8
  CUDA_CHECK(cudaMalloc(&boxes_devx,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&boxes_devy,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&boxes_devb,
                        boxes_num * boxes_dim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(boxes_devx,
                        boxes_hostx,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(boxes_devy,
                        boxes_hosty,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(boxes_devb,
                        boxes_bound,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_devb,
                                  boxes_devx,
                                  boxes_devy,
                                  mask_dev);
  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost)); // an illegal instruction was encountered
  
  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;
    // int aaa=1; std::cout<<"threadsPerBlock "<< threadsPerBlock<< std::endl
    //                     <<"boxes_num "<< boxes_num<< std::endl
    //                     <<"inblock "<< inblock<< std::endl
    //                     <<"(1ULL << inblock) "<< (1ULL << inblock)<< std::endl
    //                     <<"remv[nblock] "<< remv[nblock]<< std::endl
    //                     <<"nblock "<< nblock<< std::endl
    //                     <<"col_blocks "<< col_blocks<< std::endl;
    //                     std::cin>>aaa;
    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;
  CUDA_CHECK(cudaFree(boxes_dev)); // an illegal instruction was encountered
  CUDA_CHECK(cudaFree(mask_dev));
}
