#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/convert_layer.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

  template <typename Dtype>
__global__ void ConvertForward(const int sampleNum, const int dimNum, const int widthNum, const Dtype* in, Dtype* out) {
    int n = sampleNum*dimNum*widthNum;
    CUDA_KERNEL_LOOP(index, n) {
      int sv = index/(dimNum*widthNum);
      int wv = index%widthNum;
      int dv = (index%(dimNum*widthNum) )/widthNum;
      out[wv*sampleNum*dimNum+sv*dimNum+dv] = in[index];
  }
}


template <typename Dtype>
__global__ void ContForward(const int sampleNum, const int widthNum, Dtype* out) {
    int n = sampleNum*widthNum;
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = index < sampleNum ? 0 : 1; 
      // out[index] = 1; 
  }
}

template <typename Dtype>
void ConvertLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();


  const int sampleNum = bottom[0]->shape(0);
  const int dimNum = bottom[0]->shape(1);
  const int widthNum = bottom[0]->shape(bottom[0]->num_axes()-1);

  const int count = bottom[0]->count();
  if (top.size() > 1) {
    Dtype* cont_sequence = top[1]->mutable_gpu_data();
    ContForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      sampleNum,widthNum,cont_sequence);
    CUDA_POST_KERNEL_CHECK;
  }
  ConvertForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    sampleNum,dimNum,widthNum,bottom_data,top_data);
  CUDA_POST_KERNEL_CHECK;

  // const int _sampleNum = bottom[0]->shape(0);
  // const int _dimNum = bottom[0]->shape(1);
  // const int _heightNum = bottom[0]->shape(2);
  // const int _widthNum = bottom[0]->shape(3);

  // Dtype* words = bottom[0]->mutable_cpu_data(); 
  // cv::Mat img = cv::Mat(_heightNum*_dimNum,_widthNum,CV_32FC1,words);
  // cv::imshow("fun",img);

  // Dtype* words_out = top[0]->mutable_cpu_data(); 
  // cv::Mat img_out = cv::Mat(_heightNum*_dimNum,_widthNum,CV_32FC1, words_out);
  // cv::imshow("fun_out",img_out);
  // cv::waitKey(0); 

  // Dtype pixel_max = 0;
  // Dtype pixel_min = 0;
  // for(int pixel = 0; pixel < bottom[0]->count(); pixel++){
  //   if(words[pixel] < pixel_min){pixel_min = words[pixel];}
  //   if(words[pixel] > pixel_max){pixel_max = words[pixel];}
  // }
  // std::cout << "pixel_max:" << pixel_max << std::endl;
  // std::cout << "pixel_min:" << pixel_min << std::endl;

  // Dtype pixel_max_out = 0;
  // Dtype pixel_min_out = 0;
  // for(int pixel = 0; pixel < top[0]->count(); pixel++){
  //   if(words_out[pixel] < pixel_min_out){pixel_min_out = words_out[pixel];}
  //   if(words_out[pixel] > pixel_max_out){pixel_max_out = words_out[pixel];}
  // }
  // std::cout << "pixel_max_out:" << pixel_max_out << std::endl;
  // std::cout << "pixel_min_out:" << pixel_min_out << std::endl;
}

template <typename Dtype>
__global__ void ConvertBackward(const int sampleNum, const int dimNum, const int widthNum, const Dtype* in, Dtype* out) {
    int n = sampleNum*dimNum*widthNum;
    CUDA_KERNEL_LOOP(index, n) {
      int sv = index/(dimNum*widthNum);
      int wv = index%widthNum;
      int dv = (index%(dimNum*widthNum) )/widthNum;
      out[index] = in[wv*sampleNum*dimNum+sv*dimNum+dv];
  }
}


template <typename Dtype>
void ConvertLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int sampleNum = bottom[0]->shape(0);
    const int dimNum = bottom[0]->shape(1);
    const int widthNum = bottom[0]->shape(bottom[0]->num_axes()-1);


    const int count = bottom[0]->count();

    ConvertBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        sampleNum,dimNum,widthNum,top_diff,bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvertLayer);

// INSTANTIATE_LAYER_GPU_FORWARD(TransitionLayer);
// INSTANTIATE_LAYER_GPU_BACKWARD(TransitionLayer);
}  // namespace caffe
