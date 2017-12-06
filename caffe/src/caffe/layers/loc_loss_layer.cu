#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/loc_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LocLossForwardGPU(const int nthreads, const Dtype* locs, 
	Dtype threshold, Dtype* loss_array) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		Dtype loss = (Dtype)0;
		
		if(locs[index] < -threshold) {
			loss += (locs[index] + threshold) * (locs[index] + threshold) / 2;
		} else if(locs[index] > threshold) {
			loss += (locs[index] - threshold) * (locs[index] - threshold) / 2;
		}
		
		loss_array[index] = loss;
  }
}

template <typename Dtype>
void LocLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	string prefix = "LocLossLayer::Forward_gpu::\t";

	const Dtype* locs = bottom[0]->gpu_data();
	Dtype* loss_array = loss_.mutable_gpu_data();
	
	caffe_gpu_set(loss_.count(), (Dtype)0, loss_array);
	
	const int nthreads = N;
	LocLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, locs, threshold, loss_array);
	
	Dtype loss;
	caffe_gpu_asum(nthreads, loss_array, &loss);
	loss /= nthreads;
	
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void LocLossBackwardGPU(const int nthreads, const Dtype* locs, 
	Dtype threshold, Dtype* dLocs) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		if(locs[index] < -threshold) {
			dLocs[index] = locs[index] + threshold;
		} else if(locs[index] > threshold) {
			dLocs[index] = locs[index] - threshold;
		}
  }
}

template <typename Dtype>
void LocLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* locs = bottom[0]->gpu_data();
	Dtype* dloc = bottom[0]->mutable_gpu_diff();
	
	const int nthreads = N;
	LocLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, locs, threshold, dloc);
	     
	caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0] / nthreads, dloc);
}

INSTANTIATE_LAYER_GPU_FUNCS(LocLossLayer);

}  // namespace caffe
