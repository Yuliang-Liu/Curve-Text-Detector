#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/st_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void STLossForwardGPU(const int nthreads, int N, 
		int output_H_, int output_W_, const Dtype* theta, Dtype* loss_array) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		Dtype input_x = s * 2.0 / output_H_ - 1;
		Dtype input_y = t * 2.0 / output_W_ - 1;
		
		Dtype output_x = theta[6*i] * input_x + theta[6*i+1] * input_y + theta[6*i+2];
		Dtype output_y = theta[6*i+3] * input_x + theta[6*i+4] * input_y + theta[6*i+5];
		
		Dtype loss = (Dtype)0;
		
		if(output_x < -1) {
			loss += (output_x + 1) * (output_x + 1) / 2;
		} else if(output_x > 1) {
			loss += (output_x - 1) * (output_x - 1) / 2;
		}
		
		if(output_y < -1) {
			loss += (output_y + 1) * (output_y + 1) / 2;
		} else if(output_y > 1) {
			loss += (output_y - 1) * (output_y - 1) / 2;
		}
		
		loss_array[index] = loss;
  }
}

template <typename Dtype>
void STLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	string prefix = "STLossLayer::Forward_gpu::\t";

	const Dtype* theta = bottom[0]->gpu_data();
	Dtype* loss_array = loss_.mutable_gpu_data();
	
	caffe_gpu_set(loss_.count(), (Dtype)0, loss_array);
	
	const int nthreads = N * output_H_ * output_W_;
	STLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, output_H_, output_W_, theta, loss_array);
	
	Dtype loss;
	caffe_gpu_asum(nthreads, loss_array, &loss);
	loss /= nthreads;
	
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void STLossBackwardGPU(const int nthreads, int N, 
		int output_H_, int output_W_, const Dtype* theta, Dtype* dtheta_tmp) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int i = index / (output_W_ * output_H_);

		Dtype input_x = s * 2.0 / output_H_ - 1;
		Dtype input_y = t * 2.0 / output_W_ - 1;
		
		Dtype output_x = theta[6*i] * input_x + theta[6*i+1] * input_y + theta[6*i+2];
		Dtype output_y = theta[6*i+3] * input_x + theta[6*i+4] * input_y + theta[6*i+5];
		
		Dtype d1 = (Dtype)0, d2 = (Dtype)0;
		
		if(output_x < -1) {
			d1 = output_x + 1;
		} else if(output_x > 1) {
			d1 = output_x - 1;
		}
		
		if(output_y < -1) {
			d2 = output_y + 1;
		} else if(output_y > 1) {
			d2 = output_y - 1;
		}
		
		dtheta_tmp[(6*i) * (output_H_ * output_W_) + s * output_W_ + t] = d1 * input_x;
		dtheta_tmp[(6*i+1) * (output_H_ * output_W_) + s * output_W_ + t] = d1 * input_y;
		dtheta_tmp[(6*i+2) * (output_H_ * output_W_) + s * output_W_ + t] = d1;
		dtheta_tmp[(6*i+3) * (output_H_ * output_W_) + s * output_W_ + t] = d2 * input_x;
		dtheta_tmp[(6*i+4) * (output_H_ * output_W_) + s * output_W_ + t] = d2 * input_y;
		dtheta_tmp[(6*i+5) * (output_H_ * output_W_) + s * output_W_ + t] = d2;
  }
}

template <typename Dtype>
void STLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* theta = bottom[0]->gpu_data();
	Dtype* dtheta_tmp = dtheta_tmp_.mutable_gpu_data();
	Dtype* all_ones_vec = all_ones_vec_.mutable_gpu_data();
	Dtype* dtheta = bottom[0]->mutable_gpu_diff();
	
	caffe_gpu_set(all_ones_vec_.count(), (Dtype)1, all_ones_vec);
	
	const int nthreads = N * output_H_ * output_W_;
	STLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, output_H_, output_W_, theta, dtheta_tmp);
	     
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N * 6, 1, output_H_ * output_W_, 
			(Dtype)1., dtheta_tmp, all_ones_vec, (Dtype)0., dtheta);
			
	caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0] / nthreads, dtheta);
}

INSTANTIATE_LAYER_GPU_FUNCS(STLossLayer);

}  // namespace caffe
