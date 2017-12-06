#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConvertData_gpu(const int nthreads, const Dtype* in_data,
    const int in_offset, const int  out_offset, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      out_data[out_offset+index] = in_data[in_offset+index];
  }
}

template <typename Dtype>
void ConvertData(const Blob<Dtype>* data, Blob<Dtype>* convert_data) {
  int num = data->num();
  const int nthreads = data->count(1);
  // LOG(INFO) << "data->count(): " << data->count();
  // LOG(INFO) << "nthreads: " << nthreads;
  Dtype* out_data = convert_data->mutable_gpu_data();
  const Dtype* in_data = data->gpu_data();
  for (int i = 0; i < num; ++i) {
    const int in_offset = data->offset(i);
    const int out_offset = data->offset(num - 1 - i);
    ConvertData_gpu<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, in_data, in_offset, out_offset, out_data);
  }
}

template <typename Dtype>
void BLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // const int widthNum = bottom[0]->shape(0);
  // const int sampleNum = bottom[0]->shape(1);
  // const int dimNum = bottom[0]->shape(2);

  // LOG(INFO) << "widthNum" << widthNum;
  // LOG(INFO) << "sampleNum" << sampleNum;
  // LOG(INFO) << "dimNum" << dimNum;
  // getchar();

  // const Dtype* tmp = bottom[0]->cpu_data();
  // for (int i = 0; i < widthNum; ++i)
  // {
  //   LOG(INFO) << " i " << i << " bottom: " << tmp[i*sampleNum*dimNum+0*dimNum+0];
  // }
  // getchar();
  // const Dtype* tmp2 = bottom[1]->cpu_data();
  // for (int i = 0; i < widthNum; ++i)
  // {
  //   LOG(INFO) << " i " << i << " top: " << tmp2[i*sampleNum+0];
  // }
  // getchar();
  // Hacky fix for test time... reshare all the shared blobs.
  // TODO: somehow make this work non-hackily.
  for (int m = 0; m < 2; ++m) {
    if (this->phase_ == TEST) {
      unrolled_net_[m]->ShareWeights();
    }

    DCHECK_EQ(recur_input_blobs_[m].size(), recur_output_blobs_[m].size() );
    for (int i = 0; i < recur_input_blobs_[m].size(); ++i) {
      const int count = recur_input_blobs_[m][i]->count();
      DCHECK_EQ(count, recur_output_blobs_[m][i]->count());
      const Dtype* timestep_T_data = recur_output_blobs_[m][i]->gpu_data();
      Dtype* timestep_0_data = recur_input_blobs_[m][i]->mutable_gpu_data();
      caffe_copy(count, timestep_T_data, timestep_0_data);
    }
    // @Helios: in backward subnet, input blob should be reverse.
    if (m == 1) {
      ConvertData(bottom[0], x_input_blob_[m]);
      ConvertData(bottom[1], cont_input_blob_[m]);
      unrolled_net_[m]->ForwardPrefilled();
      ConvertData(output_blobs_[1][0], concate_iuput_blob_[1]);
      // LOG(INFO) << "end";
    } else {
      unrolled_net_[m]->ForwardPrefilled();      
    }
  }
  // @Helios: do forward of concate layer
  concate_layer_->Forward(concate_iuput_blob_, concate_ouput_blob_);
  
}


template <typename Dtype>
void ConvertDiff(const Blob<Dtype>* data, Blob<Dtype>* convert_data) {
  int num = data->num();
  const int nthreads = data->count(1);
  // LOG(INFO) << "data->count(): " << data->count();
  // LOG(INFO) << "nthreads: " << nthreads;
  Dtype* out_data = convert_data->mutable_gpu_diff();
  const Dtype* in_data = data->gpu_diff();
  for (int i = 0; i < num; ++i) {
    const int in_offset = data->offset(i);
    const int out_offset = data->offset(num - 1 - i);
    ConvertData_gpu<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, in_data, in_offset, out_offset, out_data);
  }
}

template <typename Dtype>
__global__ void AddupDiff(const int nthreads, const   Dtype* in_data,
    const int m, Dtype* out_data) {
  if (m == 0) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        out_data[index] = in_data[index];
    }
  } else if(m == 1) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        out_data[index] += in_data[index];
        out_data[index] /= 2;
    } 
  }
}

template <typename Dtype>
void BLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";

  // TODO: skip backpropagation to inputs and parameters inside the unrolled
  // net according to propagate_down[0] and propagate_down[2]. For now just
  // backprop to inputs and parameters unconditionally, as either the inputs or
  // the parameters do need backward (or Net would have set
  // layer_needs_backward_[i] == false for this layer).

  // @Helios: do backward of concate layer.
  concate_layer_->Backward(concate_ouput_blob_, concate_propagate_down,
      concate_iuput_blob_);
  // @Helios: need to sum up diff of forward & backward subnet.

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();    
  int nthreads = bottom[0]->count();
  for (int m = 0; m < 2; ++m) {
    if (m == 1) {
      // @Helios: need to convert the diff for backward net.
      ConvertDiff(concate_iuput_blob_[m], output_blobs_[m][0]);
      unrolled_net_[m]->Backward();  
    } else {
      unrolled_net_[m]->Backward();      
    }
    const Dtype* input_diff = x_input_blob_[m]->gpu_diff();
    AddupDiff<Dtype>
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, input_diff, m, bottom_diff);
  }
}



INSTANTIATE_LAYER_GPU_FORWARD(BLSTMLayer);
INSTANTIATE_LAYER_GPU_BACKWARD(BLSTMLayer);
}  // namespace caffe
