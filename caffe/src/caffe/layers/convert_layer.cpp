#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/convert_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvertLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() > 1 ) {
    vector<int> cont_scape;
    cont_scape.push_back(bottom[0]->shape(bottom[0]->num_axes()-1 ) );
    cont_scape.push_back(bottom[0]->shape(0));
    top[1]->Reshape(cont_scape);
  }

  vector<int> sequence_scape;
  sequence_scape.push_back(bottom[0]->shape(bottom[0]->num_axes()-1));
  sequence_scape.push_back(bottom[0]->shape(0));
  sequence_scape.push_back(bottom[0]->shape(1));

  top[0]->Reshape(sequence_scape);  
}

template <typename Dtype>
void ConvertLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() > 1 ) {
    vector<int> cont_scape;
    cont_scape.clear();
    cont_scape.push_back(bottom[0]->shape(bottom[0]->num_axes()-1 ) );
    cont_scape.push_back(bottom[0]->shape(0));
    top[1]->Reshape(cont_scape);
  }
  vector<int> sequence_scape;
  sequence_scape.clear();
  sequence_scape.push_back(bottom[0]->shape(bottom[0]->num_axes()-1));
  sequence_scape.push_back(bottom[0]->shape(0));
  sequence_scape.push_back(bottom[0]->shape(1));

  top[0]->Reshape(sequence_scape); 
}

template <typename Dtype>
void ConvertLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* cont_sequence = top[1]->mutable_cpu_data();
  const int sampleNum = bottom[0]->shape(0);
  const int dimNum = bottom[0]->shape(1);
  const int widthNum = bottom[0]->shape(bottom[0]->num_axes()-1);

  for(int tv = 0; tv < widthNum; tv++){
    if(tv == 0){
      for(int sv = 0; sv < sampleNum; sv++){
        cont_sequence[tv*sampleNum+sv] = 0;
      }
    }else{
      for(int sv = 0; sv < sampleNum; sv++){
        cont_sequence[tv*sampleNum+sv] = 1;
      }
    }

  }

  #pragma omp parallel for
  for(int sv = 0; sv < sampleNum; sv++){
    for(int dv = 0; dv < dimNum; dv++){
      for(int wv = 0; wv < widthNum; wv++){
        top_data[wv*sampleNum*dimNum+sv*dimNum+dv] = bottom_data[sv*dimNum*widthNum+dv*widthNum+wv];
      }
    }
  }
}

template <typename Dtype>
void ConvertLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int sampleNum = bottom[0]->shape(0);
    const int dimNum = bottom[0]->shape(1);
    const int widthNum = bottom[0]->shape(bottom[0]->num_axes()-1);

    for(int sv = 0; sv < sampleNum; sv++){
      for(int dv = 0; dv < dimNum; dv++){
        for(int wv = 0; wv < widthNum; wv++){
           bottom_diff[sv*dimNum*widthNum+dv*widthNum+wv] = top_diff[wv*sampleNum*dimNum+sv*dimNum+dv];
        }
      }
    }
    
  }

}


#ifdef CPU_ONLY
STUB_GPU(ConvertLayer);
#endif

INSTANTIATE_CLASS(ConvertLayer);
REGISTER_LAYER_CLASS(Convert);

}  // namespace caffe
