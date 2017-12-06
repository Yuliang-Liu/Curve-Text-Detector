#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/accuracy_layer.hpp"

namespace caffe {

// int argmax(vector<int> vec) {
//   int maxidx = 0;
//   int maxval = 0;
//   for (int i = 0; i < vec.size(); ++i){
//     if (maxval < vec[i]){
//       maxidx = i;
//       maxval = vec[i];
//     }
//    }
//    return maxidx; 
// }

template <typename Dtype>
void SequenceAccuracySoftmaxbasedLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  this->top_k_ = this->layer_param_.accuracy_param().top_k();

  this->has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (this->has_ignore_label_) {
    this->ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }

  iter_count_ = count_ = accuracy_ = 0;
  // @Helios: resize the stream container.
  for (int i = 0; i < bottom[0]->count(1); ++i) {
    true_label_count_.push_back(vector<int>() );
  }
  for (int i = 0; i < true_label_count_.size(); ++i) {
    true_label_count_[i].resize(bottom[0]->count(2), 0);
  }
  true_label_.resize(bottom[0]->count(1), -1); 

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);


  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
    ignore_label_num_ = 0;
  }
  normalize_ = this->layer_param_.loss_param().normalize();


}

template <typename Dtype>
void SequenceAccuracySoftmaxbasedLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(this->top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  this->label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  this->outer_num_ = bottom[0]->count(0, this->label_axis_);
  this->inner_num_ = bottom[0]->count(this->label_axis_ + 1);
  CHECK_EQ(this->outer_num_ * this->inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(this->label_axis_);
    top[1]->Reshape(top_shape_per_class);
    this->nums_buffer_.Reshape(top_shape_per_class);
  }


  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }

}

template <typename Dtype>
void SequenceAccuracySoftmaxbasedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int oriInputLength = bottom[0]->shape(0);//input sequence length
  const int nFrame      = bottom[0]->shape(1);
  const int nClasses    = bottom[0]->shape(2);

    // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  vector<vector<Dtype> > theResult(nFrame,vector<Dtype>(nClasses,0));

  const Dtype* label11 = bottom[1]->cpu_data();
  // const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* prob_data = prob_.cpu_data();

  for(int tv = 0; tv < oriInputLength; tv++){
    for(int iv = 0; iv < nFrame; iv++){
      const int label_value = static_cast<int>(label11[tv*nFrame+iv]);
      if(has_ignore_label_ && label_value == ignore_label_)
        continue;
      for(int cv = 0; cv < nClasses; cv++){
        theResult[iv][cv] += prob_data[tv*nFrame*nClasses+iv*nClasses+cv];
      }
    }
  }


  for(int iv = 0; iv < nFrame; iv++){
    float maxScore = 0;
    int preLabel = 0;
    for(int cv = 1; cv < nClasses; cv++){
      if(maxScore < theResult[iv][cv]){
        maxScore = theResult[iv][cv];
        preLabel = cv;
      }
    }
    count_++;
    if(preLabel == label11[0*nFrame+iv])
      accuracy_++;
  }

  LOG(INFO) << "iter_count_: " << iter_count_;
  const int num_batch = bottom[0]->shape(0); 
  iter_count_ += num_batch;

  // if (count == 0) {
  //   top[0]->mutable_cpu_data()[0] = 1;    
  // } else {
  //   top[0]->mutable_cpu_data()[0] = (Dtype)accuracy / count;    
  // }
  if (iter_count_ >= this->layer_param_.accuracy_param().accuracy_iter() ) {
    LOG(INFO) << "right num: " << accuracy_;    
    LOG(INFO) << "total num: " << count_;
    LOG(INFO) << "overall accuracy: " << accuracy_ / count_;
    accuracy_ = count_ = iter_count_ = 0;
  }
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      // if (this->nums_buffer_.cpu_data()[i] != 0) {
      //   LOG(INFO) << "i: " << i;
      //   LOG(INFO) << "each accuracy: " <<  top[1]->cpu_data()[i] / this->nums_buffer_.cpu_data()[i];
      // }
      top[1]->mutable_cpu_data()[i] =
          this->nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / this->nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}



INSTANTIATE_CLASS(SequenceAccuracySoftmaxbasedLayer);
REGISTER_LAYER_CLASS(SequenceAccuracySoftmaxbased);



}  // namespace caffe
