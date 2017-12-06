#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/accuracy_layer.hpp"

namespace caffe {

int argmax(vector<int> vec) {
  int maxidx = 0;
  int maxval = 0;
  for (int i = 0; i < vec.size(); ++i){
    if (maxval < vec[i]){
      maxidx = i;
      maxval = vec[i];
    }
   }
   return maxidx; 
}

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::LayerSetUp(
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
}

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::Reshape(
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
}

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();

  const int num_batch = bottom[0]->shape(0); 
  const int dim_batch = bottom[0]->count() / num_batch;

  const int num_stream = bottom[0]->shape(1); 
  const int dim_stream = dim_batch / num_stream;
  const int num_labels = bottom[0]->shape(this->label_axis_);

  vector<Dtype> maxval(this->top_k_+1);
  vector<int> max_id(this->top_k_+1);
  if (top.size() > 1) {
    caffe_set(this->nums_buffer_.count(), Dtype(0), this->nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;

  for (int m = 0; m < num_stream; ++m) {
    for (int i = 0; i < num_batch; ++i) {
      for (int j = 0; j < this->inner_num_; ++j) {
        const int label_value =
            static_cast<int>(bottom_label[i * num_stream + m * this->inner_num_ + j]);
        // LOG(INFO) << "label_value: " << label_value;
        // if (this->has_ignore_label_ && label_value == this->ignore_label_) 

        if (i == 0) { // the first point of a character, record the label
          true_label_[m] = label_value;
        }
        if (top.size() > 1) ++this->nums_buffer_.mutable_cpu_data()[label_value];
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, num_labels);
        // Top-k accuracy
        std::vector<std::pair<Dtype, int> > bottom_data_vector;
        for (int k = 0; k < num_labels; ++k) {
          bottom_data_vector.push_back(std::make_pair(
              bottom_data[i * dim_batch + m * dim_stream + k * this->inner_num_ + j], k));
        }
        std::partial_sort(
            bottom_data_vector.begin(), bottom_data_vector.begin() + this->top_k_,
            bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        // check if true label is in top k predictions
        for (int k = 0; k < this->top_k_; k++) {
          // LOG(INFO) << "bottom_data_vector[k].second: " << bottom_data_vector[k].second;
          // LOG(INFO) << "bottom_data_vector[k].first: " << bottom_data_vector[k].first;
          ++true_label_count_[m][bottom_data_vector[k].second];
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
        }
        if (i == num_batch - 1) {
          // reach the end of a word, calc the prediction of this word.          
          if (true_label_[m] == -1) {
            continue;
          }
          ++count;
          if (argmax(true_label_count_[m]) == true_label_[m]) {
            ++accuracy;
          }
          true_label_[m] = -1;
          std::fill(true_label_count_[m].begin(), true_label_count_[m].end(), 0);
          continue;
        } else if (this->layer_param_.accuracy_param().has_accuracy_ignore_label() && 
              label_value == this->layer_param_.accuracy_param().accuracy_ignore_label() ) {
            continue;
        }        
      }    
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  // LOG(INFO) << "count: " << count;  
  LOG(INFO)   << "iter_count_: " << iter_count_;
  count_ += count; accuracy_ += accuracy; iter_count_++;
  // LOG(INFO) << "accuracy / count: " << accuracy / count;
  if (count == 0) {
    top[0]->mutable_cpu_data()[0] = 1;    
  } else {
    top[0]->mutable_cpu_data()[0] = (Dtype)accuracy / count;    
  }
  if (iter_count_ >= this->layer_param_.accuracy_param().accuracy_iter() ) {
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

INSTANTIATE_CLASS(SequenceAccuracyLayer);
REGISTER_LAYER_CLASS(SequenceAccuracy);

}  // namespace caffe
