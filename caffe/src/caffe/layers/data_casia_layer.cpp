#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_casia_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>
namespace caffe {

template <typename Dtype>
DataCasiaLayer<Dtype>::DataCasiaLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataCasiaLayer<Dtype>::~DataCasiaLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataCasiaLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // the top blob is initilized according to layer parameters.
  int nIteratedIntegrals = this->layer_param_.transform_param().n_iterated_integrals();
  vector<int> top_shape;
  top_shape.push_back(1);
  if (this->layer_param_.transform_param().using_online() ) {
    top_shape.push_back((2<<nIteratedIntegrals));    
  } else {
    top_shape.push_back((2<<nIteratedIntegrals)-1);    
  }
  if (this->layer_param_.transform_param().new_height() ) {
    top_shape.push_back(this->layer_param_.transform_param().new_height());
    top_shape.push_back(this->layer_param_.transform_param().new_width());
    this->transformed_data_ .Reshape(top_shape);

    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }    
  } else if(this->layer_param_.transform_param().reshape_width() ){
    // 

  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape;
    label_shape.push_back(this->layer_param_.data_param().seq_length());
    label_shape.push_back(batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
      this->prefetch_[i].multi_label_.Reshape(label_shape);      
    }
    if (this->layer_param_.data_param().use_multi_label() ) {
      // label_shape[0] = batch_size;
      // label_shape[1] = 1;
      top[2]->Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataCasiaLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();

  // all the datum is required to be the same size.
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  caffe_set(batch->data_.count(), Dtype(0), top_data);

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    DatumSeq& datumSeqTmp = *(reader_.full().pop("Waiting for data"));

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if(this->layer_param_.transform_param().data_style() == "ordinary_image"){
    	this->data_transformer_->Transform(datumSeqTmp, &(this->transformed_data_));
    }

    // Copy label.
    if (this->output_labels_) {
      if (this->layer_param_.data_param().seq_length() < datumSeqTmp.label_size() ) {
        LOG(FATAL) << "err! " << datumSeqTmp.label_size();
      }
      for(int lv = 0; lv < this->layer_param_.data_param().seq_length(); lv++) {
        if (this->layer_param_.data_param().use_float_label() ) {
          if(lv < datumSeqTmp.float_label_size()) {
            top_label[lv*batch_size+item_id] = datumSeqTmp.float_label(lv);
              // LOG(INFO) << "top_label[lv*batch_size+item_id]: " << top_label[lv*batch_size+item_id];
            } else {
              top_label[lv*batch_size+item_id] = 0;
            }
        } else {
          // if (lv < datumSeqTmp.label_size()) {
          //   top_label[lv*batch_size+item_id] = datumSeqTmp.label(lv);
          // } else if (lv ==  datumSeqTmp.label_size() ) {
          //   top_label[lv*batch_size+item_id] = 0;
          // } else {
          //   top_label[lv*batch_size+item_id] = -1;
          // }
          if (lv < datumSeqTmp.label_size()) {
            top_label[lv*batch_size+item_id] = datumSeqTmp.label(lv);
          } else {
            top_label[lv*batch_size+item_id] = 0;
          }          
        }
       }
      // for(int lv = 0; lv < this->layer_param_.data_param().seq_length(); lv++){
      //   LOG(INFO) << "top_label[lv*batch_size+item_id]: " << top_label[lv*batch_size+item_id];
      // }

      if (this->layer_param_.data_param().use_multi_label() ) {
        Dtype* top_multi_label = batch->multi_label_.mutable_cpu_data();
        // top_multi_label[item_id] = datumSeqTmp.multi_task_label();
        for(int lv = 0; lv < this->layer_param_.data_param().seq_length(); lv++) {
          if(lv < datumSeqTmp.multi_task_label_size() ) {
            top_multi_label[lv*batch_size+item_id] = datumSeqTmp.multi_task_label(lv);
          } else {
            top_multi_label[lv*batch_size+item_id] = 0;
          }        
        }
      }

    }  

    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<DatumSeq*>(&datumSeqTmp));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataCasiaLayer);
REGISTER_LAYER_CLASS(DataCasia);

}  // namespace caffe
