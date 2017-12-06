#ifndef CAFFE_CASIA_DATA_LAYER_HPP_
#define CAFFE_CASIA_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class DataCasiaLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataCasiaLayer(const LayerParameter& param);
  virtual ~DataCasiaLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "DataCasia"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return -1; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReaderSeq reader_;
};

}  // namespace caffe

#endif  // CAFFE_CASIA_DATA_LAYER_HPP_
