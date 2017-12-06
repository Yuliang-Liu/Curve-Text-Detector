#ifndef ST_LOSS_LAYERS_HPP_
#define ST_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* Input: theta
 * Output: loss, one single value
 *
 * This loss layer tends to force the crops to be in the range
 * of image space when performing spatial transformation
 */

template <typename Dtype>
class STLossLayer : public LossLayer<Dtype> {
public:
  explicit STLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "STLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  Blob<Dtype> loss_;
  Blob<Dtype> dtheta_tmp_;
  Blob<Dtype> all_ones_vec_;

  int N, output_H_, output_W_;
};

}  // namespace caffe

#endif  // ST_LOSS_LAYERS_HPP_
