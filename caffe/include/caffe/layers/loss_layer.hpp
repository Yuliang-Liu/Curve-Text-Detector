#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
// #include "caffe/util/ctc.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};
/**
 * @brief Computes the CTC loss @f$
 *          
 */
// template <typename Dtype>
// class TranscriptionLossLayer : public LossLayer<Dtype> {
//  public:
//   explicit TranscriptionLossLayer(const LayerParameter& param)
//       : LossLayer<Dtype>(param) {}
//   virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
//   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
//   virtual inline int ExactNumBottomBlobs() const { return 2; }
//   virtual inline const char* type() const { return "CTCloss"; }


//  protected:
//   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);

//   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);

//   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//   /// The internal SoftmaxLayer used to map predictions to a distribution.
//   shared_ptr<Layer<Dtype> > softmax_layer_;
//   vector<Blob<Dtype>*> softmax_bottom_vec_;
//   /// top vector holder used in call to the underlying SoftmaxLayer::Forward
//   vector<Blob<Dtype>*> softmax_top_vec_;
//   /// prob stores the output probability predictions from the SoftmaxLayer.
//   Blob<Dtype> prob_;
//   Blob<Dtype> input_;

//   /// prob stores the gradient from forward step.
//   Blob<Dtype> gradInput_;
//   /// Whether to ignore instances with a certain label.
//   bool has_ignore_label_;
//   /// The label indicating that an instance should be ignored.
//   int ignore_label_;
//   // @Helios: ignore instances number
//   int ignore_label_num_;  
//   /// Whether to normalize the loss by the total number of values present
//   /// (otherwise just by the batch size).
//   bool normalize_;

//   int softmax_axis_, outer_num_, inner_num_;

//   // @Helios: variables for forward&backward_gpu.
//   // cudaStream_t stream_;
//   ctcOptions info_;
  
// };

// /**
//  * @brief Computes the CTC loss @f$
//  *          
//  */
// template <typename Dtype>
// class TranscriptionLossSingleLayer : public LossLayer<Dtype> {
//  public:
//   explicit TranscriptionLossSingleLayer(const LayerParameter& param)
//       : LossLayer<Dtype>(param) {}
//   virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
//   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
//   virtual inline int ExactNumBottomBlobs() const { return 2; }
//   virtual inline const char* type() const { return "CTCloss"; }


//  protected:
//   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);

//   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


//   /// The internal SoftmaxLayer used to map predictions to a distribution.
//   shared_ptr<Layer<Dtype> > softmax_layer_;
//   vector<Blob<Dtype>*> softmax_bottom_vec_;
//   /// top vector holder used in call to the underlying SoftmaxLayer::Forward
//   vector<Blob<Dtype>*> softmax_top_vec_;
//   /// prob stores the output probability predictions from the SoftmaxLayer.
//   Blob<Dtype> prob_;
//   Blob<Dtype> input_;

//   /// prob stores the gradient from forward step.
//   Blob<Dtype> gradInput;
//   /// Whether to ignore instances with a certain label.
//   bool has_ignore_label_;
//   /// The label indicating that an instance should be ignored.
//   int ignore_label_;
//   // @Helios: ignore instances number
//   int ignore_label_num_;  
//   /// Whether to normalize the loss by the total number of values present
//   /// (otherwise just by the batch size).
//   bool normalize_;

//   int softmax_axis_, outer_num_, inner_num_;
// };
}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
