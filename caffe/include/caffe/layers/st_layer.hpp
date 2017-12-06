#ifndef ST_LAYER_HPP_
#define ST_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {

public:
	explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
	      to_compute_dU_ = false; 
	      global_debug = false; 
	      pre_defined_count = 0;
      }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SpatialTransformer"; }
	virtual inline int ExactNumBottomBlobs() const { return -1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

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
	inline Dtype abs(Dtype x) {
		if(x < 0) return -x; return x;
	}
	inline Dtype max(Dtype x, Dtype y) {
		if(x < y) return y; return x;
	}

	Dtype transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
	void transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
			const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);

	string transform_type_;
	string sampler_type_;

	int output_H_;
	int output_W_;

	int N, C, H, W;

	bool global_debug;
	bool to_compute_dU_;

	Blob<Dtype> dTheta_tmp;	// used for back propagation part in GPU implementation
	Blob<Dtype> all_ones_2;	// used for back propagation part in GPU implementation

	Blob<Dtype> full_theta;	// used for storing data and diff for full six-dim theta
	Dtype pre_defined_theta[6];
	bool is_pre_defined_theta[6];
	int pre_defined_count;

	Blob<Dtype> output_grid;	// standard output coordinate system, [0, 1) by [0, 1).
	Blob<Dtype> input_grid;	// corresponding coordinate on input image after projection for each output pixel.
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
