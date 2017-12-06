#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/loc_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void LocLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tLoc Loss Layer:: LayerSetUp: \t";

	threshold = (Dtype) this->layer_param_.loc_loss_param().threshold();
	std::cout<<prefix<<"Getting threshold value = "<<threshold<<std::endl;

	CHECK(threshold > 0) << "Error: threshold should be larger than zero.";
}

template <typename Dtype>
void LocLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	vector<int> tot_loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(tot_loss_shape);

	N = bottom[0]->count();

	vector<int> loss_shape(1);
	loss_shape[0] = N;
	loss_.Reshape(loss_shape);
}

template <typename Dtype>
void LocLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

template <typename Dtype>
void LocLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(LocLossLayer);
#endif

INSTANTIATE_CLASS(LocLossLayer);
REGISTER_LAYER_CLASS(LocLoss);

}  // namespace caffe
