#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/st_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  string prefix = "\t\tSpatial Transformer Layer:: LayerSetUp: \t";

  if(this->layer_param_.st_param().transform_type() == "affine") {
    transform_type_ = "affine";
  } else {
    CHECK(false) << prefix << "Transformation type only supports affine now!" << std::endl;
  }

  if(this->layer_param_.st_param().sampler_type() == "bilinear") {
    sampler_type_ = "bilinear";
  } else {
    CHECK(false) << prefix << "Sampler type only supports bilinear now!" << std::endl;
  }

  if(this->layer_param_.st_param().to_compute_du()) {
    to_compute_dU_ = true;
  }

  // std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;

  output_H_ = bottom[0]->shape(2);
  if(this->layer_param_.st_param().has_output_h()) {
    output_H_ = this->layer_param_.st_param().output_h();
  }
  output_W_ = bottom[0]->shape(3);
  if(this->layer_param_.st_param().has_output_w()) {
    output_W_ = this->layer_param_.st_param().output_w();
  }

  // std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

  // std::cout<<prefix<<"Getting pre-defined parameters"<<std::endl;

  is_pre_defined_theta[0] = false;
  if(this->layer_param_.st_param().has_theta_1_1()) {
    is_pre_defined_theta[0] = true;
    ++ pre_defined_count;
    pre_defined_theta[0] = this->layer_param_.st_param().theta_1_1();
    // std::cout<<prefix<<"Getting pre-defined theta[1][1] = "<<pre_defined_theta[0]<<std::endl;
  }

  is_pre_defined_theta[1] = false;
  if(this->layer_param_.st_param().has_theta_1_2()) {
    is_pre_defined_theta[1] = true;
    ++ pre_defined_count;
    pre_defined_theta[1] = this->layer_param_.st_param().theta_1_2();
    // std::cout<<prefix<<"Getting pre-defined theta[1][2] = "<<pre_defined_theta[1]<<std::endl;
  }

  is_pre_defined_theta[2] = false;
  if(this->layer_param_.st_param().has_theta_1_3()) {
    is_pre_defined_theta[2] = true;
    ++ pre_defined_count;
    pre_defined_theta[2] = this->layer_param_.st_param().theta_1_3();
    // std::cout<<prefix<<"Getting pre-defined theta[1][3] = "<<pre_defined_theta[2]<<std::endl;
  }

  is_pre_defined_theta[3] = false;
  if(this->layer_param_.st_param().has_theta_2_1()) {
    is_pre_defined_theta[3] = true;
    ++ pre_defined_count;
    pre_defined_theta[3] = this->layer_param_.st_param().theta_2_1();
    // std::cout<<prefix<<"Getting pre-defined theta[2][1] = "<<pre_defined_theta[3]<<std::endl;
  }

  is_pre_defined_theta[4] = false;
  if(this->layer_param_.st_param().has_theta_2_2()) {
    is_pre_defined_theta[4] = true;
    ++ pre_defined_count;
    pre_defined_theta[4] = this->layer_param_.st_param().theta_2_2();
    // std::cout<<prefix<<"Getting pre-defined theta[2][2] = "<<pre_defined_theta[4]<<std::endl;
  }

  is_pre_defined_theta[5] = false;
  if(this->layer_param_.st_param().has_theta_2_3()) {
    is_pre_defined_theta[5] = true;
    ++ pre_defined_count;
    pre_defined_theta[5] = this->layer_param_.st_param().theta_2_3();
    // std::cout<<prefix<<"Getting pre-defined theta[2][3] = "<<pre_defined_theta[5]<<std::endl;
  }

  // check the validation for the parameter theta
  // CHECK(bottom[1]->count(1) + pre_defined_count == 6) << "The dimension of theta is not six!"
  //     << " Only " << bottom[1]->count(1) << " + " << pre_defined_count << std::endl;
  // CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << "The first dimension of theta and " <<
  //     "U should be the same (batch size)" << std::endl;

  // initialize the matrix for output grid (x_i, y_i, 1.0)
  // the same matrix [-1,1] by [-1,1] for all batch elements
  // std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

  vector<int> shape_output(2);
  shape_output[0] = output_H_ * output_W_;
  shape_output[1] = 3;
  output_grid.Reshape(shape_output);

  Dtype* data = output_grid.mutable_cpu_data();
  for(int i=0; i< output_H_ * output_W_; ++i) {
    data[3 * i] = (i / output_W_) * 1.0 / output_H_ * 2 - 1;
    data[3 * i + 1] = (i % output_W_) * 1.0 / output_W_ * 2 - 1;
    data[3 * i + 2] = 1;
  }

  // initialize the matrix for input grid
  // the input grid gives, for (i,j) in the output, the corresponding position (projection)
  // on the input
  // std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

  vector<int> shape_input(3);
  shape_input[0] = bottom[0]->shape(0);
  shape_input[1] = output_H_ * output_W_;
  shape_input[2] = 2;
  input_grid.Reshape(shape_input);

  // std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  string prefix = "\t\tSpatial Transformer Layer:: Reshape: \t";

  // if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

  N = bottom[0]->shape(0);
  C = bottom[0]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);

  // reshape V
  vector<int> shape(4);

  ////// CAREFUL : does not take st_param().has_output_h() into consideration

  shape[0] = N;
  shape[1] = C;
  shape[2] = output_H_;
  shape[3] = output_W_;

  top[0]->Reshape(shape);

  // reshape dTheta_tmp
  vector<int> dTheta_tmp_shape(4);

  dTheta_tmp_shape[0] = N;
  dTheta_tmp_shape[1] = 2;
  dTheta_tmp_shape[2] = 3;
  dTheta_tmp_shape[3] = output_H_ * output_W_ * C;

  dTheta_tmp.Reshape(dTheta_tmp_shape);

  // init all_ones_2
  vector<int> all_ones_2_shape(1);
  all_ones_2_shape[0] = output_H_ * output_W_ * C;
  all_ones_2.Reshape(all_ones_2_shape);

  // reshape full_theta
  vector<int> full_theta_shape(2);
  full_theta_shape[0] = N;
  full_theta_shape[1] = 6;
  full_theta.Reshape(full_theta_shape);

  // if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
Dtype SpatialTransformerLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {

  // bool debug = false;
  string prefix = "\t\tSpatial Transformer Layer:: transform_forward_cpu: \t";
  // if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
  // if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

  Dtype res = (Dtype)0.;

  // calcul de x,y (position dans l'image)
  Dtype x = (px + 1) / 2 * H;
  Dtype y = (py + 1) / 2 * W;
  // if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

  for(int m = floor(x); m <= ceil(x); ++m)
    for(int n = floor(y); n <= ceil(y); ++n) {
      // if(debug) std::cout<<prefix<<"- (m, n) = ("<<m<<", "<<n<<")"<<std::endl;
      if(m >= 0 && m < H && n >= 0 && n < W) {
        res += (1 - abs(x - m)) * (1 - abs(y - n)) * pic[m * W + n];
        // if(debug) std::cout<<prefix<<" pic[m, n] = "<<pic[m * W + n]<<std::endl;
      }
    }

  // if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

  return res;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  string prefix = "\t\tSpatial Transformer Layer:: Forward_cpu: \t";

  // CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
  //    " CHECK in st_layer.cpp file. Line number: 240-241." << std::endl;

  // if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

  //input layers : U and theta
  const Dtype* U = bottom[0]->cpu_data();
  const Dtype* theta = bottom[1]->cpu_data();

  Dtype* full_theta_data = full_theta.mutable_cpu_data();

  // input_grid_data : pour chaque pixel de chaque image, px et py (coordonnées du pixel dans l'image input)
  Dtype* input_grid_data = input_grid.mutable_cpu_data();

  const Dtype* output_grid_data = output_grid.cpu_data();

  //output layer : V
  Dtype* V = top[0]->mutable_cpu_data();

  // intialize mutable_cpu_data arrays
  caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
  caffe_set(top[0]->count(), (Dtype)0, V);

  // compute full_theta
  // if(global_debug) std::cout<<prefix<<"compute full theta"<< std::endl;
  int k = 0;
  for(int i=0; i<6; ++i) {
    if(is_pre_defined_theta[i]) {
      for(int j = 0; j < N ; j++)
        full_theta_data[full_theta.offset(j,i)] = pre_defined_theta[i];
      // if(global_debug) std::cout<<prefix<<"assigning predefined"<<pre_defined_theta[i] << std::endl;
    } else {
      for(int j = 0; j < N ; j++){
        full_theta_data[full_theta.offset(j,i)] = theta[bottom[1]->offset(j,k)];
        // if(global_debug) std::cout<<prefix<<"assigning "<<theta[bottom[1]->offset(j,k)] << std::endl;
      }
      ++ k;
    }
  }

  // for each input in the batch
  for(int i = 0; i < N; ++i) {

    Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;

    // Matrix multiplication : coordinates = output_grid_data x theta
    // output_grid_data shape : (output_H_ * output_W_) x 3 => (x_i, y_i, 1)
    // transposed theta shape : 2 x 3
    // (theta shape : 3 x 2)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, 3, (Dtype)1.,
          output_grid_data, full_theta_data + 6 * i, (Dtype)0., coordinates);

    int row_idx; Dtype px, py;

    // parcours des pixels de l'output V
    for(int j = 0; j < C; ++j)
      for(int s = 0; s < output_H_; ++s)
        for(int t = 0; t < output_W_; ++t) {

          row_idx = output_W_ * s + t;

          px = coordinates[row_idx * 2];
          py = coordinates[row_idx * 2 + 1];
// Dtype x = (px + 1) / 2 * H;
//// std::cout<<prefix<<"Hum."<< px << ","<< py <<std::endl;

          V[top[0]->offset(i, j, s, t)] = transform_forward_cpu(
              U + bottom[0]->offset(i, j, 0, 0), px, py);
        }
  }

  // // if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
    const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {

  // bool debug = false;
  string prefix = "\t\tSpatial Transformer Layer:: transform_backward_cpu: \t";
  // // if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

  // position (x,y)
  Dtype x = (px + 1) / 2 * H;
  Dtype y = (py + 1) / 2 * W;
  // if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

  for(int m = floor(x); m <= ceil(x); ++m)
    for(int n = floor(y); n <= ceil(y); ++n) {
      // if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;
      if(m >= 0 && m < H && n >= 0 && n < W) {
        dU[m * W + n] += dV * (1 - abs(x - m)) * (1 - abs(y - n));
        dpx += caffe_sign<Dtype>(m - x) * (1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
        dpy += caffe_sign<Dtype>(n - y) * (1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
        // if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
      }
    }

  // if(debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    string prefix = "\t\tSpatial Transformer Layer:: Backward_cpu: \t";

//    CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
//        " CHECK in st_layer.cpp file. Line number: 420-421." << std::endl;

    // if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

    const Dtype* dV = top[0]->cpu_diff();
    const Dtype* input_grid_data = input_grid.cpu_data();
    const Dtype* U = bottom[0]->cpu_data();

    Dtype* dU = bottom[0]->mutable_cpu_diff();
    Dtype* dTheta = bottom[1]->mutable_cpu_diff();
    Dtype* input_grid_diff = input_grid.mutable_cpu_diff();

    caffe_set(bottom[0]->count(), (Dtype)0, dU);
    caffe_set(bottom[1]->count(), (Dtype)0, dTheta);
    caffe_set(input_grid.count(), (Dtype)0, input_grid_diff);

    // for each image in batch
    for(int i = 0; i < N; ++i) {

      // gets projection coordinate for each input
      const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
      Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;

      int row_idx; Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;

      for(int s = 0; s < output_H_; ++s)
        for(int t = 0; t < output_W_; ++t) {

          row_idx = output_W_ * s + t;

          px = coordinates[row_idx * 2];
          py = coordinates[row_idx * 2 + 1];

          for(int j = 0; j < C; ++j) {

            delta_dpx = delta_dpy = (Dtype)0.;

            transform_backward_cpu(dV[top[0]->offset(i, j, s, t)], U + bottom[0]->offset(i, j, 0, 0),
                px, py, dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);

            coordinates_diff[row_idx * 2] += delta_dpx;
            coordinates_diff[row_idx * 2 + 1] += delta_dpy;
          }

          dpx = coordinates_diff[row_idx * 2];
          dpy = coordinates_diff[row_idx * 2 + 1];

          dTheta[6 * i + 0] += dpx * (s * 1.0 / output_H_ * 2 - 1);
          dTheta[6 * i + 1] += dpx * (t * 1.0 / output_W_ * 2 - 1);
          dTheta[6 * i + 2] += dpx;
          dTheta[6 * i + 3] += dpy * (s * 1.0 / output_H_ * 2 - 1);
          dTheta[6 * i + 4] += dpy * (t * 1.0 / output_W_ * 2 - 1);
          dTheta[6 * i + 5] += dpy;
        }
    }

    // if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

}  // namespace caffe
