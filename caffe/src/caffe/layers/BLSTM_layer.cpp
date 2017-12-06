#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void BLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {



  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  this->T_ = bottom[0]->shape(0);
  this->N_ = bottom[0]->shape(1);
  LOG(INFO) << "Initializing recurrent layer: assuming input batch contains "
            << this->T_ << " timesteps of " << this->N_ << " independent streams.";

  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(this->T_, bottom[1]->shape(0));
  CHECK_EQ(this->N_, bottom[1]->shape(1));

  // If provided, bottom[2] is a static input to the recurrent net.
  this->static_input_ = (bottom.size() > 2);
  if (this->static_input_) {
    CHECK_GE(bottom[2]->num_axes(), 1);
    CHECK_EQ(this->N_, bottom[2]->shape(0));
  }

  // Create a NetParameter; setup the inputs that aren't unique to particular
  // recurrent architectures.
  NetParameter net_param;
  net_param.set_force_backward(true);

  net_param.add_input("x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  net_param.add_input_shape()->CopyFrom(input_shape);

  input_shape.Clear();
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    input_shape.add_dim(bottom[1]->shape(i));
  }
  net_param.add_input("cont");
  net_param.add_input_shape()->CopyFrom(input_shape);

  if (this->static_input_) {
    input_shape.Clear();
    for (int i = 0; i < bottom[2]->num_axes(); ++i) {
      input_shape.add_dim(bottom[2]->shape(i));
    }
    net_param.add_input("x_static");
    net_param.add_input_shape()->CopyFrom(input_shape);
  }
  // Call the child's FillUnrolledNet implementation to specify the unrolled
  // recurrent architecture.
  this->FillUnrolledNet(&net_param);

  // @Helios: set up forward and backward subnet.
  this->blobs_.clear();
  for (int m = 0; m < 2; ++m) {
    // Prepend this layer's name to the names of each layer in the unrolled net.
    const string& layer_name = this->layer_param_.name();
    if (layer_name.size() > 0) {
      for (int i = 0; i < net_param.layer_size(); ++i) {
        LayerParameter* layer = net_param.mutable_layer(i);
        layer->set_name(layer_name + "_" + layer->name() + '_' + this->int_to_str(m) );
      }
    }

    // Create the unrolled net.
    shared_ptr<Net<Dtype> > subnet(new Net<Dtype>(net_param) );
    unrolled_net_.push_back(subnet);
    unrolled_net_[m]->set_debug_info(
        this->layer_param_.recurrent_param().debug_info());

    // Setup pointers to the inputs.
    x_input_blob_.push_back(
        CHECK_NOTNULL(unrolled_net_[m]->blob_by_name("x").get() ) );
    cont_input_blob_.push_back(
        CHECK_NOTNULL(unrolled_net_[m]->blob_by_name("cont").get() ) );
    if (this->static_input_) {
      x_static_input_blob_.push_back(
          CHECK_NOTNULL(unrolled_net_[m]->blob_by_name("x_static").get() ) );
    }

    // Setup pointers to paired recurrent inputs/outputs.
    vector<string> recur_input_names;
    // h_0 && c_0
    this->RecurrentInputBlobNames(&recur_input_names);
    vector<string> recur_output_names;
    // h_T && c_T
    this->RecurrentOutputBlobNames(&recur_output_names);
    
    const int num_recur_blobs = recur_input_names.size();
    CHECK_EQ(num_recur_blobs, recur_output_names.size());
    recur_input_blobs_.push_back(vector<Blob<Dtype>*>() );
    recur_output_blobs_.push_back(vector<Blob<Dtype>*>() );
    recur_input_blobs_[m].resize(num_recur_blobs);
    recur_output_blobs_[m].resize(num_recur_blobs);
    for (int i = 0; i < recur_input_names.size(); ++i) {
      recur_input_blobs_[m][i] =
          CHECK_NOTNULL(unrolled_net_[m]->blob_by_name(recur_input_names[i]).get() );
      recur_output_blobs_[m][i] =
          CHECK_NOTNULL(unrolled_net_[m]->blob_by_name(recur_output_names[i]).get() );
    }

    // Setup pointers to outputs.
    vector<string> output_names;
    this->OutputBlobNames(&output_names);
    // LOG(INFO) << "recur_output_names: " << recur_output_names[0];
    // LOG(INFO) << "output_names:  "  << output_names[0];
    CHECK_EQ(top.size(), output_names.size())
        << "OutputBlobNames must provide an output blob name for each top.";
    output_blobs_.push_back(vector<Blob<Dtype>*>() );
    output_blobs_[m].resize(output_names.size());
    for (int i = 0; i < output_names.size(); ++i) {
      output_blobs_[m][i] =
          CHECK_NOTNULL(unrolled_net_[m]->blob_by_name(output_names[i]).get());
    }

    // We should have 2 inputs (x and cont), plus a number of recurrent inputs,
    // plus maybe a static input.
    CHECK_EQ(2 + num_recur_blobs + this->static_input_,
             unrolled_net_[m]->input_blobs().size() );

    // This layer's parameters are any parameters in the layers of the unrolled
    // net. We only want one copy of each parameter, so check that the parameter
    // is "owned" by the layer, rather than shared with another.
    for (int i = 0; i < unrolled_net_[m]->params().size(); ++i) {
      if (unrolled_net_[m]->param_owners()[i] == -1) {
        LOG(INFO) << "Adding parameter " << i << ": "
                  << unrolled_net_[m]->param_display_names()[i];
        this->blobs_.push_back(unrolled_net_[m]->params()[i]);
      }
    }
    // Check that param_propagate_down is set for all of the parameters in the
    // unrolled net; set param_propagate_down to true in this layer.
    for (int i = 0; i < unrolled_net_[m]->layers().size(); ++i) {
      for (int j = 0; j < unrolled_net_[m]->layers()[i]->blobs().size(); ++j) {
        CHECK(unrolled_net_[m]->layers()[i]->param_propagate_down(j))
            << "param_propagate_down not set for layer " << i << ", param " << j;
      }
    }

    // Set the diffs of recurrent outputs to 0 -- we can't backpropagate across
    // batches.
    for (int i = 0; i < recur_output_blobs_[m].size(); ++i) {
      caffe_set(recur_output_blobs_[m][i]->count(), Dtype(0),
                recur_output_blobs_[m][i]->mutable_cpu_diff() );
    }    
  }
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // @Helios: set up concate layer. including set up concate input&output blobs
  LayerParameter concat_layer_param;
  concat_layer_param.set_name("output_concat");
  concat_layer_param.set_type("Concat");
  // output_concat_layer.add_top("h");
  concat_layer_param.mutable_concat_param()->set_axis(2);
  concate_layer_ = LayerRegistry<Dtype>::CreateLayer(concat_layer_param);
  concate_ouput_blob_.push_back(new Blob<Dtype>() );  
  concate_iuput_blob_.push_back(output_blobs_[0][0]);
  concate_iuput_blob_.push_back(new Blob<Dtype>() );
  concate_iuput_blob_[1]->ReshapeLike(*output_blobs_[1][0]);
  concate_layer_->SetUp(concate_iuput_blob_, concate_ouput_blob_);
  concate_propagate_down.push_back(true);
  concate_propagate_down.push_back(true);
}


  
template <typename Dtype>
void BLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
  CHECK_EQ(this->T_, bottom[0]->shape(0)) << "input number of timesteps changed";
  this->N_ = bottom[0]->shape(1);
  CHECK_EQ(bottom[1]->num_axes(), 2)
      << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
  CHECK_EQ(this->T_, bottom[1]->shape(0));
  CHECK_EQ(this->N_, bottom[1]->shape(1));
  // CHECK_EQ(top.size(), output_blobs_.size());
  for (int m = 0; m < 2; ++m) {
    x_input_blob_[m]->ReshapeLike(*bottom[0]);
    vector<int> cont_shape = bottom[1]->shape();
    cont_input_blob_[m]->Reshape(cont_shape);
    if (this->static_input_) {
      x_static_input_blob_[m]->ReshapeLike(*bottom[2]);
    }
    vector<BlobShape> recur_input_shapes;
    this->RecurrentInputShapes(&recur_input_shapes);
    CHECK_EQ(recur_input_shapes.size(), recur_input_blobs_[m].size());
    for (int i = 0; i < recur_input_shapes.size(); ++i) {
      recur_input_blobs_[m][i]->Reshape(recur_input_shapes[i]);
    }
    unrolled_net_[m]->Reshape();
    if (m == 0) {
      x_input_blob_[m]->ShareData(*bottom[0]);
      // @Helios: subnets don't share diff with input now, since they need to 
      //          add up their diff and then pass to input blob.
      // x_input_blob_[m]->ShareDiff(*bottom[0]);
      cont_input_blob_[m]->ShareData(*bottom[1]);
    } else {
      // @Helios: backward subnet doesn't share data with input.
      x_input_blob_[m]->ReshapeLike(*bottom[0]);
      cont_input_blob_[m]->ReshapeLike(*bottom[1]);
    }
    if (this->static_input_) {
      x_static_input_blob_[m]->ShareData(*bottom[2]);
      LOG(INFO) << "TODO: here, we don't edit yet!";
      // x_static_input_blob_[m]->ShareDiff(*bottom[2]);
    }  
  }
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*concate_ouput_blob_[i]);
    top[i]->ShareData(*concate_ouput_blob_[i]);
    top[i]->ShareDiff(*concate_ouput_blob_[i]);
  }    
}

template <typename Dtype>
void BLSTMLayer<Dtype>::Reset() {
  // "Reset" the hidden state of the net by zeroing out all recurrent outputs.
  for (int m = 0; m < 2; ++m) {
    for (int i = 0; i < recur_output_blobs_[m].size(); ++i) {
      caffe_set(recur_output_blobs_[m][i]->count(), Dtype(0),
                recur_output_blobs_[m][i]->mutable_cpu_data() );
    }
  }
}

template <typename Dtype>
void ConvertData(const Blob<Dtype>* data, Blob<Dtype>* convert_data) {
  int num = data->num();
  const int nthreads = data->count(1);
  // LOG(INFO) << "data->count(): " << data->count();
  // LOG(INFO) << "nthreads: " << nthreads;
  Dtype* out_data = convert_data->mutable_cpu_data();
  const Dtype* in_data = data->cpu_data();
  for (int i = 0; i < num; ++i) {
    const int in_offset = data->offset(i);
    const int out_offset = data->offset(num - 1 - i);
    for (int j = 0; j < nthreads; ++j) {
      out_data[out_offset+j] = in_data[in_offset+j];
    }
  }
}

template <typename Dtype>
void BLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int m = 0; m < 2; ++m) {
    if (this->phase_ == TEST) {
      unrolled_net_[m]->ShareWeights();
    }

    DCHECK_EQ(recur_input_blobs_[m].size(), recur_output_blobs_[m].size() );
    for (int i = 0; i < recur_input_blobs_[m].size(); ++i) {
      const int count = recur_input_blobs_[m][i]->count();
      DCHECK_EQ(count, recur_output_blobs_[m][i]->count());
      const Dtype* timestep_T_data = recur_output_blobs_[m][i]->cpu_data();
      Dtype* timestep_0_data = recur_input_blobs_[m][i]->mutable_cpu_data();
      caffe_copy(count, timestep_T_data, timestep_0_data);
    }
    // @Helios: in backward subnet, input blob should be reverse.
    if (m == 1) {
      ConvertData(bottom[0], x_input_blob_[m]);
      ConvertData(bottom[1], cont_input_blob_[m]);
      unrolled_net_[m]->ForwardPrefilled();
      ConvertData(output_blobs_[1][0], concate_iuput_blob_[1]);
      // LOG(INFO) << "end";
    } else {
      unrolled_net_[m]->ForwardPrefilled();      
    }
  }
  // @Helios: do forward of concate layer
  concate_layer_->Forward(concate_iuput_blob_, concate_ouput_blob_);
}

template <typename Dtype>
void BLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";

  // TODO: skip backpropagation to inputs and parameters inside the unrolled
  // net according to propagate_down[0] and propagate_down[2]. For now just
  // backprop to inputs and parameters unconditionally, as either the inputs or
  // the parameters do need backward (or Net would have set
  // layer_needs_backward_[i] == false for this layer).
  for (int m = 0; m < 2; ++m) {
    unrolled_net_[m]->Backward();
  }
  // @Helios: need to sum up diff of forward & backward subnet.

}

#ifdef CPU_ONLY
STUB_GPU(BLSTMLayer);
#endif

INSTANTIATE_CLASS(BLSTMLayer);
REGISTER_LAYER_CLASS(BLSTM);

}  // namespace caffe
