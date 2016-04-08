#ifndef CAFFE_MULTI_HINGE_LOSS_LAYER_HPP_
#define CAFFE_MULTI_HINGE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultiHingeLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiHingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "MultiHingeLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};


}  // namespace caffe

#endif  // CAFFE_HINGE_LOSS_LAYER_HPP_
