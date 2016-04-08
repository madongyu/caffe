#include <algorithm>
#include <vector>

#include "caffe/layers/multi_hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for ( int i = 0; i < num; i++) {
    for ( int j = 0; j <  dim ; j++) {
      if (static_cast<int>(label[i*dim+j]) == 1)
        bottom_diff[i*dim+j] *= -1;
    }
  }
  for (int i = 0; i < num; i++) {
    for ( int j = 0; j < dim; j++) {
      bottom_diff[i*dim+j] = std::max(Dtype(0),1+bottom_diff[i*dim+j]);
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
}

template <typename Dtype>
void MultiHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      for ( int j = 0; j < dim; ++j) {
        if (static_cast<int>(label[i * dim + j]) == 1)
          bottom_diff[i * dim + j] *= -1;
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight * 2 / num, bottom_diff);

  }
}

INSTANTIATE_CLASS(MultiHingeLossLayer);
REGISTER_LAYER_CLASS(MultiHingeLoss);

}  // namespace caffe
