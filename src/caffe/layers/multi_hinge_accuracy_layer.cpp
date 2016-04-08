#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_hinge_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiHingeAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 }

template <typename Dtype>
void MultiHingeAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiHingeAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype accuracy = 0;
  for (int i = 0; i < num; i++) {
      for (int j = 0; j < dim; j++) {
          if ((static_cast<int>(label[i * dim + j]) == 1)
                  && bottom_data[i * dim + j] > 0)
              accuracy++;
          if ((static_cast<int>(label[i * dim + j]) == 0)
                  && bottom_data[i * dim + j] <= 0)
              accuracy++;
      }
  }
  top[0]->mutable_cpu_data()[0] = 1.0*accuracy / count;

}

INSTANTIATE_CLASS(MultiHingeAccuracyLayer);
REGISTER_LAYER_CLASS(MultiHingeAccuracy);

}  // namespace caffe
