#ifndef CAFFE_MULTI_DATA_SL_LAYER_HPP_
#define CAFFE_MULTI_DATA_SL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultiDataSlLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
  explicit MultiDataSlLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiDataSlLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiDataSl"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  class SelectiveData {
    public :
    std::string dataName;
    std::string labelName;
    int x1,y1,x2,y2;
    SelectiveData(std::string jpg,std::string png, int x1_,int y1_,int x2_,int y2_):
      dataName(jpg),labelName(png),x1(x1_),y1(y1_),x2(x2_),y2(y2_){
    }
  };

  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector< SelectiveData > lines_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
