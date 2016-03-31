#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <cmath>

#include "caffe/layers/multi_accuracy_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void MultiAccuracyMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (!this->layer_param_.multi_accuracy_param().generate_result_image()) {
    return;
  }
  const string& source = this->layer_param_.multi_accuracy_param().source_folder();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }
  infile.close();
  lines_id_ = 0;
  LOG(INFO) << "Opening file " << source << source;
  LOG(INFO) << "test source size is  " << lines_.size();

}

template <typename Dtype>
void MultiAccuracyMaskLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);

}

template <typename Dtype>
void MultiAccuracyMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int c = bottom[0]->count();
  int count = 0;
  for ( int i = 0; i < c; i++ ) {
    if ( bottom_data[i] > 0.5  &&  std::fabs(bottom_label[i]-1) < 1e-5 )
      count++;
    else if (bottom_data[i] <= 0.5 && std::fabs(bottom_label[i]-0) < 1e-5)
      count++;
  }
  top[0]->mutable_cpu_data()[0] = count*1.0/c;

  {
    if (!this->layer_param_.multi_accuracy_param().generate_result_image()) {
      return;
    }
    const int maskSize=128*128;
    int batch_num = c/maskSize;


    string fix = this->layer_param_.multi_accuracy_param().result_folder();

    for ( int k = 0; k < batch_num; ++k ) {
      {
        cv::Mat newImg = cv::Mat(128,128,CV_8UC3);
        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
        int i = 0;
        if (!this->layer_param_.multi_accuracy_param().generate_score_image()) {
          for (; it!= itend; ++it, ++i) {
            (*it)[0] = bottom_data[k*maskSize+i] >0.5 ? 255 :0;
            (*it)[1] = bottom_data[k*maskSize+i] >0.5 ? 255 :0;
            (*it)[2] = bottom_data[k*maskSize+i] >0.5 ? 255 :0;
          }
        } else {
          for (; it!= itend; ++it, ++i) {
            (*it)[0] = bottom_data[k*maskSize+i] * 255;
            (*it)[1] = bottom_data[k*maskSize+i] * 255;
            (*it)[2] = bottom_data[k*maskSize+i] * 255;
          }
        }
        string name = fix  + lines_[lines_id_].first;
        cv::imwrite( name.c_str(), newImg );
      }
      {
        cv::Mat newImg = cv::Mat(128,128,CV_8UC3);
        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
        int i = 0;
        for (; it!= itend; ++it, ++i) {
          (*it)[0] = bottom_label[k*maskSize+i] >0 ? 255 :0;
          (*it)[1] = bottom_label[k*maskSize+i] >0 ? 255 :0;
          (*it)[2] = bottom_label[k*maskSize+i] >0 ? 255 :0;
        }
        string name = fix  + lines_[lines_id_].second;
        cv::imwrite( name.c_str(), newImg );
      }

      ++lines_id_;
      if ( lines_id_ == lines_.size() ) {
        lines_id_ = 0;
        LOG(INFO) << "forward to file tail begin from top";
      }
    }
  }
}

INSTANTIATE_CLASS(MultiAccuracyMaskLayer);
REGISTER_LAYER_CLASS(MultiAccuracyMask);

}  // namespace caffe
