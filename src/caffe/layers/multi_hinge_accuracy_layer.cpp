#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_hinge_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void MultiHingeAccuracyLayer<Dtype>::LayerSetUp(
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
  {
    if (!this->layer_param_.multi_accuracy_param().generate_result_image()) {
      return;
    }

    const int resultSize=this->layer_param_.multi_accuracy_param().result_size();
    int batch_num = count/resultSize;

    string fix = this->layer_param_.multi_accuracy_param().result_folder();

    for ( int k = 0; k < batch_num; ++k ) {
      {
        cv::Mat newImg = cv::Mat(64,64,CV_8UC3);
        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
        int i = 0;
        for (; it!= itend; ++it, ++i) {
          int predict_pixel = 0;
          if (bottom_data[k*resultSize+i] <= -1 ) {
            predict_pixel = 0;
          } else if (bottom_data[k*resultSize+i] >= 1 ) {
            predict_pixel = 255;
          } else {
            predict_pixel = static_cast<uint8_t>((1
                + bottom_data[k*resultSize+i]) * 255 / 2);
          }
          (*it)[0] = predict_pixel;
          (*it)[1] = predict_pixel;
          (*it)[2] = predict_pixel;
        }
        string name = fix  + lines_[lines_id_].first;
        cv::imwrite( name.c_str(), newImg );
      }
      {
        cv::Mat newImg = cv::Mat(64,64,CV_8UC3);
        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
        int i = 0;
        for (; it!= itend; ++it, ++i) {
          (*it)[0] = label[k*resultSize+i] >0 ? 255 :0;
          (*it)[1] = label[k*resultSize+i] >0 ? 255 :0;
          (*it)[2] = label[k*resultSize+i] >0 ? 255 :0;
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

INSTANTIATE_CLASS(MultiHingeAccuracyLayer);
REGISTER_LAYER_CLASS(MultiHingeAccuracy);

}  // namespace caffe
