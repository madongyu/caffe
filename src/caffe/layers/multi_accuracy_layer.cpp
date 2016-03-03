#include <functional>
#include <utility>
#include <vector>
#include <iostream>

#include "caffe/layers/multi_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
namespace caffe {

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::LayerSetUp(
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
void MultiAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   vector<int> top_shape(0);
   top[0]->Reshape(top_shape);

}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int c = bottom[0]->count();
  int count = 0;
  for ( int i = 0; i < c; i++ ) {
    if ( bottom_data[i] > 0.5  &&  bottom_label[i] ==  1 )
      count++;
    else if (bottom_data[i] <= 0.5 &&  bottom_label[i] == 0 )
      count++;
  }
  top[0]->mutable_cpu_data()[0] = count*1.0/c;

  {
    if (!this->layer_param_.multi_accuracy_param().generate_result_image()) {
      return;
    }
    int batch_num = c/4096;
//    LOG(INFO) << "batch num for test is " << batch_num;
//    int bigNumber = 0;
//    int plusNumber = 0;
//    for ( int i = 0; i < 64; i++ ) {
//      for ( int j = 0; j < 64; j++ ) {
//        std::cout << bottom_data[i*64+j] << " ";
//        if ( j % 8 == 0 ) {
//          std::cout << std::endl;
//        }
//        if (bottom_data[i*64+j] > -1) {
//          bigNumber++;
//        }
//        if (bottom_data[i*64+j] > 1 || bottom_data[i*64+j] < 0) {
//          CHECK_EQ(1,2) << "there is not need for sigmod";
//        }
//
//        if (bottom_label[i*64+j] > 0) {
//          plusNumber++;
//        }
//      }
//    }
//    std::cout << std::endl;
//    LOG(INFO) << "bigNumber is " << bigNumber <<  "plusNumber is " << plusNumber;
//    LOG(INFO) << " vs : " << bigNumber*1.0/4096 << " : " << plusNumber*1.0/4096;


    string fix = this->layer_param_.multi_accuracy_param().result_folder();

    for ( int k = 0; k < batch_num; ++k ) {
      {
        cv::Mat newImg = cv::Mat(64,64,CV_8UC3);
        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
        int i = 0;
        for (; it!= itend; ++it, ++i) {
          (*it)[0] = bottom_data[k*4096+i] >0.5 ? 255 :0;
          (*it)[1] = bottom_data[k*4096+i] >0.5 ? 255 :0;
          (*it)[2] = bottom_data[k*4096+i] >0.5 ? 255 :0;
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
          (*it)[0] = bottom_label[k*4096+i] >0 ? 255 :0;
          (*it)[1] = bottom_label[k*4096+i] >0 ? 255 :0;
          (*it)[2] = bottom_label[k*4096+i] >0 ? 255 :0;
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

INSTANTIATE_CLASS(MultiAccuracyLayer);
REGISTER_LAYER_CLASS(MultiAccuracy);

}  // namespace caffe
