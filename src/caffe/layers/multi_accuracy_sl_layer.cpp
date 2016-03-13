#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <cmath>

#include "caffe/layers/multi_accuracy_sl_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include <sstream>
std::string axis2str2(int x1,int y1,int x2, int y2) {
  std::stringstream ss;

  ss << "_" << x1
      << "_" << y1
      << "_" << x2
      << "_" << y2;

  return ss.str();
}



namespace caffe {

template <typename Dtype>
void MultiAccuracySlLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (!this->layer_param_.multi_accuracy_param().generate_result_image()) {
    return;
  }
  const string& source = this->layer_param_.multi_accuracy_param().source_folder();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label;
  int x1,y1,x2,y2;
  while (infile >> filename >> label >> x1 >> y1 >> x2 >> y2) {
    lines_.push_back(SelectiveData(filename, label, x1, y1, x2, y2));
  }
  infile.close();
  lines_id_ = 0;
  LOG(INFO) << "Opening file " << source << source;
  LOG(INFO) << "test source size is  " << lines_.size();

}

template <typename Dtype>
void MultiAccuracySlLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   vector<int> top_shape(0);
   top[0]->Reshape(top_shape);

}

template <typename Dtype>
void MultiAccuracySlLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
//      {
//        cv::Mat newImg = cv::Mat(64,64,CV_8UC3);
//        cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
//        cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
//        int i = 0;
//        for (; it!= itend; ++it, ++i) {
//          (*it)[0] = bottom_data[k*4096+i] >0.5 ? 255 :0;
//          (*it)[1] = bottom_data[k*4096+i] >0.5 ? 255 :0;
//          (*it)[2] = bottom_data[k*4096+i] >0.5 ? 255 :0;
//        }
//        string name = fix  + lines_[lines_id_].dataName;
//        cv::imwrite( name.c_str(), newImg );
//      }
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
        string id;
        for ( int bike = 0; bike < lines_[lines_id_].labelName.size(); bike++ ) {
          if ( lines_[lines_id_].labelName[bike] != '.' ) {
            id = id + lines_[lines_id_].labelName[bike];
          } else {
            break;
          }
        }
        string name = fix  + id + axis2str2(lines_[lines_id_].x1,
            lines_[lines_id_].y1,lines_[lines_id_].x2,lines_[lines_id_].y2) + ".png";
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

INSTANTIATE_CLASS(MultiAccuracySlLayer);
REGISTER_LAYER_CLASS(MultiAccuracySl);

}  // namespace caffe
