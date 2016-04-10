#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_hinge_sl_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include <sstream>
std::string axis2str3(int x1,int y1,int x2, int y2) {
  std::stringstream ss;

  ss << "_" << x1
      << "_" << y1
      << "_" << x2
      << "_" << y2;

  return ss.str();
}



namespace caffe {

template <typename Dtype>
void MultiHingeSlAccuracyLayer<Dtype>::LayerSetUp(
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
void MultiHingeSlAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiHingeSlAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

        string id;
        for ( int bike = 0; bike < lines_[lines_id_].labelName.size(); bike++ ) {
          if ( lines_[lines_id_].labelName[bike] != '.' ) {
            id = id + lines_[lines_id_].labelName[bike];
          } else {
            break;
          }
        }
        string name = fix  + id + axis2str3(lines_[lines_id_].x1,
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

INSTANTIATE_CLASS(MultiHingeSlAccuracyLayer);
REGISTER_LAYER_CLASS(MultiHingeSlAccuracy);

}  // namespace caffe
