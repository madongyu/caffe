#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multi_data_sl_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"



#include <sstream>
std::string axis2str(int x1,int y1,int x2, int y2) {
  std::stringstream ss;

  ss << "_" << x1
      << "_" << y1
      << "_" << x2
      << "_" << y2;

  return ss.str();
}



namespace caffe {

template <typename Dtype>
MultiDataSlLayer<Dtype>::~MultiDataSlLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiDataSlLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.multi_data_param().new_height();
  const int new_width  = this->layer_param_.multi_data_param().new_width();
  string root_folder = this->layer_param_.multi_data_param().root_folder();
  bool center_flag = this->layer_param_.multi_data_param().center_flag();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
          "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multi_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string fileName;
  string label;
  int x1, y1,x2,y2;
  while (infile >> fileName >> label >> x1 >> y1 >> x2 >> y2) {
    lines_.push_back(SelectiveData(fileName, label, x1, y1, x2, y2));
  }
  infile.close();

  if (this->layer_param_.multi_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.multi_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.multi_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img_origin = ReadImageToCVMat(root_folder + lines_[lines_id_].dataName);
  CHECK(cv_img_origin.data) << "Could not load " << lines_[lines_id_].dataName;

  x1 = lines_[lines_id_].x1;
  y1 = lines_[lines_id_].y1;
  x2 = lines_[lines_id_].x2;
  y2 = lines_[lines_id_].y2;


  // left up point (x,y), right down point(x+h,y+w)
  int h = x2-x1;
  int w = y2-y1;
  CHECK_GT(w,0) << "width must > 0 ";
  CHECK_GT(h,0) << "height must > 0 ";

  cv::Rect rect(y1,x1,w,h);
  cv::Mat sub_img = cv::Mat(cv_img_origin, rect);
  LOG(INFO) << "sub_img not error ";

  cv::Mat cv_img;
  cv::resize(sub_img, cv_img, cv::Size(new_width, new_height));

  vector<int> top_shape;
  if (center_flag) {

    // Use data_transformer to infer the expected blob shape from a cv_image.

    std::vector<cv::Mat> channels(3);
    split(cv_img, channels);

    cv::Mat biasImg = cv::Mat(256,256,CV_8UC1);
    cv::Mat_<char>::iterator bias_it= biasImg.begin<char>();
    cv::Mat_<char>::iterator bias_itend= biasImg.end<char>();

    for (int h = 0; h < biasImg.rows; ++h) {
      for (int w = 0; w < biasImg.cols; ++w) {
        if ( bias_it == bias_itend ) {
          CHECK_EQ(1,2) << "Img out of dimension";
        }
        float center_bias = (h - 127.5) * (h - 127.5)
                                           + (w - 127.5) * (w - 127.5);
        float tmp = 255 * std::exp(-center_bias / 10000);
        (*bias_it) = static_cast<char>(tmp);
        bias_it++;
      }
    }
    channels.push_back(biasImg);
    cv::Mat fourImg;
    cv::merge(channels, fourImg);
    top_shape = this->data_transformer_->InferBlobShape(fourImg);

  } else {
    top_shape = this->data_transformer_->InferBlobShape(cv_img);

  }

  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.multi_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label reshape?
  vector<int> label_shape;
  label_shape.resize(2);
  label_shape[0] = batch_size;
  label_shape[1] = 4096;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void MultiDataSlLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiDataSlLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  MultiDataParameter multi_data_param = this->layer_param_.multi_data_param();
  const int batch_size = multi_data_param.batch_size();
  const int new_height = multi_data_param.new_height();
  const int new_width = multi_data_param.new_width();
  string root_folder = multi_data_param.root_folder();
  bool center_flag = multi_data_param.center_flag();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img_origin = ReadImageToCVMat(root_folder + lines_[lines_id_].dataName);
  CHECK(cv_img_origin.data) << "Could not load " << lines_[lines_id_].dataName;
  // Use data_transformer to infer the expected blob shape from a cv_img.

  int x1,y1,x2,y2;
  x1 = lines_[lines_id_].x1;
  y1 = lines_[lines_id_].y1;
  x2 = lines_[lines_id_].x2;
  y2 = lines_[lines_id_].y2;

  int h = x2-x1;
  int w = y2-y1;
  CHECK_GT(w,0) << "width must > 0 ";
  CHECK_GT(h,0) << "height must > 0 ";

  cv::Rect rect(y1,x1,w,h);
  cv::Mat sub_img = cv::Mat(cv_img_origin, rect);
  cv::Mat cv_img;
  cv::resize(sub_img, cv_img, cv::Size(new_width, new_height));

  vector<int> top_shape;

  if (center_flag) {
    std::vector<cv::Mat> channels(3);
    split(cv_img, channels);

    cv::Mat biasImg = cv::Mat(256,256,CV_8UC1);
    cv::Mat_<char>::iterator bias_it= biasImg.begin<char>();
    cv::Mat_<char>::iterator bias_itend= biasImg.end<char>();

    for (int h = 0; h < biasImg.rows; ++h) {
      for (int w = 0; w < biasImg.cols; ++w) {
        if ( bias_it == bias_itend ) {
          CHECK_EQ(1,2) << "Img out of dimension";
        }
        float center_bias = (h - 127.5) * (h - 127.5)
                                           + (w - 127.5) * (w - 127.5);
        float tmp = 255 * std::exp(-center_bias / 10000);
        (*bias_it) = static_cast<char>(tmp);
        bias_it++;
      }
    }
    channels.push_back(biasImg);
    cv::Mat fourImg;
    cv::merge(channels, fourImg);
    top_shape = this->data_transformer_->InferBlobShape(fourImg);


  } else {
    top_shape = this->data_transformer_->InferBlobShape(cv_img);

  }

  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img_origin = ReadImageToCVMat(root_folder + lines_[lines_id_].dataName);
    CHECK(cv_img_origin.data) << "Could not load " << lines_[lines_id_].dataName;
    read_time += timer.MicroSeconds();
    timer.Start();

    int x1,y1,x2,y2;
    x1 = lines_[lines_id_].x1;
    y1 = lines_[lines_id_].y1;
    x2 = lines_[lines_id_].x2;
    y2 = lines_[lines_id_].y2;

    int h = x2-x1;
    int w = y2-y1;
    CHECK_GT(w,0) << "width must > 0 ";
    CHECK_GT(h,0) << "height must > 0 ";
    cv::Rect rect(y1,x1,w,h);

    cv::Mat sub_img = cv::Mat(cv_img_origin, rect);
    cv::Mat cv_img;
    cv::resize(sub_img, cv_img, cv::Size(new_width, new_height));

    //    {
    //      string sb;
    //      for (int j=0; j < lines_[lines_id_].labelName.size(); j++ ) {
    //        if ( lines_[lines_id_].labelName[j] != '.' ) {
    //          sb = sb + lines_[lines_id_].labelName[j];
    //        } else {
    //          break;
    //        }
    //      }
    //      string name = "/home/mayfive/data/MSRA/subset/" + sb + axis2str(x1,y1,x2,y2)+ "_origin.jpg";
    //      cv::imwrite( name.c_str(), cv_img );
    //    }

    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);


    if (center_flag) {



      std::vector<cv::Mat> channels(3);
      split(cv_img, channels);

      cv::Mat biasImg = cv::Mat(256,256,CV_8UC1);
      cv::Mat_<char>::iterator bias_it= biasImg.begin<char>();
      cv::Mat_<char>::iterator bias_itend= biasImg.end<char>();

      for (int h = 0; h < biasImg.rows; ++h) {
        for (int w = 0; w < biasImg.cols; ++w) {
          if ( bias_it == bias_itend ) {
            CHECK_EQ(1,2) << "Img out of dimension";
          }
          float center_bias = (h - 127.5) * (h - 127.5)
                                        + (w - 127.5) * (w - 127.5);
          float tmp = 255 * std::exp(-center_bias / 10000);
          (*bias_it) = static_cast<char>(tmp);
          bias_it++;
        }
      }
      channels.push_back(biasImg);
      cv::Mat fourImg;
      cv::merge(channels, fourImg);

      this->data_transformer_->Transform(fourImg, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    }

    cv::Mat cv_label_origin = ReadImageToCVMat(root_folder + lines_[lines_id_].labelName,
        0, 0, false);
    CHECK(cv_label_origin.data) << "Could not load " << lines_[lines_id_].labelName;

    cv::Mat sub_label = cv::Mat(cv_label_origin, rect);

    cv::Mat cv_label;
    cv::resize(sub_label, cv_label, cv::Size(64, 64));

    //    {
    //      string sb;
    //      for (int j=0; j < lines_[lines_id_].labelName.size(); j++ ) {
    //        if ( lines_[lines_id_].labelName[j] != '.' ) {
    //          sb = sb + lines_[lines_id_].labelName[j];
    //        } else {
    //          break;
    //        }
    //      }
    //      string name = "/home/mayfive/data/MSRA/subset/" + sb + axis2str(x1,y1,x2,y2)+ "_label.jpg";
    //      cv::imwrite( name.c_str(), cv_label );
    //
    //    }

    cv::Mat_<uchar>::iterator it= cv_label.begin<uchar>();
    cv::Mat_<uchar>::iterator itend= cv_label.end<uchar>();
    int i = 0;
    for (; it!= itend; ++it, ++i) {
      prefetch_label[4096*item_id+i] = (*it) >= 128 ? 1:0;
    }
    CHECK_EQ(i,4096) << "label image dimension miss match";

    trans_time += timer.MicroSeconds();
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.multi_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  //  LOG(INFO) << "Test input output";
  //  int testIndex = 13;
  //
  //  cv::Mat cv_label = ReadImageToCVMat(root_folder + lines_[testIndex].second,
  //      64, 64, is_color);
  //  CHECK(cv_label.data) << "Could not load " << lines_[10].second;
  //
  //  cv::Mat_<cv::Vec3b>::iterator it= cv_label.begin<cv::Vec3b>();
  //  cv::Mat_<cv::Vec3b>::iterator itend= cv_label.end<cv::Vec3b>();
  //  int i = 0;
  //
  //  double *testLabel = new double[4096];
  //
  //  for (; it!= itend; ++it, ++i) {
  //    testLabel[i] = (*it)[testIndex] > 0 ? 1:0;
  //  }
  //
  //  {
  //    cv::Mat newImg = cv::Mat(64,64,CV_8UC3);
  //    cv::Mat_<cv::Vec3b>::iterator it= newImg.begin<cv::Vec3b>();
  //    cv::Mat_<cv::Vec3b>::iterator itend= newImg.end<cv::Vec3b>();
  //
  //    int i = 0;
  //    for (; it!= itend; ++it, ++i) {
  //      (*it)[0] = prefetch_label[testIndex*4096+i] > 0 ? 255 :0;
  //      (*it)[1] = prefetch_label[testIndex*4096+i] > 0 ? 255 :0;
  //      (*it)[2] = prefetch_label[testIndex*4096+i] > 0 ? 255 :0;
  //    }
  //    string fix = "/home/mayfive/";
  //    string name = fix + "test"  + lines_[testIndex].second;
  //    cv::imwrite( name.c_str(), newImg );
  //    LOG(INFO) << "Test write out madongyu success";
  //
  //    cv::Mat orignImg = cv::Mat(256,256,CV_8UC3);
  //    //prefetch_data
  //    {
  //      cv::Mat_<cv::Vec3b>::iterator it= orignImg.begin<cv::Vec3b>();
  //      cv::Mat_<cv::Vec3b>::iterator itend= orignImg.end<cv::Vec3b>();
  //
  //      int i = 0;
  //
  //      for ( int h = 0; h < 256; h++ ) {
  //        for ( int w = 0; w < 256; w++ ) {
  //          for ( int c = 0; c < 3; c++ ) {
  //            (*it)[c] = prefetch_data[testIndex*3*256*256+(c * 256 + h) * 256 + w];
  //          }
  //          ++it;
  //        }
  //      }
  //      string fix = "/home/mayfive/";
  //      string name = fix +"test" + lines_[testIndex].first;
  //      cv::imwrite( name.c_str(), orignImg );
  //    }
  //  }
  //  CHECK_EQ(1,2) << "exit from madongyu";

  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiDataSlLayer);
REGISTER_LAYER_CLASS(MultiDataSl);

}  // namespace caffe
#endif  // USE_OPENCV
