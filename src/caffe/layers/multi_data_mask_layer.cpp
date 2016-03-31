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
#include "caffe/layers/multi_data_mask_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiDataMaskLayer<Dtype>::~MultiDataMaskLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiDataMaskLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.multi_data_param().new_height();
  const int new_width  = this->layer_param_.multi_data_param().new_width();
  const bool is_color  = this->layer_param_.multi_data_param().is_color();
  string root_folder = this->layer_param_.multi_data_param().root_folder();
  string mask_folder = this->layer_param_.multi_data_param().mask_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
          "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multi_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
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
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.

  vector<int> top_shape;




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

  cv::Mat cv_mask = ReadImageToCVMat(mask_folder + lines_[lines_id_].second,
      new_height, new_width, false);
  CHECK(cv_mask.data) << "Could not load " << lines_[lines_id_].second;

  cv::Mat fiveImg;
  channels.push_back(cv_mask);

  cv::merge(channels, fiveImg);

  top_shape= this->data_transformer_->InferBlobShape(fiveImg);

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
  label_shape[1] = 128*128;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void MultiDataMaskLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiDataMaskLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  const bool is_color = multi_data_param.is_color();

  string root_folder = multi_data_param.root_folder();
  string mask_folder = multi_data_param.mask_folder();


  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.

  vector<int> top_shape;

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

  cv::Mat cv_mask = ReadImageToCVMat(mask_folder + lines_[lines_id_].second,
      new_height, new_width, false);
  CHECK(cv_mask.data) << "Could not load " << lines_[lines_id_].second;
  channels.push_back(cv_mask);

  cv::Mat fiveImg;
  cv::merge(channels, fiveImg);



  top_shape = this->data_transformer_->InferBlobShape(fiveImg);


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
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();



    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

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

    cv::Mat cv_mask = ReadImageToCVMat(mask_folder + lines_[lines_id_].second,
        new_height, new_width, false);
    CHECK(cv_mask.data) << "Could not load " << lines_[lines_id_].second;


    channels.push_back(cv_mask);
    cv::Mat fiveImg;
    cv::merge(channels, fiveImg);
    this->data_transformer_->Transform(fiveImg, &(this->transformed_data_));




    cv::Mat cv_label = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
        128, 128, false);
    CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;

    cv::Mat_<uchar>::iterator it= cv_label.begin<uchar>();
    cv::Mat_<uchar>::iterator itend= cv_label.end<uchar>();
    int i = 0;
    for (; it!= itend; ++it, ++i) {
      prefetch_label[128*128*item_id+i] = (*it) >= 128 ? 1:0;
    }
    CHECK_EQ(i,128*128) << "label image dimension miss match";

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

  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiDataMaskLayer);
REGISTER_LAYER_CLASS(MultiDataMask);

}  // namespace caffe
#endif  // USE_OPENCV
