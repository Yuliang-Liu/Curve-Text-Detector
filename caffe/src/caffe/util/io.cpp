#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);


  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}
// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}
bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

bool ReadFileToDatumSeq(const string& filename, const vector<int> label,
    DatumSeq* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    for(int lv = 0; lv < label.size(); lv++){
      datum->add_label(label[lv]);
    }
    // datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}
void CVMatToDatumSeq(const cv::Mat& cv_img, DatumSeq* datum) {
  // CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

bool ReadImageToDatumSeq(const string& filename, vector<int>& label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, DatumSeq* datumSeq) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  // cv::namedWindow("blur Adjustment");cv::imshow("blur Adjustment", cv_img);
  // cv::waitKey();
  if (cv_img.data) {
    datumSeq->set_width(cv_img.cols);
    datumSeq->set_height(cv_img.rows);
    datumSeq->set_channels(cv_img.channels());
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatumSeq(filename, label, datumSeq);

      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datumSeq->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      for(int lv = 0; lv < label.size(); lv++){
        datumSeq->add_label(label[lv]);
      }

      datumSeq->set_encoded(true);
      return true;
    }
    CVMatToDatumSeq(cv_img, datumSeq);
    for(int lv = 0; lv < label.size(); lv++){
      // LOG(INFO) << "lv " << lv <<" label[lv] " << label[lv]; getchar();
      datumSeq->add_label(label[lv]);
    }
    return true;
  } else {
    return false;
  }
}


cv::Mat ReadImageToCVMat_constomized(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);

 // cv::Mat img = cv::Mat(heightNum*dimNum,widthNum,CV_32FC1,words);
  // cv::imshow("fun",cv_img_origin);
  // cv::waitKey(0); 
  // std::cout << "need confirm" << std::endl;
  // getchar();

  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }

  if(cv_img_origin.rows <= height && cv_img_origin.cols <= width){
    //roi
    // std::cout << "in roi" << std::endl;
    // std::cout << height << " " << width << std::endl;
    // std::cout << cv_img_origin.rows << " " << cv_img_origin.cols << std::endl;    
    if (is_color) {
      cv_img = cv::Mat(height,width,CV_8UC3, cv::Scalar(0, 0, 0) );
      cv::Mat imageROI = cv_img(cv::Rect(width/2-cv_img_origin.cols/2,height/2-cv_img_origin.rows/2,
                                             cv_img_origin.cols,cv_img_origin.rows));
      
      cv_img_origin.copyTo(imageROI);      
    } else{
      cv_img = cv::Mat(height,width,CV_8UC1,cv::Scalar(0));      
      cv::Mat imageROI = cv_img(cv::Rect(width/2-cv_img_origin.cols/2,height/2-cv_img_origin.rows/2,
                                           cv_img_origin.cols,cv_img_origin.rows));      
      cv_img_origin.copyTo(imageROI);      
    }
  
  } else {
    // depend on which dimension is out of boundary, resize the image by the scale.
    // if both dimensions're out of boundary, choose the larger scale.
    // std::cout << height << " " << width << std::endl;
    // std::cout << cv_img_origin.rows << " " << cv_img_origin.cols << std::endl;     
    if (cv_img_origin.rows > height && cv_img_origin.cols > width) {
      float scale_width = (float)width / cv_img_origin.cols;
      float scale_height = (float)height / cv_img_origin.rows;
      float im_scale = scale_width > scale_height ? scale_height : scale_width;  
      cv::resize(cv_img_origin, cv_img_origin, cv::Size(0,0), im_scale, im_scale);
    } else if (cv_img_origin.rows > height) {
      float scale_height = (float)height / cv_img_origin.rows;
      cv::resize(cv_img_origin, cv_img_origin, cv::Size(0,0), scale_height, scale_height);          
    } else {
      float scale_width = (float)width / cv_img_origin.cols;
      cv::resize(cv_img_origin, cv_img_origin, cv::Size(0,0) , scale_width, scale_width);      
    }
   
    if (is_color) {
      cv_img = cv::Mat(height,width,CV_8UC3, cv::Scalar(0, 0, 0));
      cv::Mat imageROI = cv_img(cv::Rect(width/2-cv_img_origin.cols/2,height/2-cv_img_origin.rows/2,
                                         cv_img_origin.cols,cv_img_origin.rows));
      cv_img_origin.copyTo(imageROI);            

    } else{
      cv_img = cv::Mat(height,width,CV_8UC3, cv::Scalar(0) );
      cv::Mat imageROI = cv_img(cv::Rect(width/2-cv_img_origin.cols/2,height/2-cv_img_origin.rows/2,
                                         cv_img_origin.cols,cv_img_origin.rows));
      cv_img_origin.copyTo(imageROI);            
    }

  }


  // if (height > 0 && width > 0) {
  //   cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  // } else {
  //   cv_img = cv_img_origin;
  // }
  return cv_img;
}


bool ReadImageToDatumSeq_constomized(const string& filename, vector<int>& label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, DatumSeq* datumSeq) {
  cv::Mat cv_img = ReadImageToCVMat_constomized(filename, height, width, is_color);

  if (cv_img.data) {
    datumSeq->set_width(cv_img.cols);
    datumSeq->set_height(cv_img.rows);
    datumSeq->set_channels(cv_img.channels());
    // std::cout << "cv_img.channels(): " << cv_img.channels() << std::endl;
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatumSeq(filename, label, datumSeq);

      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datumSeq->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      for(int lv = 0; lv < label.size(); lv++){
        datumSeq->add_label(label[lv]);
      }

      datumSeq->set_encoded(true);
      return true;
    }
    CVMatToDatumSeq(cv_img, datumSeq);
    for(int lv = 0; lv < label.size(); lv++){
      // LOG(INFO) << "lv " << lv <<" label[lv] " << label[lv]; getchar();
      datumSeq->add_label(label[lv]);
    }
    return true;
  } else {
    return false;
  }
}


bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMatNative(const DatumSeq& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const DatumSeq& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

cv::Mat DatumSeqtoCVMat(const DatumSeq& datum, bool is_color = true) {
  cv::Mat img;
  if (is_color) {
    img = cv::Mat::zeros(datum.height(), datum.width(), CV_8UC3);
  } else {
    img = cv::Mat::zeros(datum.height(), datum.width(), CV_8U);
  }
  for (int h = 0; h < datum.height(); ++h) {
    uchar* ptr = img.ptr<uchar>(h);
    int img_index = 0;
    const string& buffer = datum.data();
    for (int w = 0; w < datum.width(); ++w) {
      for (int c = 0; c < datum.channels(); ++c) {
        int datum_index = (c * datum.height() + h) * datum.width() + w;
        ptr[img_index++] = static_cast<uchar>(buffer[datum_index]);
      }
    }
  }  
  return img;
}

#endif  // USE_OPENCV
}  // namespace caffe
