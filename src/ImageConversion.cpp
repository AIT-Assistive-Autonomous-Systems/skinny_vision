#include "skinny_vision/ImageConversion.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <unordered_map>

namespace skinny_vision::conversion {
class UnrecognizedFormatException : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

auto ToOpenCV(const std::string &ros_image_encoding) -> int {
  static const std::unordered_map<std::string, int> ToCVSensorMessageFormat{
      {sensor_msgs::image_encodings::TYPE_8UC1, CV_8UC1},     {sensor_msgs::image_encodings::TYPE_8UC2, CV_8UC2},
      {sensor_msgs::image_encodings::TYPE_8UC3, CV_8UC3},     {sensor_msgs::image_encodings::TYPE_8UC4, CV_8UC4},
      {sensor_msgs::image_encodings::TYPE_8SC1, CV_8SC1},     {sensor_msgs::image_encodings::TYPE_8SC2, CV_8SC2},
      {sensor_msgs::image_encodings::TYPE_8SC3, CV_8SC3},     {sensor_msgs::image_encodings::TYPE_8SC4, CV_8SC4},
      {sensor_msgs::image_encodings::TYPE_16UC1, CV_16UC1},   {sensor_msgs::image_encodings::TYPE_16UC2, CV_16UC2},
      {sensor_msgs::image_encodings::TYPE_16UC3, CV_16UC3},   {sensor_msgs::image_encodings::TYPE_16UC4, CV_16UC4},
      {sensor_msgs::image_encodings::TYPE_16SC1, CV_16SC1},   {sensor_msgs::image_encodings::TYPE_16SC2, CV_16SC2},
      {sensor_msgs::image_encodings::TYPE_16SC3, CV_16SC3},   {sensor_msgs::image_encodings::TYPE_16SC4, CV_16SC4},
      {sensor_msgs::image_encodings::TYPE_32SC1, CV_32SC1},   {sensor_msgs::image_encodings::TYPE_32SC2, CV_32SC2},
      {sensor_msgs::image_encodings::TYPE_32SC3, CV_32SC3},   {sensor_msgs::image_encodings::TYPE_32SC4, CV_32SC4},
      {sensor_msgs::image_encodings::TYPE_32FC1, CV_32FC1},   {sensor_msgs::image_encodings::TYPE_32FC2, CV_32FC2},
      {sensor_msgs::image_encodings::TYPE_32FC3, CV_32FC3},   {sensor_msgs::image_encodings::TYPE_32FC4, CV_32FC4},
      {sensor_msgs::image_encodings::TYPE_64FC1, CV_64FC1},   {sensor_msgs::image_encodings::TYPE_64FC2, CV_64FC2},
      {sensor_msgs::image_encodings::TYPE_64FC3, CV_64FC3},   {sensor_msgs::image_encodings::TYPE_64FC4, CV_64FC4},
      {sensor_msgs::image_encodings::BAYER_RGGB8, CV_8UC1},   {sensor_msgs::image_encodings::BAYER_BGGR8, CV_8UC1},
      {sensor_msgs::image_encodings::BAYER_GBRG8, CV_8UC1},   {sensor_msgs::image_encodings::BAYER_GRBG8, CV_8UC1},
      {sensor_msgs::image_encodings::BAYER_RGGB16, CV_16UC1}, {sensor_msgs::image_encodings::BAYER_BGGR16, CV_16UC1},
      {sensor_msgs::image_encodings::BAYER_GBRG16, CV_16UC1}, {sensor_msgs::image_encodings::BAYER_GRBG16, CV_16UC1},
      {sensor_msgs::image_encodings::MONO8, CV_8UC1},         {sensor_msgs::image_encodings::RGB8, CV_8UC3},
      {sensor_msgs::image_encodings::BGR8, CV_8UC3},          {sensor_msgs::image_encodings::MONO16, CV_16UC1},
      {sensor_msgs::image_encodings::RGB16, CV_16UC3},        {sensor_msgs::image_encodings::BGR16, CV_16UC3},
      {sensor_msgs::image_encodings::RGBA8, CV_8UC4},         {sensor_msgs::image_encodings::BGRA8, CV_8UC4},
      {sensor_msgs::image_encodings::RGBA16, CV_16UC4},       {sensor_msgs::image_encodings::BGRA16, CV_16UC4},
      {sensor_msgs::image_encodings::YUV422, CV_8UC2},        {sensor_msgs::image_encodings::YUV422_YUY2, CV_8UC2},

  };  // NOLINT(hicpp-signed-bitwise)

  try {
    { return ToCVSensorMessageFormat.at(ros_image_encoding); }
  } catch (std::out_of_range &exc) {
    throw UnrecognizedFormatException("Unrecognized format conversions to OpenCV: " + ros_image_encoding);
  }
}

auto GetConversionCode(int inputType) -> int {
  switch (inputType) {
    case CV_8UC1:  // Grayscale
      return cv::COLOR_GRAY2BGR;
    case CV_8UC4:  // BGRA
      return cv::COLOR_BGRA2BGR;
    case CV_16UC1:  // Bayer pattern
      return cv::COLOR_BayerBG2BGR;
    default:
      return -1;  // Unknown or unsupported format
  }
}
}  // namespace skinny_vision::conversion