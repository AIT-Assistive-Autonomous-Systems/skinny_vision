#include "skinny_vision/decoder/OpenCvDecoder.hpp"

#include <opencv2/imgcodecs.hpp>

using namespace skinny_vision;

namespace cv::cuda {
struct GpuMat {};
}  // namespace cv::cuda

auto OpenCvDecoder::decode(const std::vector<unsigned char>& source) -> std::variant<cv::Mat, cv::cuda::GpuMat> {
  const cv::Mat_<unsigned char> in(1, source.size(), const_cast<uchar*>(source.data()));
  const cv::Mat decoded_image = cv::imdecode(in, cv::IMREAD_UNCHANGED);

  return decoded_image;
}