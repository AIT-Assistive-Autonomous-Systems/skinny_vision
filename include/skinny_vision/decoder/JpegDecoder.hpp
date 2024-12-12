#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <variant>
#include <vector>

namespace skinny_vision {

class JpegDecoder {
 public:
  virtual ~JpegDecoder() = default;

  JpegDecoder(const JpegDecoder&) = delete;
  auto operator=(const JpegDecoder&) -> JpegDecoder& = delete;

  JpegDecoder(JpegDecoder&&) = default;
  auto operator=(JpegDecoder&&) -> JpegDecoder& = default;

  virtual auto decode(const std::vector<unsigned char>& data) -> std::variant<cv::Mat, cv::cuda::GpuMat> = 0;

 protected:
  JpegDecoder() = default;
};
}  // namespace skinny_vision