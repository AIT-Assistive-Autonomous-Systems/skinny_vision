#pragma once

#include "JpegDecoder.hpp"

namespace skinny_vision {

class OpenCvDecoder : public JpegDecoder {
 public:
  OpenCvDecoder() = default;
  ~OpenCvDecoder() override = default;

  OpenCvDecoder(const OpenCvDecoder&) = delete;
  auto operator=(const OpenCvDecoder&) -> OpenCvDecoder& = delete;

  OpenCvDecoder(OpenCvDecoder&&) = default;
  auto operator=(OpenCvDecoder&&) -> OpenCvDecoder& = default;

  auto decode(const std::vector<unsigned char>& data) -> std::variant<cv::Mat, cv::cuda::GpuMat> override;
};
}  // namespace skinny_vision