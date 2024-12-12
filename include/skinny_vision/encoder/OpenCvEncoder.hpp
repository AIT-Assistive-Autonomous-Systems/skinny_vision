#pragma once

#include "skinny_vision/encoder/JpegEncoder.hpp"

namespace skinny_vision {

struct OpenCvEncoder_settings {
  int quality;
};

class OpenCvEncoder : public JpegEncoder {
 public:
  using SettingsType = OpenCvEncoder_settings;

  OpenCvEncoder(OpenCvEncoder_settings settings);
  ~OpenCvEncoder() override;

  OpenCvEncoder(const OpenCvEncoder&) = delete;
  auto operator=(const OpenCvEncoder&) -> OpenCvEncoder& = delete;

  OpenCvEncoder(OpenCvEncoder&&) = default;
  auto operator=(OpenCvEncoder&&) -> OpenCvEncoder& = default;

  void encode(const ContainerStorage& image, std::vector<unsigned char>& output) override;

 private:
  OpenCvEncoder_settings settings;
};
}  // namespace skinny_vision