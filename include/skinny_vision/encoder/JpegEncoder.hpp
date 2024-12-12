#pragma once

#include "skinny_vision/ContainerStorage.hpp"

namespace skinny_vision {

class JpegEncoder {
 public:
  virtual ~JpegEncoder() = default;

  JpegEncoder(const JpegEncoder&) = delete;
  auto operator=(const JpegEncoder&) -> JpegEncoder& = delete;

  JpegEncoder(JpegEncoder&&) = default;
  auto operator=(JpegEncoder&&) -> JpegEncoder& = default;

  auto encode(const ContainerStorage& image) -> std::vector<unsigned char> {
    std::vector<unsigned char> result;
    encode(image, result);
    return result;
  }

  virtual void encode(const ContainerStorage& image, std::vector<unsigned char>& output) = 0;

 protected:
  JpegEncoder() = default;
};
}  // namespace skinny_vision