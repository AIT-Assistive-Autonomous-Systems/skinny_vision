#pragma once

#include "skinny_vision/encoder/JpegEncoder.hpp"

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

namespace skinny_vision {

struct NvJpegEncoder_settings {
  int quality;
  int device_id;
};

class NvJpegEncoder : public JpegEncoder {
 public:
  using SettingsType = NvJpegEncoder_settings;

  NvJpegEncoder(NvJpegEncoder_settings settings);
  ~NvJpegEncoder() override;

  NvJpegEncoder(const NvJpegEncoder&) = delete;
  auto operator=(const NvJpegEncoder&) -> NvJpegEncoder& = delete;

  NvJpegEncoder(NvJpegEncoder&&) = default;
  auto operator=(NvJpegEncoder&&) -> NvJpegEncoder& = default;

  void encode(const ContainerStorage& image, std::vector<unsigned char>& output) override;

 private:
  nvjpegHandle_t nv_handle{};
  nvjpegEncoderState_t nv_enc_state{};
  nvjpegEncoderParams_t nv_enc_params{};
  cudaStream_t stream{};

  NvJpegEncoder_settings settings;
};
}  // namespace skinny_vision
#endif