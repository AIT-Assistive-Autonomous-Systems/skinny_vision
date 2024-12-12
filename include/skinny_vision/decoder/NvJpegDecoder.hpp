#pragma once

#include "JpegDecoder.hpp"

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <opencv2/core/cuda.hpp>
namespace skinny_vision {

struct NvJpegDecoder_settings {
  int device_id;
};

class NvJpegDecoder : public JpegDecoder {
 public:
  NvJpegDecoder(NvJpegDecoder_settings settings);
  ~NvJpegDecoder() override;

  NvJpegDecoder(const NvJpegDecoder&) = delete;
  auto operator=(const NvJpegDecoder&) -> NvJpegDecoder& = delete;

  NvJpegDecoder(NvJpegDecoder&&) = default;
  auto operator=(NvJpegDecoder&&) -> NvJpegDecoder& = default;

  auto decode(const std::vector<unsigned char>& data) -> std::variant<cv::Mat, cv::cuda::GpuMat> override;

 private:
  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_jpeg_state;
  cudaStream_t stream;

  NvJpegDecoder_settings settings;
};
}  // namespace skinny_vision
#endif