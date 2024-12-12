#pragma once

#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <stdexcept>

#define checkCudaErrors(msg) _checkCudaErrors(msg, __FILE__, __LINE__)

namespace skinny_vision {

inline auto _cudaGetErrorEnum(cudaError_t error) -> const char* { return cudaGetErrorName(error); }

inline auto _cudaGetErrorEnum(nvjpegStatus_t error) -> const char* {
  switch (error) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";

    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
      return "NVJPEG_STATUS_INCOMPLETE_BITSTREAM";

    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
      return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
  }

  return "<unknown>";
}

template <typename T>
void _checkCudaErrors(T result, const char* file, const int line) {
  if (result) {
    RCLCPP_ERROR(rclcpp::get_logger("JpegEncoder"), "%s(%i) : CUDA error : %s (%d).\n", file, line,
                 _cudaGetErrorEnum(result), static_cast<int>(result));
    throw std::runtime_error("CUDA error");
  }
}
}  // namespace skinny_vision
#endif  // HAVE_OPENCV_CUDAIMGPROC