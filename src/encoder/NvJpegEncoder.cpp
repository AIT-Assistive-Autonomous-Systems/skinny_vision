#include "skinny_vision/encoder/NvJpegEncoder.hpp"

#include "skinny_vision/ImageConversion.hpp"

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "skinny_vision/cuda_helpers.hpp"

using namespace skinny_vision;
using skinny_vision::conversion::GetConversionCode;

// From ros_genicam_camera conversions
namespace {
auto ConvertToTargetEncoding(const cv::cuda::GpuMat &image) -> cv::cuda::GpuMat {
  if (image.type() == CV_8UC3) {
    // No conversion needed
    return image;
  }

  int conversionCode = GetConversionCode(image.type());

  if (conversionCode == -1) {
    RCLCPP_ERROR(rclcpp::get_logger("NvJpegEncoder"),  // NOLINT: External logger
                 "%s(%i) : Unsupported image format : %d.", __FILE__, __LINE__, image.type());

    throw std::runtime_error("Unsupported image format");
  }

  RCLCPP_WARN(rclcpp::get_logger("NvJpegEncoder"),  // NOLINT: External logger
              "Performing color conversion for JPEG encoding");

  cv::cuda::GpuMat converted;
  cv::cuda::cvtColor(image, converted, conversionCode);

  return converted;
}
}  // namespace

NvJpegEncoder::NvJpegEncoder(NvJpegEncoder_settings settings) : settings(settings) {
  // Create the JPEG encoder
  checkCudaErrors(cudaSetDevice(settings.device_id));  // TODO: error check

  checkCudaErrors(cudaStreamCreate(&stream));

  // initialize nvjpeg structures
  checkCudaErrors(nvjpegCreateSimple(&nv_handle));
  checkCudaErrors(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
  checkCudaErrors(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
  checkCudaErrors(nvjpegEncoderParamsSetQuality(nv_enc_params, settings.quality, stream));
  checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));
}

NvJpegEncoder::~NvJpegEncoder() {
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(nvjpegEncoderStateDestroy(nv_enc_state));
  checkCudaErrors(nvjpegEncoderParamsDestroy(nv_enc_params));
  checkCudaErrors(nvjpegDestroy(nv_handle));
}

void NvJpegEncoder::encode(const ContainerStorage &image, std::vector<unsigned char> &output) {
  cv::cuda::GpuMat cudaImage;
  if (std::holds_alternative<cv::cuda::GpuMat>(image)) {
    cudaImage = std::get<cv::cuda::GpuMat>(image);
  } else if (std::holds_alternative<cv::Mat>(image)) {
    cv::Mat cpu_image = std::get<cv::Mat>(image);
    cudaImage.upload(cpu_image);
  } else if (!std::holds_alternative<cv::cuda::GpuMat>(image)) {
    RCLCPP_ERROR(rclcpp::get_logger("NvJpegEncoder"),  // NOLINT: External logger
                 "Storage does not contain an image");

    throw std::runtime_error("Storage does not contain an image");
  }

  auto converted = ConvertToTargetEncoding(cudaImage);

  nvjpegImage_t nv_image;
  memset(&nv_image, 0, sizeof(nv_image));

  nv_image.channel[0] = converted.data;
  nv_image.pitch[0] = (unsigned int)converted.step;
  nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGRI;  // NVJPEG_INPUT_RGB

  // Compress image
  checkCudaErrors(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &nv_image, input_format, converted.cols,
                                    converted.rows, stream));

  // get compressed stream size
  size_t length{0};
  checkCudaErrors(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, nullptr, &length, stream));
  // get stream itself
  output.resize(length);
  auto buffer = output.data();

  cudaStreamSynchronize(stream);
  //   cv::cuda::GpuMat encodedImage(1, length, CV_8UC1);
  checkCudaErrors(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, buffer, &length, 0));
}
#endif