#include "skinny_vision/encoder/OpenCvEncoder.hpp"

#include <rclcpp/rclcpp.hpp>

#include "skinny_vision/ImageConversion.hpp"
#include "skinny_vision/cuda_helpers.hpp"

using namespace skinny_vision;
using skinny_vision::conversion::GetConversionCode;

namespace {
auto ConvertToTargetEncoding(const cv::Mat &image) -> cv::Mat {
  if (image.type() == CV_8UC3) {
    // No conversion needed
    return image;
  }

  int conversionCode = GetConversionCode(image.type());

  if (conversionCode == -1) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenCvEncoder"),  // NOLINT: External logger
                 "%s(%i) : Unsupported image format : %d.", __FILE__, __LINE__, image.type());

    throw std::runtime_error("Unsupported image format");
  }

  RCLCPP_WARN(rclcpp::get_logger("NvJpegEncoder"),  // NOLINT: External logger
              "Performing color conversion for JPEG encoding");

  cv::Mat converted;
  cv::cvtColor(image, converted, conversionCode);

  return converted;
}
}  // namespace

OpenCvEncoder::OpenCvEncoder(OpenCvEncoder_settings settings) : settings(settings) {}

OpenCvEncoder::~OpenCvEncoder() = default;

void OpenCvEncoder::encode(const ContainerStorage &image, std::vector<unsigned char> &output) {
  cv::Mat cpuImage;
  if (std::holds_alternative<cv::cuda::GpuMat>(image)) {
    cv::cuda::GpuMat gpu_image = std::get<cv::cuda::GpuMat>(image);
    gpu_image.download(cpuImage);
  } else if (!std::holds_alternative<cv::Mat>(image)) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenCvEncoder"),  // NOLINT: External logger
                 "Storage does not contain an image");

    throw std::runtime_error("Storage does not contain an image");
  }

  cpuImage = std::get<cv::Mat>(image);
  auto convertedImage = ConvertToTargetEncoding(cpuImage);

  cv::imencode(".jpeg", convertedImage, output, {cv::IMWRITE_JPEG_QUALITY, settings.quality});
}