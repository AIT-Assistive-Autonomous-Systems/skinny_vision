#include "skinny_vision/ImageContainer.hpp"

#include <opencv2/core/mat.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <variant>

#include "skinny_vision/ImageConversion.hpp"
#include "skinny_vision/decoder/JpegDecoder.hpp"
#include "skinny_vision/encoder/JpegEncoder.hpp"

using namespace skinny_vision;

namespace {

template <class T, class S, class D>
auto check_and_move(S &source, D &target) -> bool {
  if (std::holds_alternative<T>(source)) {
    target.template emplace<T>(std::move(std::get<T>(source)));
    return true;
  }

  return false;
}

template <typename S>
void decode_storage(S &storage, const std::shared_ptr<JpegDecoder> &decoder) {
  if (!decoder) {
    throw std::runtime_error("No decoder set for decoding");
  }

  auto &msg = std::get<sensor_msgs::msg::CompressedImage>(storage);
  auto decoded_image = decoder->decode(msg.data);

  if (check_and_move<cv::cuda::GpuMat>(decoded_image, storage)) {
  } else if (check_and_move<cv::Mat>(decoded_image, storage)) {
  } else {
    throw std::runtime_error("Decoded image is not a cv::Mat or cv::cuda::GpuMat");
  }
}
}  // namespace

ImageContainer::ImageContainer(const sensor_msgs::msg::CompressedImage &msg) : msg_header(msg.header), storage(msg) {}

ImageContainer::ImageContainer(std::shared_ptr<JpegEncoder> encoder, cv::Mat image, std_msgs::msg::Header header)
    : msg_header(std::move(header)), encoder(std::move(encoder)), storage(std::move(image)) {}

ImageContainer::ImageContainer(std::shared_ptr<JpegEncoder> encoder, cv::cuda::GpuMat image,
                               std_msgs::msg::Header header)
    : msg_header(std::move(header)), encoder(std::move(encoder)), storage(std::move(image)) {}

ImageContainer::ImageContainer(std::shared_ptr<JpegEncoder> encoder, sensor_msgs::msg::Image image)
    : msg_header(image.header), encoder(std::move(encoder)), storage(std::move(image)) {}

ImageContainer::~ImageContainer() = default;

void ImageContainer::set_decoder(std::shared_ptr<JpegDecoder> decoder) { this->decoder = std::move(decoder); }

auto ImageContainer::cv_mat() -> cv::Mat {
  if (std::holds_alternative<sensor_msgs::msg::CompressedImage>(storage)) {
    decode_storage(storage, decoder);
  }

  if (std::holds_alternative<cv::Mat>(storage)) {
    return std::get<cv::Mat>(storage);
  } else if (std::holds_alternative<sensor_msgs::msg::Image>(storage)) {
    auto &stored_image = std::get<sensor_msgs::msg::Image>(storage);
    auto opencvEncoding = conversion::ToOpenCV(stored_image.encoding);
    return {static_cast<int>(stored_image.height), static_cast<int>(stored_image.width), opencvEncoding,
            stored_image.data.data(), stored_image.step};
  } else if (std::holds_alternative<cv::cuda::GpuMat>(storage)) {
    // RCLCPP_WARN(rclcpp::get_logger("skinny_vision"), "Converting cv::cuda::GpuMat to cv::Mat");

    cv::cuda::GpuMat gpu_image = std::get<cv::cuda::GpuMat>(storage);
    cv::Mat cpu_image;
    gpu_image.download(cpu_image);
    return cpu_image;
  }

  throw std::runtime_error("no storage");
}

auto ImageContainer::cv_cuda_mat() -> cv::cuda::GpuMat {
  if (std::holds_alternative<sensor_msgs::msg::CompressedImage>(storage)) {
    decode_storage(storage, decoder);
  }

  if (std::holds_alternative<cv::cuda::GpuMat>(storage)) {
    return std::get<cv::cuda::GpuMat>(storage);
  } else if (std::holds_alternative<cv::Mat>(storage) || std::holds_alternative<sensor_msgs::msg::Image>(storage)) {
    // RCLCPP_WARN(rclcpp::get_logger("skinny_vision"), "Converting cv::Mat to cv::cuda::GpuMat");

    cv::Mat cpu_image = cv_mat();
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(cpu_image);
    return gpu_image;
  }

  throw std::runtime_error("no storage");
}

auto ImageContainer::both() -> std::pair<cv::Mat, cv::cuda::GpuMat> {
  if (std::holds_alternative<sensor_msgs::msg::CompressedImage>(storage)) {
    decode_storage(storage, decoder);
  }

  if (std::holds_alternative<cv::cuda::GpuMat>(storage)) {
    auto &gpu_image = std::get<cv::cuda::GpuMat>(storage);
    cv::Mat cpu_image;
    gpu_image.download(cpu_image);

    return {cpu_image, gpu_image};
  } else if (std::holds_alternative<cv::Mat>(storage) || std::holds_alternative<sensor_msgs::msg::Image>(storage)) {
    auto cpu_image = cv_mat();
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(cpu_image);

    return {cpu_image, gpu_image};
  }

  throw std::runtime_error("no storage");
}

auto ImageContainer::header() -> const std_msgs::msg::Header & { return msg_header; }

auto ImageContainer::convert_to_message() const -> std::unique_ptr<sensor_msgs::msg::CompressedImage> {
  auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
  convert_into(*msg);
  return msg;
}

void ImageContainer::convert_into(sensor_msgs::msg::CompressedImage &msg) const {
  if (encoder) {
    encoder->encode(storage, msg.data);
    msg.header = msg_header;
    return;
  }

  throw std::runtime_error("No image or encoder set for conversion");
}
