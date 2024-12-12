#pragma once

#include <memory>
#include <rclcpp/type_adapter.hpp>
#include <sensor_msgs/msg/detail/image__struct.hpp>

#include "ContainerStorage.hpp"

namespace skinny_vision {
class JpegEncoder;
class JpegDecoder;

class ImageContainer {
 public:
  using PublishedType = rclcpp::TypeAdapter<ImageContainer, sensor_msgs::msg::CompressedImage>;

  ImageContainer() = default;
  ~ImageContainer();

  ImageContainer(const ImageContainer& other) = default;
  ImageContainer(ImageContainer&& other) = default;

  auto operator=(const ImageContainer& other) -> ImageContainer& = default;
  auto operator=(ImageContainer&& other) -> ImageContainer& = default;

  explicit ImageContainer(const sensor_msgs::msg::CompressedImage& msg);
  ImageContainer(std::shared_ptr<JpegEncoder> encoder, cv::Mat image, std_msgs::msg::Header header);
  ImageContainer(std::shared_ptr<JpegEncoder> encoder, cv::cuda::GpuMat image, std_msgs::msg::Header header);
  ImageContainer(std::shared_ptr<JpegEncoder> encoder, sensor_msgs::msg::Image image);

  void set_decoder(std::shared_ptr<JpegDecoder> decoder);

  template <typename T>
  [[nodiscard]] auto holds_type() const -> bool {
    return std::holds_alternative<T>(storage);
  }

  [[nodiscard]] auto cv_mat() -> cv::Mat;
  [[nodiscard]] auto cv_cuda_mat() -> cv::cuda::GpuMat;
  [[nodiscard]] auto both() -> std::pair<cv::Mat, cv::cuda::GpuMat>;

  [[nodiscard]] auto header() -> const std_msgs::msg::Header&;
  [[nodiscard]] auto convert_to_message() const -> std::unique_ptr<sensor_msgs::msg::CompressedImage>;

  void convert_into(sensor_msgs::msg::CompressedImage& msg) const;

 private:
  std_msgs::msg::Header msg_header;
  std::shared_ptr<JpegEncoder> encoder;
  std::shared_ptr<JpegDecoder> decoder;
  ContainerStorage storage;
};
}  // namespace skinny_vision

template <>
struct rclcpp::TypeAdapter<skinny_vision::ImageContainer, sensor_msgs::msg::CompressedImage> {
  using is_specialized = std::true_type;
  using custom_type = skinny_vision::ImageContainer;
  using ros_message_type = sensor_msgs::msg::CompressedImage;

  static void convert_to_ros_message(const custom_type& source, ros_message_type& destination) {
    source.convert_into(destination);
  }

  static void convert_to_custom(const ros_message_type& source, custom_type& destination) {
    destination = skinny_vision::ImageContainer(source);
  }
};
