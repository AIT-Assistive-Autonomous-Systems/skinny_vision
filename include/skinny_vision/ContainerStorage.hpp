#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <variant>

namespace skinny_vision {
using ContainerStorage =
    std::variant<cv::Mat, cv::cuda::GpuMat, sensor_msgs::msg::CompressedImage, sensor_msgs::msg::Image, std::monostate>;
}