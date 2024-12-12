#include <ament_index_cpp/get_package_share_directory.hpp>
#include <catch2/catch_all.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv_modules.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include "catch2/catch_test_macros.hpp"
#include "opencv2/core/cuda.hpp"
#include "skinny_vision/ImageContainer.hpp"
#include "skinny_vision/decoder/NvJpegDecoder.hpp"
#include "skinny_vision/decoder/OpenCvDecoder.hpp"
#include "skinny_vision/encoder/NvJpegEncoder.hpp"
#include "skinny_vision/encoder/OpenCvEncoder.hpp"

using skinny_vision::OpenCvDecoder;
using skinny_vision::OpenCvEncoder;
using skinny_vision::OpenCvEncoder_settings;

#ifdef HAVE_OPENCV_CUDAIMGPROC
using skinny_vision::NvJpegDecoder;
using skinny_vision::NvJpegDecoder_settings;
using skinny_vision::NvJpegEncoder;
using skinny_vision::NvJpegEncoder_settings;
#endif

namespace fs = std::filesystem;

namespace {
fs::path get_test_image_path() {
  fs::path package_share_directory{ament_index_cpp::get_package_share_directory("skinny_vision")};
  return package_share_directory.parent_path().parent_path() / "test" / "resources" / "test_200x300.tiff";
}
}  // namespace

#ifdef HAVE_OPENCV_CUDAIMGPROC
SCENARIO("An NvJpegEncoder", "[encoder]") {
  NvJpegEncoder_settings settings{90, 0};
  NvJpegEncoder encoder(settings);

  SECTION("can encode a large image") {
    fs::path image_path = get_test_image_path();
    cv::Mat image = cv::imread(image_path);

    cv::cuda::GpuMat image_gpu;
    image_gpu.upload(image);

    std::vector<uint8_t> data;
    encoder.encode(image_gpu, data);

    REQUIRE(data.size() > 0);
  }
}
#endif

std::vector<unsigned char> encode(cv::Mat image) {
  OpenCvEncoder_settings settings{90};
  OpenCvEncoder encoder(settings);

  std::vector<uint8_t> data;
  encoder.encode(image, data);

  return data;
}

SCENARIO("A", "[decoder]") {
  fs::path image_path = get_test_image_path();
  cv::Mat image = cv::imread(image_path);
  auto raw_data = encode(image);
  REQUIRE(raw_data.size() > 0);

#ifdef HAVE_OPENCV_CUDAIMGPROC
  SECTION("NvJpegDecoder can decode a large image") {
    NvJpegDecoder_settings settings{0};
    NvJpegDecoder decoder(settings);

    auto result = decoder.decode(raw_data);
    cv::cuda::GpuMat gpu_image = std::get<cv::cuda::GpuMat>(result);
    cv::Mat cpu_image;
    gpu_image.download(cpu_image);

    REQUIRE(cpu_image.total() > 0);
  }

  SECTION("Container decoding") {
    NvJpegDecoder_settings settings{0};
    std::shared_ptr<skinny_vision::JpegDecoder> decoder = std::make_shared<NvJpegDecoder>(settings);

    sensor_msgs::msg::CompressedImage compressed_image;
    compressed_image.format = "jpeg";
    compressed_image.data = raw_data;

    skinny_vision::ImageContainer container(compressed_image);
    container.set_decoder(decoder);

    cv::cuda::GpuMat gpu_image = container.cv_cuda_mat();
    cv::Mat cpu_image;
    gpu_image.download(cpu_image);

    REQUIRE(cpu_image.total() > 0);
  }
#endif

  SECTION("OpenCvDecoder can decode a large image") {
    OpenCvDecoder decoder;
    auto result = decoder.decode(raw_data);
    cv::Mat cpu_image = std::get<cv::Mat>(result);

    REQUIRE(cpu_image.total() > 0);
  }
}

SCENARIO("An OpenCvEncoder", "[encoder]") {
  OpenCvEncoder_settings settings{90};
  OpenCvEncoder encoder(settings);

  SECTION("can encode a large image") {
    // load image from disk
    fs::path image_path = get_test_image_path();
    cv::Mat image = cv::imread(image_path);

    std::vector<uint8_t> data;
    encoder.encode(image, data);

    REQUIRE(data.size() > 0);
  }
}