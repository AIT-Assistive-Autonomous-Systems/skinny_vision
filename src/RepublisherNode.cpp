#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "skinny_vision/ImageContainer.hpp"
#include "skinny_vision/ImageConversion.hpp"
#include "skinny_vision/encoder/NvJpegEncoder.hpp"
#include "skinny_vision/encoder/OpenCvEncoder.hpp"

namespace {
constexpr const char *INPUT_TOPIC_PARAM = "input_topic";
constexpr const char *OUTPUT_TOPIC_PARAM = "output_topic";
constexpr const char *GPU_ID_PARAM = "gpu_id";
constexpr const char *JPEG_QUALITY_PARAM = "quality";

constexpr int DEFAULT_GPU_ID{0};
constexpr int DEFAULT_JPEG_QUALITY{95};
}  // namespace

namespace skinny_vision {
class RepublishNode : public rclcpp::Node {
 public:
  RepublishNode(const rclcpp::NodeOptions &options) : Node("republish_node", options) {
    this->declare_parameter<std::string>(INPUT_TOPIC_PARAM, "input");
    this->declare_parameter<std::string>(OUTPUT_TOPIC_PARAM, "output");
    this->declare_parameter<int>(GPU_ID_PARAM, DEFAULT_GPU_ID);
    this->declare_parameter<int>(JPEG_QUALITY_PARAM, DEFAULT_JPEG_QUALITY);

    auto input_topic = this->get_parameter(INPUT_TOPIC_PARAM).as_string();
    auto output_topic = this->get_parameter(OUTPUT_TOPIC_PARAM).as_string();
    auto gpu_id = this->get_parameter(GPU_ID_PARAM).as_int();
    auto quality = this->get_parameter(JPEG_QUALITY_PARAM).as_int();

    RCLCPP_INFO_STREAM(get_logger(),  // NOLINT: External logger
                       "Republishing from " << input_topic << " to " << output_topic);
    RCLCPP_INFO_STREAM(get_logger(), "GPU ID: " << gpu_id);         // NOLINT: External logger
    RCLCPP_INFO_STREAM(get_logger(), "JPEG Quality: " << quality);  // NOLINT: External logger

#ifdef HAVE_OPENCV_CUDAIMGPROC
    NvJpegEncoder_settings settings{static_cast<int>(quality), static_cast<int>(gpu_id)};
    encoder = std::make_shared<NvJpegEncoder>(settings);
#else
    OpenCvEncoder_settings settings{static_cast<int>(quality)};
    encoder = std::make_shared<OpenCvEncoder>(settings);
#endif

    rclcpp::QoS default_qos(5);

    auto qos_override_options = rclcpp::QosOverridingOptions({
        rclcpp::QosPolicyKind::Depth,
        rclcpp::QosPolicyKind::Durability,
        rclcpp::QosPolicyKind::History,
        rclcpp::QosPolicyKind::Reliability,
    });

    rclcpp::SubscriptionOptions sub_options;
    rclcpp::PublisherOptions pub_options;

    pub_options.qos_overriding_options = qos_override_options;
    sub_options.qos_overriding_options = qos_override_options;

    publisher_compressed =
        this->create_publisher<ImageContainer::PublishedType>(output_topic, default_qos, pub_options);

    subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        input_topic, default_qos,
        [this](std::unique_ptr<sensor_msgs::msg::Image> msg) { this->republish(std::move(msg)); }, sub_options);
  }

  ~RepublishNode() override = default;

  RepublishNode(RepublishNode &&) = delete;
  RepublishNode(const RepublishNode &) = delete;

  auto operator=(RepublishNode &&other) -> RepublishNode & = delete;
  auto operator=(const RepublishNode &other) -> RepublishNode & = delete;

 private:
  void republish(std::unique_ptr<sensor_msgs::msg::Image> msg) {
    if (publisher_compressed->get_subscription_count() > 0) {
#ifdef HAVE_OPENCV_CUDAIMGPROC
      auto opencvEncoding = conversion::ToOpenCV(msg->encoding);
      cv::Mat cpu_image = cv::Mat(static_cast<int>(msg->height), static_cast<int>(msg->width), opencvEncoding,
                                  msg->data.data(), msg->step);

      cv::cuda::GpuMat gpu_image;
      gpu_image.upload(cpu_image);
      auto contained_msg = std::make_unique<ImageContainer>(encoder, std::move(gpu_image), msg->header);
#else
      auto contained_msg = std::make_unique<ImageContainer>(encoder, std::move(*msg));
      msg.reset();
#endif
      publisher_compressed->publish(std::move(contained_msg));
    }
  }

  std::shared_ptr<JpegEncoder> encoder;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber;
  rclcpp::Publisher<ImageContainer::PublishedType>::SharedPtr publisher_compressed;
};
}  // namespace skinny_vision

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(skinny_vision::RepublishNode)  // NOLINT: External macro
