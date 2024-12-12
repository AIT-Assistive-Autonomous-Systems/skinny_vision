# Skinny Vision

## Overview

This package provides type-adaptation of cv::cuda::GpuMat and sensor_msgs::msgs::CompressedImage with hardware accelerated compression/decompression.

When using intra-process comunication the cv::Mat / cv::cuda::GpuMat will be handed over directly while when using inter-process comunication these will be compressed. If compiling with an opencv version with CUDA support you can pick a NvJpeg based encoder / decoder for hardware accelerated versions of the encoders / decoders.


## Usage

It is recomended that there is parity between selected encoder and decoder so there are no unecessary copies, eg: put the image into the GPU and use a NvJpegEncoder on publishers and NvJpegDecoder on subscribers and take the image also from the gpu.

### Limitations

* Currently the encoders / decoders work with JPEG compression and expect a BGR image as input.
* NvJpeg encoder / decoders are also not thread friendly. It is best to have a single dedicated thread when using encoding / decoding.


### Example

This node is responsible for acquiring/generating the images that are then pushed downstream. It uses type-adaptation for the published images. The container supports 3 types of images: cv::Mat; cv::cuda::GpuMat; and sensor_msgs::msg::Image. The latter being usefull for republishing of raw images with the adavantages provided by the container.

A simple example of a capture device is:

```cpp
#include <skinny_vision/ImageContainer.hpp>

namespace skinny_vision {
class JpegEncoder;
}

class SamplePublisher : public rclcpp::Node{
    ...

    private:
    std::shared_ptr<skinny_vision::JpegEncoder> encoder;
    rclcpp::Publisher<skinny_vision::ImageContainer::PublishedType>::SharedPtr publisher_compressed;
}

```

And when publishing a cv::Mat :
```cpp
#include <skinny_vision/encoder/OpenCvEncoder.hpp>

using skinny_vision::ImageContainer;
using skinny_vision::JpegEncoder;
using skinny_vision::JpegEncoder_settings;

SamplePublisher::SamplePublisher(...) {
    ...
    int compression_quality{90};

    OpenCvEncoder_settings settings{compression_quality};
    encoder = std::make_shared<OpenCvEncoder>(settings);

    publisher_compressed = this->create_publisher<ImageContainer::PublishedType>(topicName, 10);
}

void publish() {
    ...
    cv::Mat cvImage(..);

    // we can check if there are subscribers before attempting to publish
    if (publisher_compressed->get_subscription_count() > 0) {
    {
        // Generate the neader
        std_msgs::msg::Header header;
        header.stamp = now();

        // Create the skinny container to hold our image & publish
        auto result = std::make_unique<skinny_vision::ImageContainer>(encoder, std::move(cvImage), header);
        publisher_compressed->publish(std::move(result))
    }
}

```

While when publishing a cv::cuda::GpuMat:
```cpp
#include <skinny_vision/encoder/NvJpegEncoder.hpp>

using skinny_vision::ImageContainer;
using skinny_vision::JpegEncoder;
using skinny_vision::JpegEncoder_settings;

SamplePublisher::SamplePublisher(...) {
    ...
    int compression_quality{90};
    int gpu_id{0};

    NvJpegEncoder_settings settings{compression_quality, gpu_id};
    encoder = std::make_shared<NvJpegEncoder>(settings);

    publisher_compressed = this->create_publisher<ImageContainer::PublishedType>(topicName, 10);
}

void publish_cpu() {
    ...

    // Get an image to the GPU
    cv::Mat cvImage(..);
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(cvImage);

    // we can check if there are subscribers before attempting to publish
    if (publisher_compressed->get_subscription_count() > 0) {
    {
        // Generate the neader
        std_msgs::msg::Header header;
        header.stamp = now();

        // Create the skinny container to hold our image & publish
        auto result = std::make_unique<skinny_vision::ImageContainer>(encoder, std::move(gpuImage), header);
        publisher_compressed->publish(std::move(result))
    }
}
```
### User device

This node gets an image from a subscription that will have zero-copy if intra-comunication is used. Otherwise it gets the compressed image, decompresses and leaves it in the CPU / GPU depending on the decoder.

Header:
```cpp

#include <skinny_vision/ImageContainer.hpp>

namespace skinny_vision {
class JpegDecoder;
}

class SampleSubscriber : public rclcpp::Node
{
    ...
    public:
    void image_topic_callback(std::unique_ptr<skinny_vision::ImageContainer> image);

    private:
    std::shared_ptr<skinny_vision::JpegDecoder> decoder;
    rclcpp::Subscription<skinny_vision::ImageContainer::PublishedType>::SharedPtr subscriber;
}

```

Implementation for a NvJpeg decoder:

```cpp
#include <skinny_vision/decoder/NvJpegDecoder.hpp>

using skinny_vision::ImageContainer;
using skinny_vision::JpegDecoder;
using skinny_vision::JpegDecoder_settings;

SampleSubscriber::SampleSubscriber(...) {
    ...
    int gpu_id{0};

    NvJpegDecoder_settings settings{gpu_id};
    decoder = std::make_shared<NvJpegDecoder>(settings);

    subscriber = create_subscription<ImageContainer::PublishedType>(
        inputTopicName, 10, [this](std::unique_ptr<ImageContainer> image) { image_topic_callback(std::move(image)); });
}

void SampleSubscriber::image_topic_callback(std::unique_ptr<ImageContainer> image) {
    image->set_decoder(decoder);

    // Since we use a NvJpeg decoder we expect the decoded image to be in the GPU
    // If no decoding happens we still expect it to be in the GPU
    cv::cuda::GpuMat cudaImage = image->cv_cuda_mat();

    ...
}
```
