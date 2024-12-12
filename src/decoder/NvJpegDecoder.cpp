#include "skinny_vision/decoder/NvJpegDecoder.hpp"

#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/cudaarithm.hpp>

#include "skinny_vision/cuda_helpers.hpp"

using namespace skinny_vision;

NvJpegDecoder::NvJpegDecoder(NvJpegDecoder_settings settings) : settings(settings) {
  checkCudaErrors(cudaSetDevice(settings.device_id));  // TODO: error check
  checkCudaErrors(nvjpegCreateSimple(&nv_handle));
  checkCudaErrors(nvjpegJpegStateCreate(nv_handle, &nv_jpeg_state));
  checkCudaErrors(cudaStreamCreate(&stream));
}

NvJpegDecoder::~NvJpegDecoder() {
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(nvjpegJpegStateDestroy(nv_jpeg_state));
  checkCudaErrors(nvjpegDestroy(nv_handle));
}

auto NvJpegDecoder::decode(const std::vector<unsigned char>& data) -> std::variant<cv::Mat, cv::cuda::GpuMat> {
  nvjpegImage_t iout;
  // We iterate through all components but implementation will only support interleaved
  for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
    iout.channel[i] = nullptr;
    iout.pitch[i] = 0;
  }

  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels{3};
  nvjpegChromaSubsampling_t subsampling{};

  checkCudaErrors(nvjpegGetImageInfo(nv_handle, data.data(), data.size(), &channels, &subsampling, widths, heights));

  // prepare output buffer
  cv::cuda::GpuMat c1(heights[0], widths[0], CV_8UC1);
  cv::cuda::GpuMat c2(heights[0], widths[0], CV_8UC1);
  cv::cuda::GpuMat c3(heights[0], widths[0], CV_8UC1);

  iout.channel[0] = (unsigned char*)c1.cudaPtr();
  iout.pitch[0] = c1.step;
  iout.channel[1] = (unsigned char*)c2.cudaPtr();
  iout.pitch[1] = c2.step;
  iout.channel[2] = (unsigned char*)c3.cudaPtr();
  iout.pitch[2] = c3.step;

  checkCudaErrors(cudaStreamSynchronize(stream));
  cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
  checkCudaErrors(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
  checkCudaErrors(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

  checkCudaErrors(cudaEventRecord(startEvent, stream));

  checkCudaErrors(nvjpegDecode(nv_handle, nv_jpeg_state, data.data(), data.size(), NVJPEG_OUTPUT_BGR, &iout, stream));
  checkCudaErrors(cudaEventRecord(stopEvent, stream));

  std::vector<cv::cuda::GpuMat> channel_mats;
  channel_mats.push_back(c1);
  channel_mats.push_back(c2);
  channel_mats.push_back(c3);

  cv::cuda::GpuMat result(heights[0], widths[0], CV_8UC3);
  cv::cuda::merge(channel_mats, result);

  return result;
}
#endif