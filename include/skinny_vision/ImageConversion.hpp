#pragma once
#include <string>

namespace skinny_vision::conversion {
auto ToOpenCV(const std::string &ros_image_encoding) -> int;
auto GetConversionCode(int inputType) -> int;
}  // namespace skinny_vision::conversion