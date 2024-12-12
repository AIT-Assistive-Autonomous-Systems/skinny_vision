cmake_minimum_required(VERSION 3.8)
project(skinny_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED)
endif()

add_compile_options(-fPIC)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(OpenCV REQUIRED   
#  COMPONENTS
#     opencv_core
#     opencv_imgproc
#     opencv_imgcodecs
#     opencv_cudaimgproc
#   CONFIG
)

find_package(CUDA)

if(CUDA_FOUND)
else()
  set(CUDA_INCLUDE_DIRS "")
endif()

function(build_library)
  if(CUDA_FOUND)
    cuda_add_library(${ARGN})
  else()
    add_library(${ARGN})
  endif()
endfunction()

build_library(${PROJECT_NAME}
  src/ImageContainer.cpp
  src/ImageConversion.cpp
  src/encoder/NvJpegEncoder.cpp
  src/decoder/NvJpegDecoder.cpp
  src/encoder/OpenCvEncoder.cpp
  src/decoder/OpenCvDecoder.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  vision_msgs
  sensor_msgs
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)

if(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    nvjpeg
  )
endif()

# Build component
build_library(skinny_vision_republisher SHARED
  src/RepublisherNode.cpp
)

target_include_directories(skinny_vision_republisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(skinny_vision_republisher
  ${PROJECT_NAME}
)

ament_target_dependencies(skinny_vision_republisher
  rclcpp_components
  rclcpp
)

rclcpp_components_register_node(skinny_vision_republisher
  PLUGIN "skinny_vision::RepublishNode"
  EXECUTABLE republish_node
)

# Install

ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  vision_msgs
  sensor_msgs
)

ament_export_targets(
  export_${PROJECT_NAME} HAS_LIBRARY_TARGET
)

install(TARGETS 
  ${PROJECT_NAME}
  skinny_vision_republisher

  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

# Testing
if(BUILD_TESTING)
  include(FetchContent)
  find_package(ament_cmake_catch2)

  set(CATCH_TAG "v3.1.0")
  set(CATCH_BUILD_TESTING OFF CACHE BOOL "")
  set(CATCH_BUILD_EXAMPLES OFF CACHE BOOL "")
  FetchContent_Declare(Catch2
          GIT_REPOSITORY https://github.com/catchorg/Catch2.git
          GIT_TAG ${CATCH_TAG}
          )
  FetchContent_MakeAvailable(Catch2)

  file(GLOB_RECURSE TEST_SOURCES test/*.cpp)
  ament_add_catch2(${PROJECT_NAME}-test 
    TIMEOUT 120
    ${TEST_SOURCES}
  )

  target_include_directories(${PROJECT_NAME}-test 
    PRIVATE
    include
    test
  )

  target_link_libraries(${PROJECT_NAME}-test PRIVATE
    Catch2::Catch2WithMain
    ${PROJECT_NAME}
  )

  install(
    DIRECTORY test/resources/
    DESTINATION test/resources/
  )
endif()

ament_package()