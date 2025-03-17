cmake_minimum_required(VERSION 2.6)
project(ImageConvolution)

# Set C++ Standard
set(CMAKE_CXX_FLAGS "-std=c++0x")

# Include directories (add 'include/' for STB)
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

# Define source files
set(SOURCES
  ${CMAKE_SOURCE_DIR}/src/convolution.cpp
)

# Create Executable
add_executable(ImageConvolution ${SOURCES})


