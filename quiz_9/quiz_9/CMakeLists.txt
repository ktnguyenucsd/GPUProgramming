cmake_minimum_required(VERSION 3.10)
project(Quiz9 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(${CMAKE_SYSTEM_NAME})
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(WINDOWS TRUE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LINUX TRUE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

if(${CMAKE_CL_64})
  set(ARCH_64 TRUE)
  set(PROCESSOR_ARCH "x64")
  message(64)
endif(${CMAKE_CL_64})

# Enable CUDA
enable_language(CUDA)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lnvToolExt -gencode arch=compute_52,code=sm_52)

if(WINDOWS)
    set(CUDA_TOOLKIT_SAMPLES_DIR "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1")
    add_definitions(-DBUILD_THIRDPARTY:BOOL=ON)
endif(WINDOWS)

find_package(NVTX REQUIRED)
find_package(FREEGLUT REQUIRED)

# Build the Src directory
add_subdirectory(Src)
