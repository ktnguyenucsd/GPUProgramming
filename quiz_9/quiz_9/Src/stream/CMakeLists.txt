cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
  ${CMAKE_SOURCE_DIR}/include
)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 52)

# Define CUDA Source Files
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

# Create CUDA Executable (Now Includes image_loader.cpp)
add_executable(quiz quiz.cu image_loader.cpp)
set_target_properties(quiz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(quiz ${CUDAToolkit_LIBRARIES})
