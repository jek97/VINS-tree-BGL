cmake_minimum_required(VERSION 3.5)
project(vins)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)
set(CMAKE_BUILD_TYPE Release)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(message_filters REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_models REQUIRED)
find_package(image_transport REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(visual_odometry_interfaces REQUIRED) 
find_package(sensor_msgs REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

# ============================================================================
# ONNX Runtime Configuration
# ============================================================================
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")

if(EXISTS ${ONNXRUNTIME_ROOT})
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
    
    # Find the actual library file
    find_library(ONNXRUNTIME_LIB 
        NAMES onnxruntime libonnxruntime.so libonnxruntime.so.1
        PATHS ${ONNXRUNTIME_ROOT}/lib
        NO_DEFAULT_PATH
    )
    
    if(ONNXRUNTIME_LIB)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_LIB})
        set(CMAKE_BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
        set(CMAKE_INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
        set(CMAKE_SKIP_BUILD_RPATH FALSE)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    else()
        message(WARNING "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT}/lib")
    endif()
else()
    message(WARNING "ONNX Runtime root directory not found: ${ONNXRUNTIME_ROOT}")
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ============================================================================
# TensorRT Configuration
# ============================================================================
set(TENSORRT_ROOT "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT library directory")

include_directories(${TENSORRT_ROOT})
link_directories(${TENSORRT_LIB_DIR})

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
find_library(NVPARSERS_LIB nvparsers PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)
find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_DIR} NO_DEFAULT_PATH)

if(NVINFER_LIB AND NVONNXPARSER_LIB)
    message(STATUS "Found TensorRT:")
    message(STATUS "  nvinfer: ${NVINFER_LIB}")
    message(STATUS "  nvonnxparser: ${NVONNXPARSER_LIB}")
    if(NVINFER_PLUGIN_LIB)
        message(STATUS "  nvinfer_plugin: ${NVINFER_PLUGIN_LIB}")
    endif()
    
    set(TENSORRT_LIBS
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
    )
    
    if(NVINFER_PLUGIN_LIB)
        list(APPEND TENSORRT_LIBS ${NVINFER_PLUGIN_LIB})
    endif()
    if(NVPARSERS_LIB)
        list(APPEND TENSORRT_LIBS ${NVPARSERS_LIB})
    endif()
else()
    message(FATAL_ERROR "TensorRT libraries not found in ${TENSORRT_LIB_DIR}")
endif()

# ============================================================================
# Include camera_models
# ============================================================================
include_directories("../camera_models/include" "./weights")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# ============================================================================
# Build vins_lib
# ============================================================================
add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/ICPCostFunction_p2p.cpp
    src/factor/ICPCostFunction_p2l.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/vins_node.cpp
)

# Link libraries to vins_lib
target_link_libraries(vins_lib  
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES} 
    ${YAML_CPP_LIBRARIES}
    ${ONNXRUNTIME_LIBS}
    ${TENSORRT_LIBS}
    ${CUDA_LIBRARIES}
    cudart
)

ament_target_dependencies(vins_lib 
    rclcpp 
    rcpputils 
    std_msgs 
    visualization_msgs 
    geometry_msgs 
    nav_msgs 
    tf2 
    tf2_ros 
    tf2_eigen 
    cv_bridge 
    camera_models 
    image_transport 
    visual_odometry_interfaces 
    sensor_msgs 
    geometry_msgs
)

# ============================================================================
# Build vins_node executable
# ============================================================================
add_executable(vins_node src/vins_node.cpp)

ament_target_dependencies(vins_node 
    rclcpp 
    rcpputils 
    message_filters 
    std_msgs 
    visualization_msgs 
    geometry_msgs 
    nav_msgs 
    tf2 
    tf2_ros 
    tf2_eigen 
    cv_bridge 
    camera_models 
    image_transport 
    visual_odometry_interfaces 
    sensor_msgs 
    geometry_msgs
)

# Link vins_node with vins_lib and all required libraries
target_link_libraries(vins_node 
    vins_lib
    ${ONNXRUNTIME_LIBS}
    ${TENSORRT_LIBS}
    ${CUDA_LIBRARIES}
    cudart
)

# ============================================================================
# Installation
# ============================================================================

# Install launch files
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

# Install nodes
install(
    TARGETS vins_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install binaries and libraries
install(
    TARGETS vins_node vins_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install additional directories
install(
    DIRECTORY
        launch
        config
    DESTINATION share/${PROJECT_NAME}
)

ament_package()