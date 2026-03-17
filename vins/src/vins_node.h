#ifndef VINS_NODE_H
#define VINS_NODE_H

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/ximgproc.hpp>
#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include <set>
#include <unordered_map>
#include <numeric>


#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
// #include <sensor_msgs/image_encodings.h>
#include "utility/image_encodings.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.h>
#include <geometry_msgs/msg/transform_stamped.h>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include "tf2/exceptions.h"
#include <tf2_eigen/tf2_eigen.hpp>
#include "utility/CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "visual_odometry_interfaces/msg/vi_forest.hpp"
#include "visual_odometry_interfaces/msg/vi_tree.hpp"
#include "visual_odometry_interfaces/msg/vi_node.hpp"
#include "tree.h"

extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_seg_img_deb; // for debug

extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
extern rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;
extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_tree_match;
extern rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_tree_match_info;


#endif