/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>
#include <sensor_msgs/msg/camera_info.hpp>
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>

using namespace std;

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

const double FOCAL_LENGTH = 805.0;
const int WINDOW_SIZE = 50;
const int NUM_OF_F = 10000; // IMPORTANT: set at least as WINDOW_SIZE * MAX_CNT
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern int USE_GPU;
extern int USE_GPU_ACC_FLOW;
extern int USE_GPU_CERES;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int IMU_FILTER;
extern double IMU_FILTER_ALPHA;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string IMAGE0_INFO_TOPIC, IMAGE1_INFO_TOPIC, IMU_FRAME;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int USE_TOPIC;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;

extern int USE_TREE;
extern std::string TREE_COLOR_TOPIC;
extern std::string TREE_DEPTH_TOPIC;
extern std::string TREE_DEPTH_INFO_TOPIC;
extern double DOWNSAMPLE_P;
extern Eigen::Matrix4d T_lcam_tree;
extern bool tree_lcam_flag;    
extern bool tree_baselink_flag;
extern Eigen::Vector3d base_link_z;
extern double TREE_DPP_DEPTH_SCALE;
extern double TREE_DPP_MIN_D;
extern double TREE_DPP_MAX_D;
extern int TREE_DPP_B_FILTER_F;
extern int TREE_DPP_T_FILTER_F;
extern double TREE_DPP_T_FILTER_A;
extern int TREE_DPP_HF_FILTER_F;
extern std::string YOLO_MODEL_PATH;
extern float TS_CONFIDENCE_T;
extern float TS_IOU_T;
extern double TS_MASK_T;
extern double H_CLUSTERING_T_SQ;
extern int DOWN_SKEL_K;
extern int MAX_T_CNT;
extern int TP_FD_LENGHT;
extern double TREE_METRIC_MATCH_THRESH;
extern double STAT_OUT_REJ_K;
extern double MIN_TREE_PARALLAX;
extern int ICP_P2L;
extern double TREE_OUTLIERS_TRESH;

extern std::map<std::string, sensor_msgs::msg::CameraInfo> ref_frame;
extern std::string mahalanobis_filePath;
extern Eigen::MatrixXd M_dist;

void readParameters(std::string config_file);

void read_mahalanobis_matrix(const std::string& filename);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
