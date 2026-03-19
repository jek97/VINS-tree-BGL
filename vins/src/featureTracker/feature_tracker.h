/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#define GPU_MODE 1


#include <cstdio>
#include <iostream>
#include <queue>
#include <string>
#include <fstream> // debug
#include <execinfo.h>
#include <csignal>
#include <random>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "opencv2/core.hpp"
#include <opencv2/core/eigen.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <random> 
#include <chrono> 

#ifdef GPU_MODE
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../tree.h"


using namespace std;
using namespace camodocal;
using namespace Eigen;


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    pair<double, vector<pair<int, ObservedTree>>> trackForest(double _cur_time, ObservedForest &cur_forest);
    void evaluate_fd(ObservedForest &forest);
    int hammingDistance(const std::vector<uint8_t>& fd_brief1, const std::vector<uint8_t>& fd_brief2);
    pair<double, vector<pair<pair<string, string>, double>>> isomorphism(ObservedTree tree_0, ObservedTree tree_1);
    void match(string node_0, string node_1, ObservedTree& graph_0, ObservedTree& graph_1, vector<pair<double, pair<string, string>>>& final_matches);
    ObservedTree subtree(const ObservedTree& tree, const string& node_id);
    ObservedTree extended_subtree(const ObservedTree& tree, const string& node_id);
    pair<vector<vector<double>>, vector<vector<double>>> tree_bipartite_capacity_cost_evaluation(const ObservedTree& graph_L, const ObservedTree& graph_R);
    class BpMatcher
    {
        public:
            BpMatcher(int nodeCount);
            
            bool search(int src, int sink);
            vector<double> getMaxFlow(vector<vector<double>>& capacity_mat, vector<vector<double>>& cost_mat, int src, int sink);
            vector<pair<double, pair<string, string>>> get_tree_matchings(const ObservedTree& graph_L, const ObservedTree& graph_R);
            vector<pair<int, vector<pair<pair<string, string>, double>>>> get_forest_matchings(const vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>>& tree_matches);
        
            int N = 0; // Stores the number of nodes
            vector<vector<double>> cap, flow, cost; // Stores the capacity, flow, cost per unit flow of each edge
            vector<double> dist, pi; // Stores the current shortest known distance from src to i and node potential
            vector<int> dad;
            vector<bool> found; // Stores the found edges
            const double INF = INT_MAX / 2 - 1;
    };
    void removeNode(const string& node, ObservedTree& graph);
    pair<vector<vector<double>>, vector<vector<double>>> forest_bipartite_capacity_cost_evaluation(vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>> tree_matches);
    vector<pair<int, vector<pair<pair<string, string>, double>>>> remove_statistical_outliers(const vector<pair<int, vector<pair<pair<string, string>, double>>>>& complete_matches);
    void setMask();
    void addPoints();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void setIntrinsicParameter_topic(const sensor_msgs::msg::CameraInfo &camera_info, const string camera_name);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    cv::Scalar genRandomColor();
    cv::Mat drawForest(const ObservedForest &forest, const Eigen::Matrix4d T_tree_lcam, const vector<cv::Scalar> circle_colors, const vector<cv::Scalar> line_colors);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds, set<int> &remove_t_PtsIds);
    cv::Mat getTrackImage();
    std::tuple<cv::Mat, sensor_msgs::msg::CameraInfo, double> getTreeMatch();
    bool inBorder(const cv::Point2f &pt);

    // log
    void logMessage(const std::string& message);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;

    ObservedForest prev_forest;
    ObservedForest last_model_forest; // world-frame model forest from the estimator, used for matching
    Eigen::Matrix3d last_R;           // last known robot rotation (world←IMU)
    Eigen::Vector3d last_P;           // last known robot position in world frame
    Eigen::Matrix3d last_ric;         // camera←IMU rotation extrinsic
    Eigen::Vector3d last_tic;         // camera←IMU translation extrinsic
    double _prev_time;
    int new_ids = 0;

    Eigen::Matrix3d K_mat; // defined to reproject the 3d point in the image for visualization purposes
    bool K_mat_f = false;
    std::tuple<cv::Mat, sensor_msgs::msg::CameraInfo, double> match_img; // variable to save the matching image and later pass it to the estimator and publish it
};
