/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include "../tree.h"
using namespace std;

#include <eigen3/Eigen/Dense>
#include <Eigen/SVD>
using namespace Eigen;

// #include <ros/console.h>
#include <rcpputils/asserts.hpp>

#include "parameters.h"
#include "../utility/tic_toc.h"


#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR


class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        is_stereo = false;
    }
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    bool is_stereo;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class TreePerFrame
{
  public:
    TreePerFrame(const TreeNode &_tree, double td, int _frame_count)
    {
        point.x() = _tree.x; // position x
        point.y() = _tree.y; // position y
        point.z() = _tree.z; // position z
        velocity.x() = _tree.v_x; // velocity x
        velocity.y() = _tree.v_y; // velocity y
        velocity.z() = _tree.v_z; // velocity z
        n.x() = _tree.n_x; // unit vector to parent node x
        n.y() = _tree.n_y; // unit vector to parent node y
        n.z() = _tree.n_z; // unit vector to parent node z
        track_cnt; // tracking counter (how many consecutive frames we've seen the feature)
        cur_td = td;
        frame = _frame_count;
    }
    double cur_td;
    int frame;
    Vector3d point, velocity, n;
    int track_cnt;
};

class TreePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<TreePerFrame> tree_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    TreePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
    bool has_frame(int frame) const;
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    int get_tree_FeatureCount();
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    bool addFeatureTreeCheckParallax(int frame_count, const double header, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const pair<double, vector<TreeNode>> &tree, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void set_tree_Depth(const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    VectorXd get_tree_DepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    double computeErrorPointToLine(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Eigen::Vector3d>& model_axes, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) ;
    double computeError(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    std::pair<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, double> estimatePoseSVD(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Vector3d>& model_axes, double& cond_num);
    bool estimatePoseICPPrior(std::vector<Eigen::Vector3d> model, std::vector<Eigen::Vector3d> scene, const std::vector<Vector3d>& model_axes, int max_iterations, double convergence_threshold, Eigen::Matrix3d& R_init, Eigen::Vector3d& t_init, double& cond_num_final);
    Eigen::MatrixXd buildEpipolarConstraintMatrix( vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex, set<int> &tree_outlierIndex);
    void logMessage(const std::string& message); // DEBUG
    list<FeaturePerId> feature;
    list<TreePerId> t_feature;
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;

    double X_effective_max = 0;
    double last_track_num_plateau = 0;
    double noise_var_estimate = 50;
    double var_threshold = 0;
    double var_threshold_max = 0;

    const double gamma = 1e-4;
    const double vt_K = 2.5;
    const double alpha_slow = 0.03;
    const double alpha_fast = 0.3;
    const double betha = 0.01;
    
    double accumulator_timer = 0;
    const double accumulator_timer_thresh = 1000;
    const double delta_time_0 = 4.0; // in seconds
    const int min_track_num = 20;
    double prev_time = 0;
    

    
    double filtered_last_track_num = 0;
    double filtered_last_t_track_num = 0;
    double filtered_long_track_num = 0;
    double filtered_long_t_track_num = 0;
    double filtered_new_feature_num = 0;
    double filtered_new_t_feature_num = 0;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    double compensated_tree_Parallax2(const TreePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[2];
};

#endif