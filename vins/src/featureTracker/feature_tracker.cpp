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

#include "feature_tracker.h"
#include <boost/graph/topological_sort.hpp>

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    has_tree_Prediction = false;
}

void FeatureTracker::setMask() // basically given the points for which we got a matching it prioritize the points that has been track for a long time, and remove the feature that are too close to each other
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255)); // init a white mask as big as the image

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++) // for all the current points
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i]))); // add them in the new variable

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         }); // The vector cnt_pts_id is sorted in descending order by track_cnt (how long each feature has been tracked). This ensures that features tracked for a longer time are prioritized.

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id) // For each feature in the sorted list
    {
        if (mask.at<uchar>(it.second.first) == 255) // If the feature's position on the mask is 255 (eligible), it is added back to cur_pts, ids, and track_cnt. indeed if the feature is closer to another then MIN_DIST this position in the mask will be black thanks to the drawing of the circle in the next lines
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); // draw a black circle of radius MIN_DIST around the feature
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts) // for all the new points obtained add them to the variables curr_pts (position), ids (ID), track_cnt (initialize their tracking count to 1)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{   
    // initialize variables
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear(); // clear the cur_pts vector

    if (prev_pts.size() > 0) // if you have some points from the previous iteration
    {
        vector<uchar> status;
        if(!USE_GPU_ACC_FLOW) // if not using the GPU
        {
            TicToc t_o;
            
            vector<float> err;
            // basically this if/else is an optimization of the code
            if(hasPrediction) // if you have some prediction, meaning that you projected some points of the previous iteration on the new frame 
            {
                cur_pts = predict_pts; // set as current points the projected ones
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW); // use the projected points as initial estimation for the optical flow (forced by the flag cv::OPTFLOW_USE_INITIAL_FLOW), enforcing the algorithm to find the previous points in the new frame
                
                int succ_num = 0; // init number of points for which a matching between previous and current frame as been found
                for (size_t i = 0; i < status.size(); i++) // for all the points given in input
                {
                    if (status[i]) // if the optical flow found a matching with the new frame
                        succ_num++; // increase the number of points for which a matching between previous and current frame as been found
                }
                if (succ_num < 10) // if it's less then 10
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3); // evaluate the optical flow but going deeper in the image pyramid while looking for matching
            }
            else 
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3); // evaluate directly the optical flow with a deeper image pyramid level
            // reverse check
            if(FLOW_BACK) // in this case an additional check is performed looking for the new points in the previous image to increase the algorithm precision 
            {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = prev_pts;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW); // evaluate the optical flow going back in time
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5) // if the point was matched in both the time directions and the distance of the back in time projected point with themself is smaller then 0.5 then keep it, otherwise remove it from the succesful matching
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("temporal optical flow costs: %fms\n", t_o.toc());
        }
#ifdef GPU_MODE // from here on same thing but done on GPU
        else
        {
            TicToc t_og;
            cv::cuda::GpuMat prev_gpu_img(prev_img);
            cv::cuda::GpuMat cur_gpu_img(cur_img);
            cv::cuda::GpuMat prev_gpu_pts(prev_pts);
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat gpu_status;
            if(hasPrediction)
            {
                cur_gpu_pts = cv::cuda::GpuMat(predict_pts);
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
                
                vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp_cur_pts);
                cur_pts = tmp_cur_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                int succ_num = 0;
                for (size_t i = 0; i < tmp_status.size(); i++)
                {
                    if (tmp_status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                {
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                    cur_gpu_pts.download(tmp1_cur_pts);
                    cur_pts = tmp1_cur_pts;

                    vector<uchar> tmp1_status(gpu_status.cols);
                    gpu_status.download(tmp1_status);
                    status = tmp1_status;
                }
            }
            else
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
            if(FLOW_BACK)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

                vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
                reverse_gpu_pts.download(reverse_pts);

                vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());
        }
#endif
    
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i])) // for all the newly obtained points, that are basically the previous points projected forward in time, if they are too close to the border remove them (note inborder return true if you are in the center, false if you are close to the border)
                status[i] = 0;
        reduceVector(prev_pts, status); // actually remove the points that passed all these selection criterias
        reduceVector(cur_pts, status); // actually remove the points that passed all these selection criterias
        reduceVector(ids, status); // actually remove the points that passed all these selection criterias
        reduceVector(track_cnt, status); // actually remove the points that passed all these selection criterias
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt) 
        n++;

    if (1)
    {
        //rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask(); // basically given the points for which we got a matching it prioritize the points that has been track for a long time, and remove the feature that are too close to each other
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // printf("set mask costs %fms\n", t_m.toc());
        ROS_DEBUG("detect feature begins");
        
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size()); // evaluate how many feature we can add to reach MAX_CNT features
        if(!USE_GPU)
        {
            if (n_max_cnt > 0) // if we have less then MAX_CNT features 
            {
                TicToc t_t;
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask); // find other n_max_cnt feature to get exactly MAX_CNT feature to track
                // printf("good feature to track costs: %fms\n", t_t.toc());
                //std::cout << "n_pts size: "<< n_pts.size()<<std::endl;
            }
            else // else keep the new feature variable empty
                n_pts.clear();
            // sum_n += n_pts.size();
            // printf("total point from non-gpu: %d\n",sum_n);
        }
#ifdef GPU_MODE // same but with GPU
        // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
        // printf("good feature to track costs: %fms\n", t_t.toc());
        else
        {
            if (n_max_cnt > 0)
            {
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                TicToc t_g;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat d_prevPts;
                TicToc t_gg;
                cv::cuda::GpuMat gpu_mask(mask);
                // printf("gpumat cost: %fms\n",t_gg.toc());
                cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(), MAX_CNT - cur_pts.size(), 0.01, MIN_DIST);
                // cout << "new gpu points: "<< MAX_CNT - cur_pts.size()<<endl;
                detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
                // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
                if(!d_prevPts.empty())
                    n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
                else
                    n_pts.clear();
                // sum_n += n_pts.size();
                // printf("total point from gpu: %d\n",sum_n);
                // printf("gpu good feature to track cost: %fms\n", t_g.toc());
            }
            else 
                n_pts.clear();
        }
#endif

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints(); // for all the new points obtained add them to the variables curr_pts (position), ids (ID), track_cnt (initialize their tracking count to 1)
        // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        // printf("selectFeature costs: %fms\n", t_a.toc());
    }
    
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]); // project the 3D points to the image plane, in the case of 2D points (as here) the points are undistorted following the camera parameters
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map); // evaluate the point velocity in the image or init it to 0 if it's the first time you see them

    if(!_img1.empty() && stereo_cam) // if we are in the stereo camera case
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty()) // if there are points obtained in the left camera
        {
            //printf("stereo image; track feature on right image\n");
            
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            if(!USE_GPU_ACC_FLOW) // if you don't use the gpu
            {
                TicToc t_check;
                vector<float> err;
                // cur left ---- cur right
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3); // search a matching of the points of the left camera in the right one
                // reverse check cur right ---- cur left
                if(FLOW_BACK) // additional check as done for the left image view in time, here done in space between left and right camera
                {
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5) // [don't understand the inborder function it should be negated ?]
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("left right optical flow cost %fms\n",t_check.toc());
            }
#ifdef GPU_MODE // same but with GPU
            else
            {
                TicToc t_og1;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat right_gpu_Img(rightImg);
                cv::cuda::GpuMat cur_gpu_pts(cur_pts);
                cv::cuda::GpuMat cur_right_gpu_pts;
                cv::cuda::GpuMat gpu_status;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_Img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
                cur_right_gpu_pts.download(tmp_cur_right_pts);
                cur_right_pts = tmp_cur_right_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                if(FLOW_BACK)
                {   
                    cv::cuda::GpuMat reverseLeft_gpu_Pts;
                    cv::cuda::GpuMat status_gpu_RightLeft;
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(right_gpu_Img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                    vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                    reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                    reverseLeftPts = tmp_reverseLeft_Pts;

                    vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                    status_gpu_RightLeft.download(tmp1_status);
                    statusRightLeft = tmp1_status;
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
            }
#endif
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status); // [probably here the idea was to discard points that don't match with the right one, keeping only points that are in both the images, i think it needs to be on ?]
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]); // undistort the point
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map); // obtain their velocity in the image
            
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if(SHOW_TRACK) // if we want to show the track of the point between current and last frame
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    
    // set current variable as prev variable for next iteration
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear(); // variable used to draw the track
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame; // create the feature frame to fill in the buffer
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam) // if we are in the stereocamera case fill also another feature to insert in the feature buffer with the right features
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}


cv::Mat FeatureTracker::drawForest(const vector<vector<Ex_TreeNode>> &forest, const Eigen::Matrix4d T_tree_lcam, const vector<cv::Scalar> circle_colors, const vector<cv::Scalar> line_colors)
{
    cv::Mat img(ref_frame["tree_camera"].height, ref_frame["tree_camera"].width, CV_8UC3, cv::Scalar(255, 255, 255));

    // fill each image with a circle for each node and a line for each edge
    for(size_t i = 0; i < forest.size(); i++)
    {
        for (const auto& node : forest[i])
        {   
            // evaluate the point position in the image 
            // get the 3d position
            Eigen::Vector4d p_3d_h;
            p_3d_h << node.x, node.y, node.z, 1;
                                
            // project the point in the tree view
            Eigen::Vector4d p_3dt_h = T_tree_lcam * p_3d_h;
            Eigen::Vector3d p_3dt = p_3dt_h.block<3, 1>(0, 0);
            
            // project in the image plane
            Eigen::Vector3d p_uvs = K_mat * p_3dt;
            Eigen::Vector2d p_img;
            p_img << p_uvs[0] / p_uvs[2], p_uvs[1] / p_uvs[2];

            // convert point in cv format
            cv::Point p_img_cv(static_cast<int>(p_img.x()), static_cast<int>(p_img.y()));
            
            // draw a circle for the node
            cv::circle(img, p_img_cv, 3, circle_colors[i], -1);

            // draw also the edge to the sons nodes
            for (const auto& son : node.ex_sons) // given the sons
            { 
                // find the son in the vector
                auto it = std::find_if(forest[i].begin(), forest[i].end(), 
                [&son](const Ex_TreeNode& n) { return n.ex_id == son; });

                if (it != forest[i].end()) {  // If a matching node is found
                // evaluate the point position in the image 
                // get the 3d position
                Eigen::Vector4d p_3d_h_son;
                p_3d_h_son << it->x, it->y, it->z, 1;

                // project the point in the tree view
                Eigen::Vector4d p_3dt_h_son = T_tree_lcam * p_3d_h_son;
                Eigen::Vector3d p_3dt_son = p_3dt_h_son.block<3, 1>(0, 0);
                
                // project in the image plane
                Eigen::Vector3d p_uvs_son = K_mat * p_3dt_son;
                Eigen::Vector2d p_img_son;
                p_img_son << p_uvs_son[0] / p_uvs_son[2], p_uvs_son[1] / p_uvs_son[2];

                // convert point in cv format
                cv::Point p_img_cv_son(static_cast<int>(p_img_son.x()), static_cast<int>(p_img_son.y()));

                // draw a line for the son
                cv::line(img, p_img_cv, p_img_cv_son, line_colors[i], 2, cv::LINE_8);
        
                }
            }
        } 
    }
    

    return img;
}

void FeatureTracker::evaluate_fd(ObservedForest& forest)
{
    using VD = boost::graph_traits<ObservedTree>::vertex_descriptor;

    for (auto& tree : forest)
    {
        const auto nv = boost::num_vertices(tree);
        if (nv == 0) continue;

        // boost::topological_sort with back_inserter outputs vertices in DFS
        // finish order: sinks (tips) are finished first, source (root) last.
        // This gives the exact bottom-up order we need without sentinel tricks.
        std::vector<VD> order;
        order.reserve(nv);
        boost::topological_sort(tree, std::back_inserter(order));

        // desc[v] = all descendant vertex descriptors of v, filled bottom-up.
        // With vecS, VD is size_t so plain vector indexing is valid.
        std::vector<std::vector<VD>> desc(nv);

        for (VD v : order)
        {
            auto& nd = tree[v];
            nd.fd.clear();

            if (boost::out_degree(v, tree) == 0)
            {
                // Tip: no children, no branching structure → zero descriptor.
                nd.fd.assign(TP_FD_LENGHT, 0.0);
                continue;   // desc[v] stays empty
            }

            std::vector<double> fd;
            fd.reserve(boost::out_degree(v, tree));

            for (auto e : boost::make_iterator_range(boost::out_edges(v, tree)))
            {
                const VD c = boost::target(e, tree);

                // Subtree of c: {c} followed by all of c's descendants.
                // desc[c] is already populated because c was processed earlier
                // (topological order guarantees children before parents).
                std::vector<VD> sub;
                sub.reserve(1 + desc[c].size());
                sub.push_back(c);
                sub.insert(sub.end(), desc[c].begin(), desc[c].end());

                const int sub_n = static_cast<int>(sub.size());

                // Map vertex descriptor → adjacency-matrix row/column index.
                std::unordered_map<VD, int> idx;
                idx.reserve(sub_n);
                for (int i = 0; i < sub_n; ++i)
                    idx[sub[i]] = i;

                // Build undirected adjacency matrix in O(|sub|) via BGL out-edges.
                // The old code did this with O(n² · max_degree) nested find() calls.
                Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(sub_n, sub_n);
                for (int i = 0; i < sub_n; ++i)
                {
                    for (auto oe : boost::make_iterator_range(boost::out_edges(sub[i], tree)))
                    {
                        auto jt = idx.find(boost::target(oe, tree));
                        if (jt != idx.end())
                        {
                            adj(i, jt->second) = 1.0;
                            adj(jt->second, i) = 1.0;  // undirected
                        }
                    }
                }

                // Eigendecomposition of the symmetric adjacency matrix.
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(adj);
                const Eigen::VectorXd& eig = solver.eigenvalues();
                std::vector<double> eig_vals(eig.data(), eig.data() + eig.size());
                std::sort(eig_vals.begin(), eig_vals.end(), std::greater<double>());

                // Contribution of child c: sum of its top-k eigenvalues,
                // where k = number of children of c (its out-degree).
                const int k = std::min(static_cast<int>(boost::out_degree(c, tree)),
                                       static_cast<int>(eig_vals.size()));
                fd.push_back(std::accumulate(eig_vals.begin(), eig_vals.begin() + k, 0.0));
            }

            // Sort descending; resize() truncates if too long, pads 0 if too short.
            std::sort(fd.begin(), fd.end(), std::greater<double>());
            fd.resize(TP_FD_LENGHT, 0.0);
            nd.fd = std::move(fd);

            // Propagate descendants upward: desc[v] = children ∪ their descendants.
            for (auto e : boost::make_iterator_range(boost::out_edges(v, tree)))
            {
                const VD c = boost::target(e, tree);
                desc[v].push_back(c);
                desc[v].insert(desc[v].end(), desc[c].begin(), desc[c].end());
            }
        }
    }
}

ObservedTree FeatureTracker::subtree(const ObservedTree& tree, const string& node_id)
{
    using VD = boost::graph_traits<ObservedTree>::vertex_descriptor;

    // Locate the root vertex by ex_id.
    auto [vs_begin, vs_end] = boost::vertices(tree);
    auto root_it = std::find_if(vs_begin, vs_end,
        [&](VD v){ return tree[v].ex_id == node_id; });

    if (root_it == vs_end) {
        std::cerr << "Warning featuretracker subtree: ex_id '" << node_id << "' not found.\n";
        return ObservedTree{};
    }
    const VD root_vd = *root_it;

    ObservedTree sub;
    std::unordered_map<VD, VD> vd_map; // source VD -> sub VD (excludes root)

    // BFS: collect all descendants, add as vertices to sub.
    // Root itself is excluded (matches original behaviour).
    std::queue<VD> q;
    q.push(root_vd);
    while (!q.empty()) {
        VD cur = q.front(); q.pop();
        if (cur != root_vd)
            vd_map[cur] = boost::add_vertex(tree[cur], sub);
        for (auto e : boost::make_iterator_range(boost::out_edges(cur, tree)))
            q.push(boost::target(e, tree));
    }

    // Add edges between descendant vertices (edges from root to its direct
    // children are excluded because root is not in vd_map).
    for (const auto& [src_vd, sub_src] : vd_map) {
        for (auto e : boost::make_iterator_range(boost::out_edges(src_vd, tree))) {
            auto jt = vd_map.find(boost::target(e, tree));
            if (jt != vd_map.end())
                boost::add_edge(sub_src, jt->second, sub);
        }
    }
    return sub;
}

ObservedTree FeatureTracker::extended_subtree(const ObservedTree& tree, const string& node_id)
{
    using VD = boost::graph_traits<ObservedTree>::vertex_descriptor;

    // Find the component of node_id and collect all distinct components.
    int node_component = -1;
    std::set<int> components;
    for (auto v : boost::make_iterator_range(boost::vertices(tree))) {
        components.insert(tree[v].component);
        if (tree[v].ex_id == node_id)
            node_component = tree[v].component;
    }

    if (components.size() <= 1)
        return subtree(tree, node_id);

    // Multi-component: normal subtree + isolated vertices from other components.
    ObservedTree sub = subtree(tree, node_id);
    for (auto v : boost::make_iterator_range(boost::vertices(tree)))
        if (tree[v].component != node_component)
            boost::add_vertex(tree[v], sub);
    return sub;
}


int FeatureTracker::hammingDistance(const std::vector<uint8_t>& fd_brief1, const std::vector<uint8_t>& fd_brief2) 
{
    int distance = 0;
    for (size_t i = 0; i < fd_brief1.size(); ++i) {
        // XOR the bytes and count set bits
        uint8_t xor_result = fd_brief1[i] ^ fd_brief2[i];
        distance += __builtin_popcount(xor_result);
    }
    return distance;
}

pair<vector<vector<double>>, vector<vector<double>>> FeatureTracker::tree_bipartite_capacity_cost_evaluation(const ObservedTree& graph_L, const ObservedTree& graph_R)
{
    // vecS vertex container guarantees VDs are consecutive 0..N-1, so graph[i] == vertex i directly.
    const int nL = static_cast<int>(boost::num_vertices(graph_L));
    const int nR = static_cast<int>(boost::num_vertices(graph_R));

    // create cost matrix and capacity matrix, given cost[i][j] = cost edge connecting node i to node j, NOTE node i, j = 0 = source , node i, j = last = sink
    vector<vector<double>> cost_mat(nL + nR + 2, std::vector<double>(nL + nR + 2, 0.0));
    vector<vector<double>> capacity_mat(nL + nR + 2, std::vector<double>(nL + nR + 2, 0.0)); // first and last elements are source and sink

    // capacity matrix: all zeroes except for the edges connecting the L nodes (graph_L) to the R nodes (graph_R), the edges connecting source to the L nodes (graph_L) and the edges connecting the R nodes (graph_R) to sink
    // cost matrix: all zeroes except for the edges connecting the L nodes (graph_L) to the R nodes (graph_R)
    for(int i = 0; i < nL; ++i)
    {
        const ObservedNode& nl = graph_L[i];
        for(int j = 0; j < nR; ++j)
        {
            const ObservedNode& nr = graph_R[j];

            // extract point position
            Eigen::Vector3d p_0(nl.x, nl.y, nl.z);
            Eigen::Vector3d p_1(nr.x, nr.y, nr.z);

            // project them in the image
            Eigen::Vector3d prev_node_uv = K_mat * p_0;
            Eigen::Vector2d prev_node_2d_p(prev_node_uv[0]/prev_node_uv[2], prev_node_uv[1]/prev_node_uv[2]);

            Eigen::Vector3d cur_node_uv = K_mat * p_1;
            Eigen::Vector2d cur_node_2d_p(cur_node_uv[0]/cur_node_uv[2], cur_node_uv[1]/cur_node_uv[2]);

            // distance
            double dist_2d = (cur_node_2d_p - prev_node_2d_p).norm();

            // weight it based on the average depth of the points
            double prev_depth = p_0.norm();
            double cur_depth = p_1.norm();

            // get their topological feature descriptor
            Eigen::VectorXd prev_topological_fd = Eigen::Map<const Eigen::VectorXd>(nl.fd.data(), nl.fd.size());
            Eigen::VectorXd cur_topological_fd  = Eigen::Map<const Eigen::VectorXd>(nr.fd.data(), nr.fd.size());

            // evaluate distance
            double topological_distance = (cur_topological_fd - prev_topological_fd).norm();

            // final distance
            double custom_dist = (dist_2d * ((prev_depth + cur_depth) / 2)) + topological_distance;
            cost_mat[i + 1][nL + 1 + j] = custom_dist;

            // capacity matrix
            if((p_0 - p_1).norm() <= TREE_METRIC_MATCH_THRESH) // if the position error is too high assign 0 capacity on that edge, avoiding the match
                capacity_mat[i + 1][nL + 1 + j] = 1; // edges from L to R
            else
                capacity_mat[i + 1][nL + 1 + j] = 0; // edges from L to R
        }
    }

    for(int i = 0; i < nL; ++i)
        capacity_mat[0][i + 1] = 1; // edges from source to L

    for(int i = 0; i < nR; ++i)
        capacity_mat[nL + 1 + i][nL + nR + 1] = 1; // edges from R to sink

    return {capacity_mat, cost_mat};
}

FeatureTracker::BpMatcher::BpMatcher(int nodeCount)
{   
    // source: https://www.geeksforgeeks.org/dsa/minimum-cost-maximum-flow-from-a-graph-using-bellman-ford-algorithm/
    N = nodeCount;
    cap.assign(N, vector<double>(N, 0.0));
    flow.assign(N, vector<double>(N, 0.0));
    cost.assign(N, vector<double>(N, 0.0));
    dist.assign(N + 1, 0.0);
    pi.assign(N, 0.0);
    dad.assign(N, 0);
    found.assign(N, false);
}


bool FeatureTracker::BpMatcher::search(int src, int sink) {
    // Initialise found[] to false
    found = vector<bool>(N, false);

    // Initialise the dist[] to INF
    dist = vector<double>(N + 1, INF); 
    
    // Distance from the source node
    dist[src] = 0.0;

    // Iterate until src reaches N
    while (src != N) {
        int best = N;
        found[src] = true;

        for (int k = 0; k < N; k++) {
            // If already found
            if (found[k]) 
                continue;

            // Evaluate while flow is still in supply
            if (flow[k][src] != 0) {

                // Obtain the total value
                double val = dist[src] + pi[src] - pi[k] - cost[k][src];

                // If dist[k] is > minimum value
                if (dist[k] > val) {

                    // Update
                    dist[k] = val;
                    dad[k] = src;
                }
            }

            if (flow[src][k] < cap[src][k]) {
                double val = dist[src] + pi[src] - pi[k] + cost[src][k];

                // If dist[k] is > minimum value
                if (dist[k] > val) {

                    // Update
                    dist[k] = val;
                    dad[k] = src;
                }
            }

            if (dist[k] < dist[best])
                best = k;
        }

        // Update src to best for
        // next iteration
        src = best;
    }

    for (int k = 0; k < N; k++)
        pi[k] = min(pi[k] + dist[k], INF);

    // Return the value obtained at sink
    return found[sink];
}

vector<double> FeatureTracker::BpMatcher::getMaxFlow(vector<vector<double>>& capacity_mat, vector<vector<double>>& cost_mat, int src, int sink) 
{   
    cap = capacity_mat;
    cost = cost_mat;
    N = cap.size();
    found = vector<bool>(N, false);
    flow.assign(N, vector<double>(N, 0.0));
    dist = vector<double>(N + 1, 0.0);
    dad = vector<int>(N, 0);
    pi = vector<double>(N, 0.0);

    double totalflow = 0.0, totalcost = 0.0;

    // If a path exists from src to sink
    while (search(src, sink)) {
        // Set the default amount
        double amt = INF;
        int x = sink;

        while (x != src) {
            amt = min(
                amt, flow[x][dad[x]] != 0
                         ? flow[x][dad[x]]
                         : cap[dad[x]][x] - flow[dad[x]][x]);
            x = dad[x];
        }

        x = sink;

        while (x != src) {
            if (flow[x][dad[x]] != 0) {
                flow[x][dad[x]] -= amt;
                totalcost -= amt * cost[x][dad[x]];
            } else {
                flow[dad[x]][x] += amt;
                totalcost += amt * cost[dad[x]][x];
            }

            x = dad[x];
        }

        totalflow += amt;
    }

    // Return pair total cost and sink
    return {totalflow, totalcost};
}

vector<pair<double, pair<string, string>>> FeatureTracker::BpMatcher::get_tree_matchings(const ObservedTree& graph_L, const ObservedTree& graph_R)
{
    // vecS: VDs are 0..N-1, direct integer access.
    const int nL = static_cast<int>(boost::num_vertices(graph_L));
    const int nR = static_cast<int>(boost::num_vertices(graph_R));

    vector<pair<double, pair<string, string>>> matches; // a vector in which each element is cost of the matching, node of graph L, node of graph R
    for(int i = 0; i < nL; ++i) // for all the edges going from graph_L to graph_R
    {
        for(int j = 0; j < nR; ++j)
        {
            if (flow[i + 1][nL + 1 + j] > 0) // if you have flow in the solution, meaning we have a matching between the two nodes
            {
                matches.push_back(make_pair(cost[i + 1][nL + 1 + j], make_pair(graph_L[i].ex_id, graph_R[j].ex_id)));
            }
        }
    }

    return matches;
}

vector<pair<int, vector<pair<pair<string, string>, double>>>> FeatureTracker::BpMatcher::get_forest_matchings(const vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>>& tree_matches)
{   
    vector<pair<int, vector<pair<pair<string, string>, double>>>> matches(tree_matches.size(), {-1, {{{"", ""}, 0.0}}}); // initialize the matches datasturcture, for every tree in cur_forest, matched tree in prev_forest (by index) plus a vector of node matches (node in tree in cur_forest, node in tree in prev_forest)
    for(int i = 0; i < tree_matches.size(); ++i) // for all the edges going from cur_forest to prev_forest
    {
        for(int j = 0; j < tree_matches[0].size(); ++j)
        {
            if (flow[i + 1][tree_matches.size() + 1 + j] > 0) // if you have flow in the solution, meaning we have a matching between the twwo trees
            {   
                matches[i] = make_pair(j, tree_matches[i][j].second);
            }
        }
    }
    
    return matches;
}

void FeatureTracker::removeNode(string node, vector<Ex_TreeNode>& graph)
{
    // delete nodes matched on graph_0
    // find the matched element m_0 in graph_0
    auto it = std::find_if(graph.begin(), graph.end(), [&](const Ex_TreeNode& n) {
        return n.ex_id == node;
    });

    if (it != graph.end()) {
        // find the parent of m_0, p_m_0
        auto it_p = std::find_if(graph.begin(), graph.end(), [&](const Ex_TreeNode& n) {
            return n.ex_id == it->ex_parent;
        });
        
        // delete m_0 from the sons of its parent p_m_0
        if(it_p != graph.end())
        {
            auto& sons = it_p->ex_sons;
            sons.erase(std::remove(sons.begin(), sons.end(), it->ex_id), sons.end());
        }

        // for every son of m_0, s_m_0
        for(const auto& s : it->ex_sons)
        {
            auto it_s = std::find_if(graph.begin(), graph.end(), [&](const Ex_TreeNode& n) {
                return n.ex_id == s;
            });
            
            // delete m_0 from the parents of its sons s_m_0
            if (it_s != graph.end()) {
                it_s->ex_parent = "";  
            }
        }

        // finally delete the node m_0
        graph.erase(it);

    } else {
        //std::cout << "ERROR deleting node " << node << " from graph" << std::endl;
    }
}

void FeatureTracker::removeNode(const string& node_id, ObservedTree& graph)
{
    using VD = boost::graph_traits<ObservedTree>::vertex_descriptor;

    auto [vs_begin, vs_end] = boost::vertices(graph);
    auto it = std::find_if(vs_begin, vs_end,
        [&](VD v){ return graph[v].ex_id == node_id; });

    if (it == vs_end) return;

    const VD v = *it;
    boost::clear_vertex(v, graph);   // remove all incident edges first
    boost::remove_vertex(v, graph);  // then remove the vertex (renumbers VDs >= v)
}

void FeatureTracker::match(string node_0, string node_1, ObservedTree& graph_0, ObservedTree& graph_1, vector<pair<double, pair<string, string>>>& final_matches)
{   
    // get the subtree rooted at node_0 in graph_0 and at node_1 in graph_1
    ObservedTree sub_0 = subtree(graph_0, node_0);
    ObservedTree sub_1 = subtree(graph_1, node_1);

    while(boost::num_vertices(sub_0) > 0 && boost::num_vertices(sub_1) > 0)
    {   
        // evaluate capacity and cost matrix for minimum cost maximum cardinality bipartite matching
        auto [capacity, cost] = tree_bipartite_capacity_cost_evaluation(sub_0, sub_1);
        
        // find minimum cost maximum cardinality bipartite matching
        FeatureTracker::BpMatcher matcher(capacity.size());
        vector<double> ret = matcher.getMaxFlow(capacity, cost, 0, capacity.size() -1);
        vector<pair<double, pair<string, string>>> matches = matcher.get_tree_matchings(sub_0, sub_1);
        
        
        // get matching with minimum cost
        pair<double, pair<string, string>> mc_match;
        auto min_cost_match = min_element(matches.begin(), matches.end(), [](const auto& a, const auto& b){
            return a.first < b.first;
        });
        
        if(min_cost_match != matches.end())
        {
            mc_match = make_pair(min_cost_match->first, make_pair(min_cost_match->second.first, min_cost_match->second.second));

            final_matches.push_back(mc_match); // save it in the final result
        }
        else
        {
            //std::cerr << "ERROR featuretracker match: no minimum weight matching" << std::endl;
            
            string par_db;
            vector<string> sons_db;
            for(const auto& m : final_matches)
            {   
                removeNode(m.second.first, graph_0);
                removeNode(m.second.first, sub_0);
                removeNode(m.second.second, graph_1);
                removeNode(m.second.second, sub_1);
            }
            return;
        }
        
        // call again matching on the subtree rooted at the new minimum cost matching
        match(mc_match.second.first, mc_match.second.second, graph_0, graph_1, final_matches);
        
        // remove the nodes aready present in the final matching list
        for(const auto& m : final_matches)
        {   
            removeNode(m.second.first, graph_0);
            removeNode(m.second.first, sub_0);
            removeNode(m.second.second, graph_1);
            removeNode(m.second.second, sub_1);
        }
    }
    
    return;
}

pair<double, vector<pair<pair<string, string>, double>>> FeatureTracker::isomorphism(ObservedTree tree_0, ObservedTree tree_1)
{
    // Trees are passed by value (copy) so match() can mutate them freely.
    using VD = boost::graph_traits<ObservedTree>::vertex_descriptor;

    vector<pair<double, pair<string, string>>> matches;

    // Find the root of each tree: the unique vertex with in_degree == 0.
    // std::find_if over the vertex range avoids an explicit for loop and
    // the old "ex_parent == root" string sentinel.
    auto [vs0_begin, vs0_end] = boost::vertices(tree_0);
    const string node_0 = tree_0[*std::find_if(vs0_begin, vs0_end,
        [&](VD v){ return boost::in_degree(v, tree_0) == 0; })].ex_id;

    auto [vs1_begin, vs1_end] = boost::vertices(tree_1);
    const string node_1 = tree_1[*std::find_if(vs1_begin, vs1_end,
        [&](VD v){ return boost::in_degree(v, tree_1) == 0; })].ex_id;

    match(node_0, node_1, tree_0, tree_1, matches);


    // evaluate final match total cost
    pair<double, vector<pair<pair<string, string>, double>>> final_match;
    double total_cost = 0;
    int n_matches = 0;
    
    for(const auto& m : matches)
    {   
        total_cost += m.first;
        n_matches += 1;
        final_match.second.push_back(make_pair(make_pair(m.second.first, m.second.second), m.first));
    }
    final_match.first = total_cost / n_matches;
    return final_match;
}

pair<vector<vector<double>>, vector<vector<double>>> FeatureTracker::forest_bipartite_capacity_cost_evaluation(vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>> tree_matches)
{
    // create cost matrix and capacity matrix, given cost[i][j] = cost edge connecting tree i to tree j, NOTE tree i, j = 0 = source , tree i, j = last = sink 
    vector<vector<double>> cost_mat(tree_matches.size() + tree_matches[0].size() + 2, std::vector<double>(tree_matches.size() + tree_matches[0].size() + 2, 0.0));
    vector<vector<double>> capacity_mat(tree_matches.size() + tree_matches[0].size() + 2, std::vector<double>(tree_matches.size() + tree_matches[0].size() + 2, 0.0)); // first and last elements are source and sink
    
    // capacity matrix: all zeroes except for the edges connecting the L trees (cur_forest) to the R trees (prev_forest), the edges connecting source to the L trees (cur_forest) and the edges connecting the R trees (prev_forest) to sink
    // cost matrix: all zeroes except for the edges connecting the L trees (cur_forest) to the R trees (prev_forest)

    for(int i = 0; i < tree_matches.size(); ++i)
    {
        for(int j = 0; j < tree_matches[0].size(); ++j)
        {               
            // assign tree matching cost
            cost_mat[i + 1][tree_matches.size() + 1 + j] = tree_matches[i][j].first; // edges from L to R

            // capacity matrix
            capacity_mat[i + 1][tree_matches.size() + 1 + j] = 1; // edges from L to R
        }
    }

    for(int i = 0; i < tree_matches.size(); ++i)
    {
        // capacity matrix
        capacity_mat[0][i + 1] = 1; // edges from source to L
    }

    for(int i = 0; i < tree_matches[0].size(); ++i)
    {
        // capacity matrix
        capacity_mat[tree_matches.size() + 1 + i][tree_matches.size() + tree_matches[0].size() + 1] = 1; // edges from R to sink
    }

    return {capacity_mat, cost_mat};
}

vector<pair<int, vector<pair<pair<string, string>, double>>>> FeatureTracker::remove_statistical_outliers(const vector<pair<int, vector<pair<pair<string, string>, double>>>>& complete_matches)
{
    // Welford's online algorithm for mean and variance
    int n_samples = 0;
    double mean = 0.0;
    double M2 = 0.0;
    double std_dev = 0.0;

    for (const auto& tree : complete_matches)
    {
        if (tree.first != -1)
            for (const auto& match : tree.second)
            {
                ++n_samples;
                double delta = match.second - mean;
                mean += delta / n_samples;
                M2 += delta * (match.second - mean);
            }
    }

    if (n_samples <= 1)
        return complete_matches;

    std_dev = std::sqrt(M2 / (n_samples - 1));

    const double up_thresh   = mean + std_dev * STAT_OUT_REJ_K;
    const double down_thresh = mean - std_dev * STAT_OUT_REJ_K;

    auto filtered_matches = complete_matches;

    for (auto& tree : filtered_matches)
    {
        if (tree.first != -1)
        {
            auto& vec = tree.second;
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const auto& m){ return m.second >= up_thresh || m.second <= down_thresh; }),
                vec.end());
        }
    }

    return filtered_matches;
}

void FeatureTracker::logMessage(const std::string& message) 
{
    const std::string LOG_FILE_PATH = "/home/glugano/Desktop/log.txt";

    std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
    if (!logFile) {
        std::cerr << "Error: Unable to open log file." << std::endl;
        return;
    }

    logFile << message << std::endl;
}

cv::Scalar FeatureTracker::genRandomColor()
{
    // Seed the random number generator using the current time
    std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    // Define a distribution for values between 0 and 255
    std::uniform_int_distribution<int> dist(0, 255);

    // Generate random values for Blue, Green, and Red channels
    int b = dist(rng);
    int g = dist(rng);
    int r = dist(rng);

    return cv::Scalar(b, g, r);
}

pair<double, vector<TreeNode>> FeatureTracker::trackForest(double _cur_time, vector<vector<Ex_TreeNode>> &cur_forest)
{        
    // save the K matrix for point visualization (in track tree)
    if(!K_mat_f)
    {
        Eigen::VectorXd K_vec(9);
        K_vec << ref_frame["tree_camera"].k[0], 0.0, ref_frame["tree_camera"].k[2], 0.0, ref_frame["tree_camera"].k[4], ref_frame["tree_camera"].k[5], 0.0, 0.0, 1.0;
        K_mat = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_vec.data());
    }
    
    // TODO: adapt trackForest to accept ObservedForest (deferred – next refactor step).
    // evaluate_fd(cur_forest);
    
    vector<pair<int, vector<pair<pair<string, string>, double>>>> complete_matches;
    vector<int> matches_names;
    if(prev_forest.size() > 0)
    {          
        if(has_tree_Prediction)
        {   
            // do matching with predicted forest
            vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>> tree_matches(cur_forest.size(), std::vector<std::pair<double, std::vector<std::pair<std::pair<std::string, std::string>, double>>>>(predict_forest_pts.size(), {0.0, {{{"", ""}, 0.0}}})); // initialize the array that will keep matching and their cost for each combination of tree between previous and current forest
            
            for(size_t i = 0; i < cur_forest.size(); ++i) // for every tree in current forest
            {
                for(size_t j = 0; j < predict_forest_pts.size(); ++j) // for every tree in previous forest
                {   
                    // TODO: adapt call for ObservedForest (deferred – trackForest migration)
                    // tree_matches[i][j] = isomorphism(cur_forest[i], predict_forest_pts[j]);
                }
            }

            // evaluate capacity and cost matrix for minimum cost maximum cardinality bipartite matching
            auto [capacity, cost] = forest_bipartite_capacity_cost_evaluation(tree_matches);
            
            // find minimum cost maximum cardinality bipartite matching
            FeatureTracker::BpMatcher matcher(capacity.size());
            vector<double> ret = matcher.getMaxFlow(capacity, cost, 0, capacity.size() -1);

            complete_matches = matcher.get_forest_matchings(tree_matches);
            
            // evaluate number of node matched
            int total_matches = 0;
            for(size_t i = 0; i < cur_forest.size(); ++i){ // for all the current trees
                if(complete_matches[i].first != -1) // if you get a matching for that tree
                {
                    total_matches += complete_matches[i].second.size(); // sum all the node matching for that tree
                }                
            }

            if(total_matches < 10) // if you have less then _ matchings (tree wise matching)
            {   
                complete_matches.clear(); // clear previous matchings

                // run normal matching
                vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>> tree_matches(cur_forest.size(), std::vector<std::pair<double, std::vector<std::pair<std::pair<std::string, std::string>, double>>>>(prev_forest.size(), {0.0, {{{"", ""}, 0.0}}})); // initialize the array that will keep matching and their cost for each combination of tree between previous and current forest
                
                for(size_t i = 0; i < cur_forest.size(); ++i) // for every tree in current forest
                {
                    for(size_t j = 0; j < prev_forest.size(); ++j) // for every tree in previous forest
                    {   
                        // TODO: adapt call for ObservedForest (deferred – trackForest migration)
                        // tree_matches[i][j] = isomorphism(cur_forest[i], prev_forest[j]);
                    }
                }

                // evaluate capacity and cost matrix for minimum cost maximum cardinality bipartite matching
                auto [capacity, cost] = forest_bipartite_capacity_cost_evaluation(tree_matches);
                
                // find minimum cost maximum cardinality bipartite matching
                FeatureTracker::BpMatcher matcher(capacity.size());
                vector<double> ret = matcher.getMaxFlow(capacity, cost, 0, capacity.size() -1);

                complete_matches = matcher.get_forest_matchings(tree_matches);
            }
        }
        else
        {   
            // run normal matching
            vector<vector<pair<double, vector<pair<pair<string, string>, double>>>>> tree_matches(cur_forest.size(), std::vector<std::pair<double, std::vector<std::pair<std::pair<std::string, std::string>, double>>>>(prev_forest.size(), {0.0, {{{"", ""}, 0.0}}})); // initialize the array that will keep matching and their cost for each combination of tree between previous and current forest
            
            for(size_t i = 0; i < cur_forest.size(); ++i) // for every tree in current forest
            {
                for(size_t j = 0; j < prev_forest.size(); ++j) // for every tree in previous forest
                {   
                    tree_matches[i][j] = isomorphism(cur_forest[i], prev_forest[j]); // evaluate largest isomorphism, obtaining nodes matchings and related cost
                }
            }
            
            // evaluate capacity and cost matrix for minimum cost maximum cardinality bipartite matching
            auto [capacity, cost] = forest_bipartite_capacity_cost_evaluation(tree_matches);
            
            // find minimum cost maximum cardinality bipartite matching
            FeatureTracker::BpMatcher matcher(capacity.size());
            vector<double> ret = matcher.getMaxFlow(capacity, cost, 0, capacity.size() -1);
            
            complete_matches = matcher.get_forest_matchings(tree_matches);
        }
        
        // remove statistical outliers:
        vector<pair<int, vector<pair<pair<string, string>, double>>>> filtered_matches; // = complete_matches; // variable to store the filtered matching
        filtered_matches = remove_statistical_outliers(complete_matches);
        
        std::ostringstream oss;
        oss << "=========================================================================\nFT matches at time " << std::setprecision(15) << _cur_time << std::endl;
        int n_f_match = 0;

        // assign id, track count and evaluate velocity based on filtered matchings
        for(size_t i = 0; i < cur_forest.size(); ++i)
        {
            if(filtered_matches[i].first != -1)
            {
                for(const auto& match : filtered_matches[i].second)
                {
                    const string& prev_node = match.first.second;
                    const string& cur_node  = match.first.first;

                    auto prev_it = std::find_if(prev_forest[filtered_matches[i].first].begin(), prev_forest[filtered_matches[i].first].end(),
                        [&prev_node](const Ex_TreeNode& n){ return n.ex_id == prev_node; });
                    auto cur_it  = std::find_if(cur_forest[i].begin(), cur_forest[i].end(),
                        [&cur_node](const Ex_TreeNode& n){ return n.ex_id == cur_node; });

                    if (cur_it != cur_forest[i].end() && prev_it != prev_forest[filtered_matches[i].first].end())
                    {
                        oss << "    prev node " << prev_it->ex_id << " cur node " << cur_it->ex_id << " new id " << prev_it->id << std::endl;
                        ++n_f_match;
                        cur_it->id        = prev_it->id;
                        cur_it->track_cnt = prev_it->track_cnt + 1;
                        matches_names.push_back(prev_it->id);
                        cur_it->v_x = (cur_it->x - prev_it->x) / (_cur_time - _prev_time);
                        cur_it->v_y = (cur_it->y - prev_it->y) / (_cur_time - _prev_time);
                        cur_it->v_z = (cur_it->z - prev_it->z) / (_cur_time - _prev_time);
                    }
                }
            }
        }

        // count total matches before outlier removal
        int n_match = 0;
        for (const auto& cm : complete_matches)
            if (cm.first != -1)
                n_match += static_cast<int>(cm.second.size());

        oss << "Total: " << n_match << " filtered " << n_f_match << std::endl;
        logMessage(oss.str());

        //visualize the results
        // evaluate the transformation matrix from left camera (where the point are represented) and the tree camera
        // starting from the transformation matrix between tree and lcam divide it in rotation matrix and translation vector
        Eigen::Matrix3d R_lcam_tree = T_lcam_tree.block<3, 3>(0, 0);
        Eigen::Vector3d P_lcam_tree = T_lcam_tree.block<3, 1>(0, 3);
        
        // evaluate the inverce tranformation matrix
        Eigen::Matrix3d R_tree_lcam = R_lcam_tree.transpose();
        Eigen::Vector3d P_tree_lcam = - R_tree_lcam * P_lcam_tree;
            
        // assemble the final matrix
        Eigen::MatrixXd T_prov = Eigen::MatrixXd(3, 4);
        T_prov << R_tree_lcam, P_tree_lcam;
        Eigen::RowVector4d rowVec(0, 0, 0, 1); 
            
        Eigen::Matrix4d T_tree_lcam;
        T_tree_lcam << T_prov, rowVec;

        // create a vector of colors, one per tree, to use for the matched trees
        vector<cv::Scalar> match_circle_colors;
        vector<cv::Scalar> match_line_colors;
        for(size_t i = 0; i < prev_forest.size(); ++i)
        {   
            // get circle and line color
            cv::Scalar circle_color = genRandomColor();
            cv::Scalar line_color = genRandomColor();

            // save them in the relative vector
            match_circle_colors.push_back(circle_color);
            match_line_colors.push_back(line_color);
        }
        
        // draw previous forest in the images
        cv::Mat match_img_ = drawForest(prev_forest, T_tree_lcam, match_circle_colors, match_line_colors);
        
        // project a 3-D point (in left-cam frame) to a cv::Point in the tree-camera image
        auto project = [&](double x, double y, double z) -> cv::Point {
            Eigen::Vector3d p3d = (T_tree_lcam * Eigen::Vector4d(x, y, z, 1)).head<3>();
            Eigen::Vector3d uvs = K_mat * p3d;
            return cv::Point(static_cast<int>(uvs[0] / uvs[2]), static_cast<int>(uvs[1] / uvs[2]));
        };

        // draw arrows connecting matched nodes
        for(size_t i = 0; i < cur_forest.size(); ++i)
        {
            if(filtered_matches[i].first != -1)
            {
                for(const auto& match : filtered_matches[i].second)
                {
                    const string& prev_node = match.first.second;
                    const string& cur_node  = match.first.first;

                    auto prev_it = std::find_if(prev_forest[filtered_matches[i].first].begin(), prev_forest[filtered_matches[i].first].end(),
                        [&prev_node](const Ex_TreeNode& n){ return n.ex_id == prev_node; });
                    auto cur_it  = std::find_if(cur_forest[i].begin(), cur_forest[i].end(),
                        [&cur_node](const Ex_TreeNode& n){ return n.ex_id == cur_node; });

                    if (cur_it != cur_forest[i].end() && prev_it != prev_forest[filtered_matches[i].first].end())
                        cv::arrowedLine(match_img_,
                            project(prev_it->x, prev_it->y, prev_it->z),
                            project(cur_it->x,  cur_it->y,  cur_it->z),
                            cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
                }
            }
        }

        match_img = std::make_tuple(match_img_, ref_frame["tree_camera"], _cur_time);
    }
    
    // get number of nodes in current forest
    int n_nodes_cur_forest = 0;
    for(const auto& tree : cur_forest)
    {
        n_nodes_cur_forest += tree.size();
    }
   
    vector<vector<Ex_TreeNode>> selected_forest;
    if((MAX_T_CNT - n_nodes_cur_forest) > 0) // if you have space to save all the forest
    {   
        selected_forest = cur_forest; // add all the forest
    }
    else
    {   
        for(auto& tree : cur_forest) // for every tree
        {   
            // evaluate the number of nodes we can add from this tree (weighted on the number of nodes of the tree)
            int max_n_node = std::floor((tree.size() * MAX_T_CNT) / n_nodes_cur_forest);
            
            // sort the nodes based on their track_cnt (note matched nodes will have an higher one and like that we privilege the long tracked nodes in saving them for next iteration)
            // get a vector of pairs with track counter, indeces
            vector<pair<int, string>> ordered_nodes; //vector of pairs to be sorted to get the order
            std::unordered_map<std::string, Ex_TreeNode*> node_map; // node map for fast access
            
            for (auto& node : tree)
            {   
                ordered_nodes.emplace_back(node.track_cnt, node.ex_id);
                node_map[node.ex_id] = &node;
            }
           
            // sort it
            std::sort(ordered_nodes.begin(), ordered_nodes.end(), [](const auto &a, const auto &b) {
                return a.first > b.first;  // Sort by first element (descending)
            });

            int i = 0; // index where to start getting nodes from the list
            vector<Ex_TreeNode> selected_tree; // variable to keep the nodes saved 
            unordered_set<string> selected_nodes; // variable to check which node we already added to selected_tree

            while(i < ordered_nodes.size())
            {   
                string n_i = ordered_nodes[i].second; // get the node (name) in the ordered list starting from the first one
                vector<Ex_TreeNode> tmp_tree; // variable to save the path untill root
                
                // find path to root
                while(true) 
                {   
                    if(!selected_nodes.count(n_i)) // if the node hasn't already been selected in the previous iteration
                    {   
                        tmp_tree.push_back(*node_map[n_i]); // save node to temporary tree
                    }
                    else // if we already saved the feature in the selected nodes it meaans we already saved also the path from that node to the root, sso it doesn't make sense to continue the search
                    {
                        break;
                    }
                    
                    if((node_map[n_i]->ex_parent != "root") && (node_map[n_i]->ex_parent != "m_c")) // if i have a parent
                    {   
                        n_i = node_map[n_i]->ex_parent; // move to node parent in next iteration
                    }
                    else // else
                    {   
                        break; // break loop
                    }
                }
                
                if(((selected_tree.size() + tmp_tree.size()) <= max_n_node) && tmp_tree.size() > 0) // if by adding the temporary nodes (i.e. nodes from the current n_i node to the root) in the selected ones i don't surpass the maximum number of nodes i can add
                {   
                    selected_tree.insert(selected_tree.end(), tmp_tree.begin(), tmp_tree.end()); // add them
                    for(const auto& tmp_n : tmp_tree)
                    {
                        selected_nodes.insert(tmp_n.ex_id); // add their ids in the selected_nodes set
                    }
                }
                else if(selected_tree.size() == max_n_node) // if you filled selected tree with the maximum number of nodes, doesn't make sense to continue to search
                {
                    break;
                }
                else
                {   
                    i += 1; // increase i to get the next node in priority order
                    continue;
                }
                
                i += 1; // increase i to get the next node in priority order
            }   
            
            // if i still have some space (aka i didn't get able to add all the path from a matched node to the root because it was too big) add the remaining nodes from the sons of the selected nodes
            if(max_n_node > (selected_tree.size()))
            {   
                std::vector<std::string> selected_nodes_vec(selected_nodes.begin(), selected_nodes.end());
                for (size_t i = 0; i < selected_nodes_vec.size(); ++i) // for all the selected nodes
                {   
                    const auto& s_n = selected_nodes_vec[i];
                    for(const auto& son_s_n : node_map[s_n]->ex_sons) // for all the sons of the selected node
                    {
                        if(!selected_nodes.count(son_s_n) && ((max_n_node - (selected_tree.size() + 1)) > 0) && son_s_n != "tip") // if i din't select the node yet and i still have space to add a node
                        {   
                            selected_tree.push_back(*node_map[son_s_n]); // add the node to the selected tree
                            selected_nodes.insert(son_s_n); // add it to the list of selected nodes
                            selected_nodes_vec.push_back(son_s_n);
                        }
                        else
                        {   
                            break;
                        }
                    }
                }
            }
            
            // verify that all the relations sons, parent are coherent with the selected nodes: during the selection process we may not select some nodes that will remain in the parent sons attribute of their neighbors, here we remove them
            for(auto& node : selected_tree){ // for all the selected nodes
                if((!selected_nodes.count(node.ex_parent)) && (node.ex_parent != "root") && (node.ex_parent != "m_c")){ // if the parent is not among the selected nodes
                    node.ex_parent = "root";
                    ROS_ERROR("Featuretracking matching something wenty wrong");
                }
                
                // Remove sons not in selected_nodes
                node.ex_sons.erase(
                    std::remove_if(node.ex_sons.begin(), node.ex_sons.end(),
                        [&](const std::string& son) {
                            return !selected_nodes.count(son);
                        }
                    ),
                    node.ex_sons.end()
                );

                // If no sons left, add "tip"
                if (node.ex_sons.empty()) {
                    node.ex_sons.push_back("tip");
                }
            }

            selected_forest.push_back(selected_tree); // add the selected tree to the selected forest
        }
    }

    // assign a unique id to the nodes
    map<string, int> lu_table; // create a lookup table that associate nodes ex_id and new id
    for(auto& s_tree : selected_forest) // foe every selected tree
    {
        for(auto& s_n : s_tree) // for every selected node 
        {
            if(s_n.id < 0) // if we didn't assign an id yet (i.e. we didn't match the node yet)
            {
                s_n.id = new_ids; // assign new id
                lu_table[s_n.ex_id] = new_ids; /// save the new id in the lookup table
                new_ids += 1; // increase the new_ids variable
            }
            else // if we aalready assigned an id (i.e. we matched the node)
            {
                lu_table[s_n.ex_id] = s_n.id; // save the id in the lookup table
            }
        }
    }

    // save variable for next iteration
    prev_forest = selected_forest; 
    _prev_time = _cur_time;
    has_tree_Prediction = false;

    // convert forest in a single vector of nodes (ATTENTION for now i will do like that since the rest of the node reason like that, but in a future it may be convinient to modify the rest of the code to operate with the division in trees)
    vector<TreeNode> out_forest;
    int n_m_deb = 0;
    for(const auto& ex_tree : selected_forest)
    {   
        for(const auto& ex_node : ex_tree)
        {
            TreeNode node;
            // assign id
            node.id = ex_node.id;

            if (std::find(matches_names.begin(), matches_names.end(), ex_node.id) != matches_names.end()) {
                n_m_deb += 1;
            } 

            // assign position
            node.x = ex_node.x;
            node.y = ex_node.y;
            node.z = ex_node.z;

            if(ICP_P2L) // if we are gonna use a point to line cost function
            {
                // evaluate unit vector from node to its parent
                // find parent
                if((ex_node.ex_parent == "root") || (ex_node.ex_parent == "m_c"))// if its the root node
                {
                    // assign z vector pointing downward
                    node.n_x = -base_link_z.x();
                    node.n_y = -base_link_z.y();
                    node.n_z = -base_link_z.z();
                }
                else
                {   
                    string parent_id = ex_node.ex_parent;
                    auto it = std::find_if(ex_tree.begin(), ex_tree.end(),
                        [parent_id](const Ex_TreeNode& node) {
                            return node.ex_id == parent_id;
                        });

                    if (it != ex_tree.end()) // if found
                    {
                        // get current node position
                        Eigen::Vector3d p(ex_node.x, ex_node.y, ex_node.z);

                        // get parent node position
                        Eigen::Vector3d p_parent(it->x, it->y, it->z);
                        
                        // evaluate unit vector from current node to its parent
                        Eigen::Vector3d v = (p_parent - p) / (p_parent - p).norm();

                        // assign it
                        node.n_x = v.x();
                        node.n_y = v.y();
                        node.n_z = v.z();
                    }
                    else 
                    {
                        ROS_WARN("didn't found parent node for ICP normal evaluation");
                    }
                }
            }

            // assign velocity
            node.v_x = ex_node.v_x;
            node.v_y = ex_node.v_y;
            node.v_z = ex_node.v_z;
            

            // assign track counter
            node.track_cnt = ex_node.track_cnt;

            // add it to the out forest
            out_forest.push_back(node);
        }
    }
    
    pair<double, vector<TreeNode>> t_featureFrame = make_pair(_cur_time, out_forest);
    return t_featureFrame;

}


void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
    {
        stereo_cam = 1;
    }
}

void FeatureTracker::setIntrinsicParameter_topic(const sensor_msgs::msg::CameraInfo &camera_info, const string camera_name)
{   
    camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromTopic(camera_info, camera_name);
    
    m_camera.push_back(camera);

    if (camera_name == "right_camera")
    {
        stereo_cam = 1;
    }
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++) // for all the passed points
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y); // get its x, y position
        Eigen::Vector3d b;
        cam->liftProjective(a, b); // project the 3D point to the image plane
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++) // for all the ID's
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i])); // add the current point ID together with the point coordinate
    }

    // caculate points velocity
    if (!prev_id_pts.empty()) // if you still have points from the previous iteration
    {
        double dt = cur_time - prev_time; // evaluate the delta time
        
        for (unsigned int i = 0; i < pts.size(); i++) // for all the points
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]); // find the point by its ID
            if (it != prev_id_pts.end()) // if you've found it
            {
                double v_x = (pts[i].x - it->second.x) / dt; // evaluate its velocity in the image x
                double v_y = (pts[i].y - it->second.y) / dt; // evaluate its velocity in the image y
                pts_velocity.push_back(cv::Point2f(v_x, v_y)); // add the velocity to the variable 
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0)); // otherwise if it's the first time you see the point init its velocity to 0

        }
    }
    else // if you don't have previous points
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++) // for all the points
        {
            pts_velocity.push_back(cv::Point2f(0, 0)); // init their velocity to 0
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true; // set the has prediction flag
    predict_pts.clear(); // clear previous points
    predict_pts_debug.clear(); // clear previous points
    map<int, Eigen::Vector3d>::iterator itPredict; // init variable
    for (size_t i = 0; i < ids.size(); i++) // for all the indices
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i]; // get the id
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) // if the id is in the predicted points
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv); // project the 3d point in the image plane
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y())); // add the prediction to the vector
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y())); // add the prediction to teh vector
        }
        else // otherwise add the previous point ?
            predict_pts.push_back(prev_pts[i]);
    }
}

void FeatureTracker::set_tree_Prediction(map<int, Eigen::Vector3d> &predict_t_Pts)
{
    has_tree_Prediction = true; // Set the has-prediction flag
    predict_forest_pts.clear(); // Clear previous points
    predict_forest_pts_debug.clear(); // Clear previous points
    map<int, Eigen::Vector3d>::iterator itPredict; // init variable
    for (size_t i = 0; i < prev_forest.size(); ++i) // for every tree in previous forest
    {   
        vector<Ex_TreeNode> predict_tree_pts, predict_tree_pts_debug; // create the tree variable to store the predicted and non points
        for(size_t j = 0; j < prev_forest[i].size(); ++j) // for every node in the tree
        {
            int id = prev_forest[i][j].id; // Get the id of the current node
        
            // Find the corresponding node in predict_t_Pts
            itPredict = predict_t_Pts.find(id);

            if (itPredict != predict_t_Pts.end()) // If the id is found in the predicted points
            {   
                Ex_TreeNode predict_t_node = prev_forest[i][j]; // get the tree node
                // update node position with the predicted one
                predict_t_node.x = itPredict->second.x();
                predict_t_node.y = itPredict->second.y();
                predict_t_node.z = itPredict->second.z();

                predict_tree_pts.push_back(predict_t_node); // Add the prediction to the vector
                predict_tree_pts_debug.push_back(predict_t_node); // Add the prediction to the debug vector
            }
            else // Otherwise, add the previous point as it is
            {
                predict_tree_pts.push_back(prev_forest[i][j]);
            }
        }
        predict_forest_pts.push_back(predict_tree_pts); // append the tree to the forest
        predict_forest_pts_debug.push_back(predict_tree_pts_debug); // append the tree to the forest
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds, set<int> &remove_t_PtsIds)
{
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nFT ro removing outliers from prev features\nnormal features: " << std::endl;

    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end()){
            status.push_back(0);
            oss << "    remove " << ids[i] << std::endl;
        }
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);

    if (USE_TREE)
    {   
        oss << "tree features:" << std::endl;
        for (vector<Ex_TreeNode>& prev_tree : prev_forest)
        {
            // prev_tree.erase(std::remove_if(prev_tree.begin(), prev_tree.end(),
            //                        [&remove_t_PtsIds](const Ex_TreeNode &node) {
            //                            return remove_t_PtsIds.find(node.id) != remove_t_PtsIds.end(); // O(log N) lookup
            //                        }),
            //         prev_tree.end());
            // only for logging purposes
            auto it = std::remove_if(prev_tree.begin(), prev_tree.end(),
                         [&remove_t_PtsIds](const Ex_TreeNode &node) {
                             return remove_t_PtsIds.find(node.id) != remove_t_PtsIds.end();
                         });

            oss << "    remove ex id " << it->ex_id << " id " << it->id << std::endl;
            prev_tree.erase(it, prev_tree.end());


        }
    }

}


cv::Mat FeatureTracker::getTrackImage()
{
   
    return imTrack;
}

std::tuple<cv::Mat, sensor_msgs::msg::CameraInfo, double> FeatureTracker::getTreeMatch()
{
    return match_img;
}