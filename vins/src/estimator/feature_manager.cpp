/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

int TreePerId::endFrame()
{
    return tree_per_frame.back().frame;
}

bool TreePerId::has_frame(int frame) const // NEW
{
    // check if the feature has an observation at the frame
    for(const auto& obs : tree_per_frame)
    {
        if(obs.frame == frame)
        {
            return true;
        }
    }
    return false;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
    t_feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    // no need to count the tree features

    return cnt;
}

int FeatureManager::get_tree_FeatureCount()
{
    int cnt = 0;
    for (auto &it : t_feature)
    {
        it.used_num = it.tree_per_frame.size();
        cnt++;
    }
    
    return cnt;
}

void FeatureManager::logMessage(const std::string& message) 
{
    const std::string LOG_FILE_PATH = "/home/glugano/Desktop/log.txt";

    std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
    if (!logFile) {
        std::cerr << "Error: Unable to open log file." << std::endl;
        return;
    }

    logFile << message << std::endl;
}

bool FeatureManager::addFeatureTreeCheckParallax(int frame_count, const double header, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const pair<double, vector<TreeNode>> &tree, double td)
{
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nFM adding features to buffer at frame count " << frame_count << "\nnormal features: " << std::endl;

    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0; // number of feature occured in this frame and last frame
    double parallax_tree_sum = 0;
    int parallax_tree_num = 0; // number of feature occured in this frame and last frame
    last_track_num = 0; // number of features that has been already found in the previous frames
    last_average_parallax = 0;
    new_feature_num = 0; // number of new feature encountered  with the current frame
    long_track_num = 0; // number of features that has been already encountered in the previous frames, for at least 4 frames
    for (auto &id_pts : image) // for every point in the image
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // add the point (with x, y, z, p_u, p_v, v_x, v_y) in the f_per_frame variable together with the time offset td
        assert(id_pts.second[0].first == 0); // meaning the camera id is 0 (first camera)
        if(id_pts.second.size() == 2) // meaning that there are two camera ids and so stereocamera case
        {
            f_per_fra.rightObservation(id_pts.second[1].second); // add the points (obtained from the second camera) in the f_per_frame variable with the time offset td
            assert(id_pts.second[1].first == 1); // meaning the camera id is 1 (second camera)
        }

        int feature_id = id_pts.first; // get the feature_id (same as the one received from the message)
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) 
                          {
            return it.feature_id == feature_id;
                          }); // search in the feature vector, form the beginning to the end if the feature id is present, it return a pointer to the element in the feature vector that has the same feature_id, or if no such element exists it return feature.end()

        if (it == feature.end()) // if it's the first time we encounter a feature with that feature_id
        {   
            oss << "    new feature " << feature_id << " add feature " << std::endl;
            feature.push_back(FeaturePerId(feature_id, frame_count)); // add the feature_id in the feature vector, with the associated frame_count (counting the first frame in which we saw this feature)
            feature.back().feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y)
            new_feature_num++; // increase the number of new feature encountered  with the current frame
        }
        else if (it->feature_id == feature_id) // if the feature was already in the feature vector
        {
            oss << "    old feature " << feature_id << " add observation" << std::endl;
            it->feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y) obtained in the current frame
            last_track_num++; // increase the number of features that has been already found in the previous frames
            if( it-> feature_per_frame.size() >= 4) // if the feature has been encounter in at least 4 previous frames
                long_track_num++; // increase the number of features that has been already encountered in the previous frames, for at least 4 frames
        }
    }

    int last_t_track_num = 0; // number of features that has been already found in the previous frames
    int new_t_feature_num = 0; // number of new feature encountered  with the current frame
    int long_t_track_num = 0; // number of features that has been already encountered in the previous frames, for at least 4 frames
    oss << "tree features: " << std::endl;
    for (auto &id_t_pts : tree.second)
    {   
        //std::cout << id_t_pts.id << " ";
        TreePerFrame t_per_fra(id_t_pts, td, frame_count); // add the tree point (with x, y, z, v_x, v_y, v_z, fd, parent, sons) in the t_per_frame variable together with the time offset td
        int feature_id = id_t_pts.id; // get the feature_id
        auto it = find_if(t_feature.begin(), t_feature.end(), [feature_id](const TreePerId &it) 
                      {
        return it.feature_id == feature_id;
                  }); // search in the t_feature vector, form the beginning to the end if the feature id is present, it return a pointer to the element in the t_feature vector that has the same feature_id, or if no such element exists it return t_feature.end()
        if (it == t_feature.end()) // if it's the first time we encounter a tree feature with that feature_id
        {
            oss << "    new feature " << feature_id << " add feature " << std::endl;
            t_feature.push_back(TreePerId(feature_id, frame_count)); // add the feature_id in the feature vector, with the associated frame_count (counting the first frame in which we saw this feature)
            t_feature.back().tree_per_frame.push_back(t_per_fra); // in this tree feature, under the tree_per_frame field add the point data (x, y, z, v_x, v_y, v_z, parents, sons)
            new_t_feature_num++; // increase the number of new feature encountered  with the current frame
        }
        else if (it->feature_id == feature_id) // if the tree feature was already in the feature vector
        {   
            oss << "    old feature " << feature_id << " add observation" << std::endl;
            it->tree_per_frame.push_back(t_per_fra); // in this tree feature, under the tree_per_frame field add the point data (x, y, z, v_x, v_y, v_z, parents, sons) obtained in the current frame
            last_t_track_num++; // increase the number of features that has been already found in the previous frames
            if( it-> tree_per_frame.size() >= 3) // if the feature has been encounter in at least 4 previous frames
                long_t_track_num++; // increase the number of features that has been already encountered in the previous frames, for at least 4 frames
        }
    }
    oss << "Total: \nnormal features: new features " << new_feature_num << " old features " << last_track_num << "\ntree features: new features " << new_t_feature_num << " old features " << last_t_track_num << std::endl;
    logMessage(oss.str());
    ///// LOG /////

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM accumulator timer at frame count " << frame_count << std::endl;

    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    //if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num || last_t_track_num < 20 || long_t_track_num < 5) // if we are at a frame smaller then the third, or with the current frame we encountered less then 20 features already encountered in teh past, of with the current frame we encountered less then 40 features already encountered in the past for at least 4 times, if we encountered new feature twice the number of the already encountered features return TRUE (keyframe)
    // if (frame_count < 2 || last_track_num < 15 || long_track_num < 30 || new_feature_num > 0.5 * last_track_num || last_t_track_num < 15 || long_t_track_num < 5) // if we are at a frame smaller then the third, or with the current frame we encountered less then 20 features already encountered in teh past, of with the current frame we encountered less then 40 features already encountered in the past for at least 4 times, if we encountered new feature twice the number of the already encountered features return TRUE (keyframe)
    // {
    //     oss1 << "Keyframe: for number of new features" << std::endl;
    //     oss1 << "frame count " << frame_count << " last track num " << last_track_num << " long track num " << long_track_num << " new feature num " << new_feature_num << " last t track num " << last_t_track_num << " long t track num " << long_t_track_num  << std::endl;
    //     logMessage(oss1.str());
    //     ///// LOG /////
    //     return true;
    // }

    for (auto &it_per_id : feature) // for all the feature already saved, together with the newly discovered with this frame
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // if the feature first appearence is farthest then 2 frames from the current one and [the sum of the first appearence of the feature (in frames) plus the number of time we ennocuntered it is bigger then the actual frame count] = if the feature appeared the first time at least in the previous frame and it also appeared in the current frame
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // get the distance of the feature in the image between the last two frames in whic it appeared
            parallax_num++; // increase the number of feature occured in this frame and last frame
        }
    }
    for (auto &it_t_per_id : t_feature) // for all the feature already saved, together with the newly discovered with this frame
    {
        if (it_t_per_id.has_frame(frame_count - 2) && it_t_per_id.has_frame(frame_count - 1)) // if the feature first appearence is farthest then 2 frames from the current one and [the sum of the first appearence of the feature (in frames) plus the number of time we ennocuntered it is bigger then the actual frame count] = if the feature appeared the first time at least in the previous frame and it also appeared in the current frame
        {
            parallax_tree_sum += compensated_tree_Parallax2(it_t_per_id, frame_count); // get the distance of the feature in the image between the last two frames in whic it appeared
            parallax_tree_num++; // increase the number of feature occured in this frame and last frame
        }
    }
    //if (frame_count < 2 || last_track_num < 50 || long_track_num < 15  || last_t_track_num < 20 || long_t_track_num < 5) // if we are at a frame smaller then the third, or with the current frame we encountered less then 20 features already encountered in teh past, of with the current frame we encountered less then 40 features already encountered in the past for at least 4 times, if we encountered new feature twice the number of the already encountered features return TRUE (keyframe)
    //if(frame_count < 2 || last_track_num < 20 || last_t_track_num < 10 || (last_track_num + last_t_track_num) < 50)
    
    bool fast_flag = false;
    bool var_flag = false;
    // low pass signals for debugging
    double alpha_1 = 0.9;
    filtered_last_track_num = alpha_1 * last_track_num + (1 - alpha_1) * filtered_last_track_num;
    filtered_last_t_track_num = alpha_1 * last_t_track_num + (1 - alpha_1) * filtered_last_t_track_num;
    filtered_long_track_num = alpha_1 * long_track_num + (1 - alpha_1) * filtered_long_track_num;
    filtered_long_t_track_num = alpha_1 * long_t_track_num + (1 - alpha_1) * filtered_long_t_track_num;
    filtered_new_feature_num = alpha_1 * new_feature_num + (1 - alpha_1) * filtered_new_feature_num;
    filtered_new_t_feature_num = alpha_1 * new_t_feature_num + (1 - alpha_1) * filtered_new_t_feature_num;

    // low pass filter the macthing signals and increment accumulative timer:
    // raw observed value
    double x_k = long_track_num + long_t_track_num;

    oss1 << "x_k " << x_k << " \nX effective max 0 " << X_effective_max << std::endl;
    
    // update effective max
    if (x_k > X_effective_max)
    {
        X_effective_max = x_k;              // immediate increase
    }
    else
    {
        X_effective_max -= gamma;           // very slow decay
    }

    // clamp maximum theoretical value and efective one
    X_effective_max = std::min(X_effective_max, static_cast<double>(MAX_CNT + MAX_T_CNT));

    // evaluate error term between the average and the current sample
    double err = x_k - last_track_num_plateau;

    // update variance threshold
    var_threshold = vt_K * std::sqrt(noise_var_estimate);
    oss1 << "X effective max " << X_effective_max << "\nerr " << err << " \nvar_threshold " << var_threshold << " \nvar_threshold_max " << var_threshold_max << std::endl;
    if(var_threshold > var_threshold_max)
    {
        var_threshold_max = var_threshold;
        oss1 << "    updated var_threshold_max to " << var_threshold_max << std::endl;
    }

    // update plateau estimate
    if(std::abs(err) > var_threshold)
    {  
        fast_flag = true;
        oss1 << "    fast updated last_track_num_plateau from " << last_track_num_plateau;
        last_track_num_plateau += alpha_fast * err;
        oss1 << " to " << last_track_num_plateau << std::endl;
    }
    else
    {
        oss1 << "    slow updated last_track_num_plateau from " << last_track_num_plateau;
        last_track_num_plateau += alpha_slow * err;
        oss1 << " to " << last_track_num_plateau << std::endl;
        var_flag = true;
        oss1 << "    slow updated noise_var_estimate from " << noise_var_estimate;
        noise_var_estimate = (1 - betha) * noise_var_estimate + betha * (err * err); // update variance
        oss1 << " to " << noise_var_estimate << std::endl;
    }

    // evaluate distance plateau max number of tracked features
    double D_max = std::max(0.0, (X_effective_max - last_track_num_plateau));

    // evaluate timer accumulator rate
    double c_val = (accumulator_timer_thresh / delta_time_0) * (header - prev_time);
    prev_time = header;
    double rd_k = ((accumulator_timer_thresh - c_val) / std::max(1.0, (X_effective_max - min_track_num - var_threshold_max))) + 1.2;
    double r_D_max = rd_k * std::max(0.0, D_max - var_threshold);

    // update accumulator timer
    accumulator_timer = std::max(0.0, accumulator_timer + r_D_max + c_val);
    oss1 << "D_max " << D_max << "\nrd_k " << rd_k << "\nr_D_max " << r_D_max << "\nacc_timer " << accumulator_timer << "\nc_val " << c_val << std::endl;
    // evaluate distance between last and long matching  (works great but i think needs some more works on the optimization size too, the weights)
    // double last_long = 0;
    // if(last_track_num)
    // {
    //     last_long = static_cast<double>(filtered_last_track_num - filtered_long_track_num) / static_cast<double>(filtered_last_track_num);
    // }
    // double last_long_tree = 0;
    // if(last_t_track_num)
    // {
    //     last_long_tree = static_cast<double>(filtered_last_t_track_num - filtered_long_t_track_num) / static_cast<double>(filtered_last_t_track_num);
    // }
    // double cumulative_last_long = 0;
    // if(last_track_num && last_t_track_num)
    // {
    //     cumulative_last_long = static_cast<double>((filtered_last_track_num + filtered_last_t_track_num) - (filtered_long_track_num + filtered_long_t_track_num)) / static_cast<double>(filtered_long_track_num + filtered_last_t_track_num);
    // }
    oss1 << "=========================================================================\nFM keyframe selection at frame count " << frame_count << std::endl;
    if((frame_count < 5)  || (accumulator_timer > accumulator_timer_thresh))
    {
        oss1 << "Keyframe: for number of new features" << std::endl;
        oss1 << "frame count " << frame_count << " last track num " << last_track_num << " long track num " << long_track_num << " new feature num " << filtered_new_feature_num << " last t track num " << last_t_track_num << " long t track num " << long_t_track_num << " new t feature num " << filtered_new_t_feature_num << " filtered last track num " << filtered_last_track_num << " filtered long track num " << filtered_long_track_num << " filtered last t track num " << filtered_last_t_track_num << " filtered long t track num " << filtered_long_t_track_num << " last_track_num_plateau " << last_track_num_plateau << " accumulator_timer " << accumulator_timer << " var_threshold " << var_threshold << " fast_flag " << fast_flag << " var_flag " << var_flag << std::endl;
        
        double par = 0;
        if(parallax_num != 0)
            par = parallax_sum / parallax_num;
        double tree_par = 0;
        if(parallax_tree_num != 0)
            tree_par = parallax_tree_sum / parallax_tree_num;
        oss1 << "parallax " << par << " tree parallax " << tree_par << std::endl;
        logMessage(oss1.str());
        ///// LOG /////
        accumulator_timer = 0;
        
        return true;
    }
    
    if (parallax_num + parallax_tree_num == 0 ) // if none of the features appeared in this frame and last frame
    {
        oss1 << "Keyframe: for no features in  parallax evaluation" << std::endl;
        logMessage(oss1.str());
        ///// LOG /////
        accumulator_timer = 0;
        return true; // return true (keyframe)
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;

        if((FOCAL_LENGTH * (parallax_sum / parallax_num)) + (parallax_tree_sum / parallax_tree_num) >= 300)
        {
            oss1 << "Keyframe: for parallax" << std::endl;
            oss1 << "frame count " << frame_count << " last track num " << last_track_num << " long track num " << long_track_num << " new feature num " << filtered_new_feature_num << " last t track num " << last_t_track_num << " long t track num " << long_t_track_num << " new t feature num " << filtered_new_t_feature_num << " filtered last track num " << filtered_last_track_num << " filtered long track num " << filtered_long_track_num << " filtered last t track num " << filtered_last_t_track_num << " filtered long t track num " << filtered_long_t_track_num << " last_track_num_plateau " << last_track_num_plateau << " accumulator_timer " << accumulator_timer << " var_threshold " << var_threshold << " fast_flag " << fast_flag << " var_flag " << var_flag << std::endl;
            double par = 0;
            if(parallax_num != 0)
                par = parallax_sum / parallax_num;
            double tree_par = 0;
            if(parallax_tree_num != 0)
                tree_par = parallax_tree_sum / parallax_tree_num;
            oss1 << "parallax " << par << " tree parallax " << tree_par << std::endl;
            logMessage(oss1.str());
            ///// LOG /////
            accumulator_timer = 0;
        }
        else
        {
            oss1 << "No Keyframe" << std::endl;
            oss1 << "frame count " << frame_count << " last track num " << last_track_num << " long track num " << long_track_num << " new feature num " << filtered_new_feature_num << " last t track num " << last_t_track_num << " long t track num " << long_t_track_num << " new t feature num " << filtered_new_t_feature_num << " filtered last track num " << filtered_last_track_num << " filtered long track num " << filtered_long_track_num << " filtered last t track num " << filtered_last_t_track_num << " filtered long t track num " << filtered_long_t_track_num << " last_track_num_plateau " << last_track_num_plateau << " accumulator_timer " << accumulator_timer << " var_threshold " << var_threshold << " fast_flag " << fast_flag << " var_flag " << var_flag << std::endl;
            double par = 0;
            if(parallax_num != 0)
                par = parallax_sum / parallax_num;
            double tree_par = 0;
            if(parallax_tree_num != 0)
                tree_par = parallax_tree_sum / parallax_tree_num;
            oss1 << "parallax " << par << " tree parallax " << tree_par << std::endl;
            logMessage(oss1.str());
            ///// LOG /////
        }
            
        return ((FOCAL_LENGTH * (parallax_sum / parallax_num)) + (parallax_tree_sum / parallax_tree_num) >= 300); // return true if the average parallax is bigger then a threshold
    }
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0; // number of feature occured in this frame and last frame
    last_track_num = 0; // number of features that has been already found in the previous frames
    last_average_parallax = 0;
    new_feature_num = 0; // number of new feature encountered  with the current frame
    long_track_num = 0; // number of features that has been already encountered in the previous frames, for at least 4 frames
    for (auto &id_pts : image) // for every point in the image
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // add the point (with x, y, z, p_u, p_v, v_x, v_y) in the f_per_frame variable together with the time offset td
        assert(id_pts.second[0].first == 0); // meaning the camera id is 0 (first camera)
        if(id_pts.second.size() == 2) // meaning that there are two camera ids and so stereocamera case
        {
            f_per_fra.rightObservation(id_pts.second[1].second); // add the points (obtained from the second camera) in the f_per_frame variable with the time offset td
            assert(id_pts.second[1].first == 1); // meaning the camera id is 1 (second camera)
        }

        int feature_id = id_pts.first; // get the feature_id (same as the one received from the message)
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) 
                          {
            return it.feature_id == feature_id;
                          }); // search in the feature vector, form the beginning to the end if the feature id is present, it return a pointer to the element in the feature vector that has the same feature_id, or if no such element exists it return feature.end()

        if (it == feature.end()) // if it's the first time we encounter a feature with that feature_id
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // add the feature_id in the feature vector, with the associated frame_count (counting the first frame in which we saw this feature)
            feature.back().feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y)
            new_feature_num++; // increase the number of new feature encountered  with the current frame
        }
        else if (it->feature_id == feature_id) // if the feature was already in the feature vector
        {
            it->feature_per_frame.push_back(f_per_fra); // in this feature, under the feature_per_frame field add the point data (x, y, z, p_u, p_v, v_x, v_y) obtained in the current frame
            last_track_num++; // increase the number of features that has been already found in the previous frames
            if( it-> feature_per_frame.size() >= 4) // if the feature has been encounter in at least 4 previous frames
                long_track_num++; // increase the number of features that has been already encountered in the previous frames, for at least 4 frames
        }
    }

    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num) // if we are at a frame smaller then the third, or with the current frame we encountered less then 20 features already encountered in teh past, of with the current frame we encountered less then 40 features already encountered in the past for at least 4 times, if we encountered new feature twice the number of the already encountered features return TRUE (keyframe)
        return true;

    for (auto &it_per_id : feature) // for all the feature already saved, together with the newly discovered with this frame
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // if the feature first appearence is farthest then 2 frames from the current one and [the sum of the first appearence of the feature (in frames) plus the number of time we ennocuntered it is bigger then the actual frame count] = if the feature appeared the first time at least in the previous frame and it also appeared in the current frame
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // get the distance of the feature in the image between the last two frames in whic it appeared
            parallax_num++; // increase the number of feature occured in this frame and last frame
        }
    }

    if (parallax_num == 0) // if none of the features appeared in this frame and last frame
    {
        return true; // return true (keyframe)
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        return parallax_sum / parallax_num >= MIN_PARALLAX; // return true if the average parallax is bigger then a threshold
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature) // for all the features
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) // if the feature first appearence is before frame_count_l and the last appearence is after frame_count_r
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point; // get the undistorted feature position at the frame_count_l frame

            b = it.feature_per_frame[idx_r].point; // get the undistorted feature position at the frame_count_r frame
            
            corres.push_back(make_pair(a, b)); // add it to corres
        }
    }
    return corres; 
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::set_tree_Depth(const VectorXd &x)
{
    int t_feature_index = -1;
    for (auto &it_per_id : t_feature)
    {
        it_per_id.used_num = it_per_id.tree_per_frame.size();
        if (it_per_id.used_num < 3)
            continue;
        
        it_per_id.estimated_depth = x(++t_feature_index);
        
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove failures before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    double normal_features_removed = 0;
    double tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;
        if (it->solve_flag == 2) // if some feature has not been optimied 
        {
            feature.erase(it); // delete them
            normal_features_removed++;
        }
            
    }
    if (USE_TREE) 
    {
        for (auto it = t_feature.begin(), it_next = t_feature.begin();
            it != t_feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
        {
            it_next++;
            if (it->solve_flag == 2) // if some feature has not been optimied (actually useless)
            {
                t_feature.erase(it); // delete them
                tree_features_removed++;
            }
        }
    }
    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove failures removed\nnormal features " << normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "=========================================================================\nFM remove failures after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;

    if(USE_TREE)
    {
        for (auto &it_per_id : t_feature)
            it_per_id.estimated_depth = -1;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // init the vector long as the number of features
    int feature_index = -1;
    for (auto &it_per_id : feature) // for every feature
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // assign the used_num
        if (it_per_id.used_num < 4) // if smaller then 4 skip the feature
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // in any case assign the depth in the vector to 1/depth of the feature in the image
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

VectorXd FeatureManager::get_tree_DepthVector()
{
    VectorXd dep_vec(get_tree_FeatureCount()); // init the vector long as the number of features
    int feature_index = -1;
    for (auto &it_per_id : t_feature) // for every feature
    {
        it_per_id.used_num = it_per_id.tree_per_frame.size(); // assign the used_num
        if (it_per_id.used_num < 3) // if smaller then 4 skip the feature
            continue;
        dep_vec(++feature_index) = it_per_id.estimated_depth;
    }
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero(); // init the design matrix (projection matrix P) and fil it with the elements 
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // solve with the SVD the triangulation problem obtaining the £D position of the point in homogeneous coordinates
    point_3d(0) = triangulated_point(0) / triangulated_point(3); // evaluate the point position in euclidean space (x)
    point_3d(1) = triangulated_point(1) / triangulated_point(3); // evaluate the point position in euclidean space (y)
    point_3d(2) = triangulated_point(2) / triangulated_point(3); // evaluate the point position in euclidean space (z)
}

Eigen::MatrixXd FeatureManager::buildEpipolarConstraintMatrix( vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    size_t n = pts3D.size();

    // Each point gives 2 rows, each row has 12 columns for linear PnP
    Eigen::MatrixXd A(2*n, 12);

    for (size_t i = 0; i < n; ++i)
    {
        double X = pts3D[i].x;
        double Y = pts3D[i].y;
        double Z = pts3D[i].z;
        double u = pts2D[i].x;
        double v = pts2D[i].y;

        // Row for u-coordinate
        A(2*i, 0)  = X;
        A(2*i, 1)  = Y;
        A(2*i, 2)  = Z;
        A(2*i, 3)  = 1.0;
        A(2*i, 4)  = 0.0;
        A(2*i, 5)  = 0.0;
        A(2*i, 6)  = 0.0;
        A(2*i, 7)  = 0.0;
        A(2*i, 8)  = -u * X;
        A(2*i, 9)  = -u * Y;
        A(2*i, 10) = -u * Z;
        A(2*i, 11) = -u;

        // Row for v-coordinate
        A(2*i+1, 0)  = 0.0;
        A(2*i+1, 1)  = 0.0;
        A(2*i+1, 2)  = 0.0;
        A(2*i+1, 3)  = 0.0;
        A(2*i+1, 4)  = X;
        A(2*i+1, 5)  = Y;
        A(2*i+1, 6)  = Z;
        A(2*i+1, 7)  = 1.0;
        A(2*i+1, 8)  = -v * X;
        A(2*i+1, 9)  = -v * Y;
        A(2*i+1, 10) = -v * Z;
        A(2*i+1, 11) = -v;
    }

    return A;

}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r); // transform the rotational matrix in opencv format
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t); // transform the translational vector in opencv format
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);   // get intrinsic camera parameters
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1); // obtain transformation matrix that transform a 3D point in the object coordinate frame to the camera coordinate frame
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r); // transform the rotation matrix in rodriguez vector
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp); // transform the rodriguez vector in eigen rotational matrix
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp); // transform the translational vector in eigen translational vector

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

double FeatureManager::computeError(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
{
    double error = 0.0;
    for (size_t i = 0; i < model.size(); ++i) {
        Eigen::Vector3d transformed = R * model[i] + t;
        error += (scene[i] - transformed).squaredNorm();
    }
    return std::sqrt(error / model.size());
}

double FeatureManager::computeErrorPointToLine(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Eigen::Vector3d>& model_axes, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) 
{
    double error = 0.0;
    for (size_t i = 0; i < model.size(); ++i) {
        Eigen::Vector3d transformed = R * model[i] + t;
        Eigen::Vector3d diff = scene[i] - transformed;
        // Distance from point to line: ||diff × axis|| / ||axis||
        // Since axis is unit vector: dist = ||diff × axis||
        double dist = diff.cross(model_axes[i]).norm();
        error += dist * dist;
    }
    return std::sqrt(error / model.size());
}

std::pair<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, double> FeatureManager::estimatePoseSVD(const std::vector<Eigen::Vector3d>& model, const std::vector<Eigen::Vector3d>& scene, const std::vector<Vector3d>& model_axes, double& cond_num) 
{
    // variable to store the final result
    Eigen::Matrix3d R_final;
    Eigen::Vector3d t_final;
    double final_error = 1e10;

    if (model.size() != scene.size() || model.size() < 3) {
        ROS_WARN("Error: model and scene must have same size and >= 3 points\n");
        return std::make_pair(std::make_pair(R_final, t_final), final_error);
    }

    if(ICP_P2L) // point to line ICP
    {
        size_t n = model.size();

        // Build linear system: A * x = b
        // where x = [wx, wy, wz, tx, ty, tz] (rotation vector + translation)
        // Each point-to-line constraint gives us 3 equations (one for each component of cross product)
        Eigen::MatrixXd A(3 * n, 6);
        Eigen::VectorXd b(3 * n);

        for (size_t i = 0; i < n; ++i) {
            const Eigen::Vector3d& d = model_axes[i];              // axis direction (unit vector)
            const Eigen::Vector3d& p_model = model[i];             // point on axis
            const Eigen::Vector3d& p_scene = scene[i];             // scene point

            // Compute the constraint vector: (p_model - p_scene) × d
            // We want (transformed p_model - p_scene) × d  0
            Eigen::Vector3d constraint_rhs = (p_model - p_scene).cross(d);

            // For each component of the cross product (3 equations per correspondence)
            for (int j = 0; j < 3; ++j) {
                int row = 3 * i + j;

                // A matrix rows: (d × p_model) × e_j and d × e_j
                // where e_j is the j-th standard basis vector
                // Simplified: we build the Jacobian of (? × p_model + t - p_scene) × d
                // w.r.t. ? and t

                Eigen::Vector3d cross_d_p = d.cross(p_model);
                Eigen::Vector3d cross_d_e(0, 0, 0);
                cross_d_e(j) = 1.0;

                // Jacobian of (? × p_model) × d w.r.t. ?: (d × p_model) × e_j + d × (e_j × p_model)
                // Simplified form: we need the j-th row of [d ×]
                Eigen::Vector3d jac_omega = cross_d_p;  // Approximation
                A(row, 0) = cross_d_p(0);
                A(row, 1) = cross_d_p(1);
                A(row, 2) = cross_d_p(2);

                // Jacobian of t × d w.r.t. t: d × e_j (rotation of d)
                Eigen::Vector3d cross_d_basis = d.cross(cross_d_e);
                A(row, 3) = cross_d_basis(0);
                A(row, 4) = cross_d_basis(1);
                A(row, 5) = cross_d_basis(2);

                // RHS: j-th component of (p_model - p_scene) × d
                b(row) = -constraint_rhs(j);
            }
        }

        // Solve using SVD: find least-squares solution
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        Eigen::VectorXd x = svd.solve(b);

        // evaluate conditional number
        Eigen::Vector3d singular_values = svd.singularValues();
        cond_num = singular_values(0) / singular_values(2);

        // Extract rotation vector (small angle approximation)
        Eigen::Vector3d omega(x(0), x(1), x(2));
        double theta = omega.norm();

        // Compute rotation matrix using Rodrigues' formula
        if (theta > 1e-6) {
            Eigen::Vector3d axis = omega / theta;
            R_final = AngleAxisd(theta, axis).toRotationMatrix();
        } else {
            R_final = Eigen::Matrix3d::Identity();
        }

        // Extract translation
        t_final = Eigen::Vector3d(x(3), x(4), x(5));

        // Compute error
        final_error = computeErrorPointToLine(model, scene, model_axes, R_final, t_final);

    }
    else // point to point ICP
    {
        // Compute centroids
        Eigen::Vector3d model_center = Eigen::Vector3d::Zero();
        Eigen::Vector3d scene_center = Eigen::Vector3d::Zero();

        for (const auto& p : model) model_center += p;
        for (const auto& p : scene) scene_center += p;

        model_center /= model.size();
        scene_center /= scene.size();
        
        // Center the point sets
        std::vector<Eigen::Vector3d> model_centered(model.size());
        std::vector<Eigen::Vector3d> scene_centered(scene.size());

        for (size_t i = 0; i < model.size(); ++i) {
            model_centered[i] = model[i] - model_center;
            scene_centered[i] = scene[i] - scene_center;
        }

        // Build covariance matrix H
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < model.size(); ++i) {
            H += scene_centered[i] * model_centered[i].transpose();
        }
        
        // SVD decomposition: H = U * S * V^T
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // evaluate conditional number
        Eigen::Vector3d singular_values = svd.singularValues();
        cond_num = singular_values(0) / singular_values(2);

        // Compute rotation matrix
        R_final = U * V.transpose();

        // Ensure proper rotation (det(R) = 1, not -1 for reflection)
        if (R_final.determinant() < 0) {
            V.col(2) *= -1;
            R_final = U * V.transpose();
        }
        
        // Compute translation
        t_final = scene_center - R_final * model_center;

        // Compute error
        final_error = computeError(model, scene, R_final, t_final);
    }
    return std::make_pair(std::make_pair(R_final, t_final), final_error);
}

bool FeatureManager::estimatePoseICPPrior(std::vector<Eigen::Vector3d> model, std::vector<Eigen::Vector3d> scene, const std::vector<Vector3d>& model_axes, int max_iterations, double convergence_threshold, Eigen::Matrix3d& R_init, Eigen::Vector3d& t_init, double& cond_num_final) 
{
    if((model.size() != scene.size()) || (model.size() < 3))
    {
        return false;
    }
    // Initialize with provided estimates or identity
    Eigen::Matrix3d R = R_init;
    Eigen::Vector3d t = t_init;
    double prev_error = 1e10;
    double final_error = 1e10;
    double cond_num = 0.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Correspondence step: transform model and find matches
        std::vector<Vector3d> transformed(model.size());
        for (size_t i = 0; i < model.size(); ++i) {
            transformed[i] = R * model[i] + t;
        }

        // Solve SVD with current correspondences
        cond_num = 0.0;
        auto step_result = estimatePoseSVD(transformed, scene, model_axes, cond_num);

        // Incremental update
        Eigen::Matrix3d R_delta = step_result.first.first;
        Eigen::Vector3d t_delta = step_result.first.second;

        R = R_delta * R;
        t = R_delta * t + t_delta;
        final_error = step_result.second;

        // Check convergence
        double error_change = std::abs(prev_error - final_error);
        if (error_change < convergence_threshold) {
            break;
        }

        prev_error = final_error;
    }
    
    R_init = R;
    t_init = t;
    cond_num_final = cond_num;
    return true;
}

// std::pair<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, double> FeatureManager::estimatePoseICPRansac(std::vector<Eigen::Vector3d> model, std::vector<Eigen::Vector3d> scene, const std::vector<Vector3d>& model_axes, int max_iterations, double convergence_threshold) 
// {
//     if (model.size() < 3) {
//         ROS_WARN("Error: need at least 3 points for RANSAC\n");
//         return std::make_pair(std::make_pair(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()), 1e10);
//     }

//     // RANSAC parameters
//     int num_samples = 3;  // minimum samples needed for SVD
//     int ransac_iterations = std::max(100, static_cast<int>(model.size() * 2));
//     double inlier_threshold = 0.1;  // threshold for considering a point as inlier
//     int best_inlier_count = 0;
    
//     Eigen::Matrix3d R_best = Eigen::Matrix3d::Identity();
//     Eigen::Vector3d t_best = Eigen::Vector3d::Zero();
//     double best_error = 1e10;

//     // Random number generator
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, model.size() - 1);

//     // RANSAC iterations
//     for (int ransac_iter = 0; ransac_iter < ransac_iterations; ++ransac_iter) {
//         // Sample random subset of correspondences
//         std::vector<size_t> sample_indices;
//         std::set<size_t> unique_indices;
        
//         while (unique_indices.size() < num_samples) {
//             size_t idx = dis(gen);
//             unique_indices.insert(idx);
//         }
//         sample_indices.assign(unique_indices.begin(), unique_indices.end());

//         // Build sampled point sets
//         std::vector<Eigen::Vector3d> model_sample(num_samples);
//         std::vector<Eigen::Vector3d> scene_sample(num_samples);
//         std::vector<Eigen::Vector3d> axes_sample(num_samples);

//         for (size_t i = 0; i < num_samples; ++i) {
//             model_sample[i] = model[sample_indices[i]];
//             scene_sample[i] = scene[sample_indices[i]];
//             axes_sample[i] = model_axes[sample_indices[i]];
//         }

//         // Estimate pose from sample using SVD
//         auto sample_result = estimatePoseSVD(model_sample, scene_sample, axes_sample);
//         Eigen::Matrix3d R_sample = sample_result.first.first;
//         Eigen::Vector3d t_sample = sample_result.first.second;

//         // Count inliers: points where transformed model is close to scene
//         int inlier_count = 0;
//         double sample_error = 0.0;

//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R_sample * model[i] + t_sample;
//             double point_error = (transformed - scene[i]).norm();

//             if (point_error < inlier_threshold) {
//                 inlier_count++;
//             }
//             sample_error += point_error;
//         }

//         // Update best hypothesis if this sample has more inliers
//         if (inlier_count > best_inlier_count || 
//             (inlier_count == best_inlier_count && sample_error < best_error)) {
//             best_inlier_count = inlier_count;
//             best_error = sample_error;
//             R_best = R_sample;
//             t_best = t_sample;
//         }
//     }

//     // Refine with all inliers using ICP iterations on best model
//     Eigen::Matrix3d R = R_best;
//     Eigen::Vector3d t = t_best;
//     double prev_error = best_error;
//     double final_error = best_error;

//     for (int iter = 0; iter < max_iterations; ++iter) {
//         // Filter inliers based on current transformation
//         std::vector<Eigen::Vector3d> model_inliers;
//         std::vector<Eigen::Vector3d> scene_inliers;
//         std::vector<Eigen::Vector3d> axes_inliers;

//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R * model[i] + t;
//             double point_error = (transformed - scene[i]).norm();

//             if (point_error < inlier_threshold) {
//                 model_inliers.push_back(model[i]);
//                 scene_inliers.push_back(scene[i]);
//                 axes_inliers.push_back(model_axes[i]);
//             }
//         }

//         // Need minimum 3 points to estimate pose
//         if (model_inliers.size() < 3) {
//             break;
//         }

//         // Refine with inliers
//         auto step_result = estimatePoseSVD(model_inliers, scene_inliers, axes_inliers);
//         Eigen::Matrix3d R_delta = step_result.first.first;
//         Eigen::Vector3d t_delta = step_result.first.second;

//         // Update transformation
//         R = R_delta * R;
//         t = R_delta * t + t_delta;

//         // Compute error on all points
//         final_error = 0.0;
//         for (size_t i = 0; i < model.size(); ++i) {
//             Eigen::Vector3d transformed = R * model[i] + t;
//             final_error += (transformed - scene[i]).norm();
//         }

//         // Check convergence
//         double error_change = std::abs(prev_error - final_error);
//         if (error_change < convergence_threshold) {
//             break;
//         }

//         prev_error = final_error;
//     }

//     return std::make_pair(std::make_pair(R, t), final_error);
// }

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{

    if(frameCnt > 0) // if it's not the first frame?
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &it_per_id : feature) // for all the features
        {
            if (it_per_id.estimated_depth > 0) // if it's deeper then 0 (meaning i've already traingulated the point)
            {
                int index = frameCnt - it_per_id.start_frame; // get for how many frames the feature was visible
                if((int)it_per_id.feature_per_frame.size() >= index + 1) // if i have more or equal observation of that feature then index, meaing it has been visible for the whole time (for every frame) 
                {
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0]; // get the feature position in the camera frame
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame]; // get the feature position in the world frame

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z()); // change format of the point in its first appearence
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y()); // get the last appearence in the image
                    pts3D.push_back(point3d); // add it to the vector
                    pts2D.push_back(point2d);  // add it to the vector
                }
            }
        }
        //if(!USE_TREE)
        if(true)
        {
            Eigen::Matrix3d RCam;
            Eigen::Vector3d PCam;
            // trans to w_T_cam
            RCam = Rs[frameCnt - 1] * ric[0];
            PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

            if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam * ric[0].transpose(); 
                Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

                //Eigen::Quaterniond Q(Rs[frameCnt]);
                //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
                //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
            }
            
        }

        //if(USE_TREE)
        if(false)
        {   
            Eigen::Matrix3d RCam, RCam_icp, RCam_final;
            Eigen::Vector3d PCam, PCam_icp, PCam_final;
            vector<Eigen::Vector3d> old_t_obs;
            vector<Eigen::Vector3d> cur_t_obs;
            vector<Eigen::Vector3d> old_t_normals;

            // run icp p2p or p2l closed form (SVD) to get an initial estimate: 
            // get points correspondences for all the features that apear in the current frame and their first observations
            for(auto& it_per_id : t_feature) // for each tree feature
            {
                if(it_per_id.has_frame(frameCnt)) // if i observed the feature in the current frame
                {
                    auto fpf_it = std::find_if(it_per_id.tree_per_frame.begin(), it_per_id.tree_per_frame.end(),
                                                [frameCnt](const TreePerFrame& fpf) {
                                                return fpf.frame == frameCnt;
                                                });

                    // if the feature has an observation at frame frameCnt and at least also another observation and i've optimized the feature anchor point
                    if ((fpf_it != it_per_id.tree_per_frame.end()) && (it_per_id.tree_per_frame.size() > 2) && (it_per_id.estimated_depth > 0))
                    {   
                        // evaluate the old feature position in the world frame 
                        double depth = it_per_id.estimated_depth;
                        Eigen::Vector3d anchor_pt_uv = it_per_id.tree_per_frame[0].point.normalized();
                        Eigen::Vector3d old_t_obsInCam = ric[0] * (depth * anchor_pt_uv) + tic[0];
                        Eigen::Vector3d old_t_obsInWorld = Rs[it_per_id.start_frame] * old_t_obsInCam + Ps[it_per_id.start_frame];
                        Eigen::Vector3d old_t_point3d(old_t_obsInWorld.x(), old_t_obsInWorld.y(), old_t_obsInWorld.z());
                        Eigen::Vector3d old_t_norm = Eigen::Vector3d::Zero();
                        if(ICP_P2L)
                        {
                            Eigen::Vector3d old_t_normInCam = ric[0] * it_per_id.tree_per_frame[0].n;
                            Eigen::Vector3d old_t_normInWorld = Rs[it_per_id.start_frame] * old_t_normInCam;
                            old_t_norm = old_t_normInWorld.normalized();
                        }
                        

                        // get the current observation of the feature in the camera frame
                        Eigen::Vector3d new_t_point3d(fpf_it->point.x(), fpf_it->point.y(), fpf_it->point.z());
                        old_t_obs.push_back(old_t_point3d);
                        cur_t_obs.push_back(new_t_point3d);
                        
                        if(ICP_P2L){
                            old_t_normals.push_back(old_t_norm);
                        }
                    }
                }
            }

            // evaluate ICP solution and relative conditional number
            // normal ICP with prior tranformation matrix
            RCam = Rs[frameCnt - 1] * ric[0];
            PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
            RCam_icp = RCam;
            PCam_icp = PCam;
            double cond_num_icp = 0.0;
            bool icp_result_f = estimatePoseICPPrior(old_t_obs, cur_t_obs, old_t_normals, 20, 1e-3, RCam_icp, PCam_icp, cond_num_icp);
            bool pnp_result_f = solvePoseByPnP(RCam, PCam, pts2D, pts3D);

            // evaluate pnp solution  and icp soliution
            if((pnp_result_f) && (icp_result_f)) // if you got a solution from both pnp and icp
            {   
                // mix the two results
                // evaluate the pnp condition number
                Eigen::MatrixXd A = buildEpipolarConstraintMatrix(pts2D, pts3D);
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
                double cond_num_pnp = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

                std::cout << "ICP AND PNP: icp cond num " << cond_num_icp << " pnp condition number " << cond_num_pnp << std::endl;

                // for both the methods extract the angle axis decomposition of the rotation matrix and the unit vector and magnitude of the translation vector
                // pnp
                Eigen::AngleAxisd angleAxispnp(RCam);
                double angle_pnp = angleAxispnp.angle(); 
                Eigen::Vector3d rot_axis_pnp = angleAxispnp.axis(); 
                
                Eigen::Vector3d t_unit_pnp = PCam.normalized();
                double t_mag_pnp = PCam.norm();

                // icp
                Eigen::AngleAxisd angleAxisicp(RCam_icp);
                double angle_icp = angleAxisicp.angle(); 
                Eigen::Vector3d rot_axis_icp = angleAxisicp.axis(); 
                
                Eigen::Vector3d t_unit_icp = PCam_icp.normalized();
                double t_mag_icp = PCam_icp.norm();

                double icp_coef = 0.0;
                double pnp_coef = 0.0;
                // considerations: bigger the conditional number less weight we should attribute to that component
                if(cond_num_pnp > cond_num_icp) // if the pnp algorithm is worst posed then the icp one
                {
                    // evaluate proportional coeficents
                    pnp_coef = (cond_num_icp / cond_num_pnp);
                    icp_coef = 1 - icp_coef;
                    
                }
                else if(cond_num_icp > cond_num_pnp) // if the icp algorithm is worst posed then the pnp one
                {
                    // evaluate proportional coeficents
                    icp_coef = (cond_num_pnp / cond_num_icp);
                    pnp_coef = 1 - pnp_coef;
                    

                }
                else if((cond_num_pnp < 1e6) && (cond_num_icp < 1e6)) // if both the problems are not hill posed
                {
                    // evaluate proportional coeficents
                    pnp_coef = 0.5;
                    icp_coef = 0.5;
                }
                else // both the problems are hill posed
                {
                    return;
                }

                // mix the rotation in a weighted sum
                Eigen::Vector3d final_rot_axis = (icp_coef * rot_axis_icp + pnp_coef * rot_axis_pnp).normalized();
                double final_angle = icp_coef * angle_icp + pnp_coef * angle_pnp;
                RCam_final = Eigen::AngleAxisd(final_angle, final_rot_axis).toRotationMatrix();

                // mix the translation vector
                Eigen::Vector3d final_t_unit = (icp_coef * t_unit_icp + pnp_coef * t_unit_pnp).normalized();
                double final_t_mag = icp_coef * t_mag_pnp + pnp_coef * t_mag_pnp;
                PCam_final = final_t_mag * final_t_unit;
                
                // trans to w_T_imu
                Rs[frameCnt] = RCam_final * ric[0].transpose(); 
                Ps[frameCnt] = -RCam_final * ric[0].transpose() * tic[0] + PCam_final;

                std::cout << "ICP and PNP: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
                // double anglepnp = 1.57;
                // Eigen::Matrix3d RCamtestpnp;
                // RCamtestpnp << 1, 0, 0,
                //         0, 1, 0,
                //         0, 0, 1;

                // double angleicp = 1.57;
                // Eigen::Matrix3d RCamtesticp;
                // RCamtesticp << 1, 0, 0,
                //         0, 1, 0,
                //         0, 0, 1;

                
            }
            else if((!pnp_result_f) && (icp_result_f)) // if you solved only icp
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam_icp * ric[0].transpose(); 
                Ps[frameCnt] = -RCam_icp * ric[0].transpose() * tic[0] + PCam_icp;

                std::cout << "ICP ONLY: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
            }
            else if((pnp_result_f) && (!icp_result_f)) // if you solved only pnp
            {
                // trans to w_T_imu
                Rs[frameCnt] = RCam * ric[0].transpose(); 
                Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

                std::cout << "PNP ONLY: R " << Rs[frameCnt] << " t " << Ps[frameCnt] << std::endl;
            } 
            else
            {
                std::cout << "NOR PNP NEITHER ICP" << std::endl;
            }
        }
    }
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature) // for all the features
    {
        if (it_per_id.estimated_depth > 0) // if it has already a valid estimated depth, go to the next feature 
            continue;

        if(STEREO && it_per_id.feature_per_frame[0].is_stereo) // if we are in the stereo case
        {
            int imu_i = it_per_id.start_frame; // get the frame count of the first appearence of the feature
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // get the position at time 0 of the left camera
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; // get the orientation at time 0 of the left camera
            leftPose.leftCols<3>() = R0.transpose(); // save the inverce orientation matrix in the variable
            leftPose.rightCols<1>() = -R0.transpose() * t0; // save the position vector in the opposite frame
            //cout << "left pose " << leftPose << endl;

            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1]; // get the position at time 0 of the right camera
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1]; // get the orientation at time 0 of the right camera
            rightPose.leftCols<3>() = R1.transpose(); // save the inverce orientation matrix in the variable
            rightPose.rightCols<1>() = -R1.transpose() * t1; // save the position vector in the opposite frame
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2); // get the initial image position of the feature in its first appearence in the left camera
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2); // get the initial image position of the feature in its first appearence in the right camera
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;

            triangulatePoint(leftPose, rightPose, point0, point1, point3d); // given the left and right observation of the feature evaluate its 3D position in euclidean space
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>(); // transform the 3D point from world frame to left camera frame
            double depth = localPoint.z(); // get its depth
            if (depth > 0) // if it's >0 save it in the estimated depth term of the feature (if it's smaller then 0 there has been some error in the evaluation)
                it_per_id.estimated_depth = depth;
            else // else init it to INIT_DEPTH (that sould be 0 ?)
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue; // depth evaluated so go to th enext feature
        }
        else if(it_per_id.feature_per_frame.size() > 1) // else if we are not in the stereo case and we have more then one observation of the given feature 
        {
            // do the same procedure but instead of triangluating the feature given the two observations in the two cameras, do that with the successive appearence of the feature in time
            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++;
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue; // depth evaluated so go to th enext feature
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // save the number of appearence of the feature in the used_num attribute
        if (it_per_id.used_num < 4) // if there are less then 4 observations of the feature skip and go to the next one
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // initialize the two variable

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4); // init the matrix svd_a with the given sizes (number of observations of the features times 4)
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0; // initialize the projection matrix
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // get the position of the feature at time imu_i
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; // get the orientation of the feature at time imu_i
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observation of the feature
        {
            imu_j++; // increase imu_j counter

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0]; // get the postion of the feature at frame count = imu_j
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0]; // get the orientation of the feature at frame count = imu_j
            Eigen::Vector3d t = R0.transpose() * (t1 - t0); // get the relative displacements
            Eigen::Matrix3d R = R0.transpose() * R1; // get the relative orientation
            Eigen::Matrix<double, 3, 4> P; // compute the projection matrix for the current frame
            P.leftCols<3>() = R.transpose(); // invert the reference frame (orientation)
            P.rightCols<1>() = -R.transpose() * t; // invert the reference frame (translation)
            Eigen::Vector3d f = it_per_frame.point.normalized(); // get the feature undistorted image position
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0); // add the epipolar constraint for the given observation of the feature
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j) // if the current frame is the same as the initial one skip evaluation since they are not needed
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>(); // perform the SVD obtaining the position of the point in homogeneous coordinate
        double svd_method = svd_V[2] / svd_V[3]; // compute the depth
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method; // assign it to the estimated depth
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1) // in case of small or negative depth (considered invalid)
        {
            it_per_id.estimated_depth = INIT_DEPTH; // assign as depth the INIT_DEPTH value
        }

    }

    if(USE_TREE)
    {
        for (auto &it_per_id : t_feature) // for all the features
        {
            if (it_per_id.estimated_depth > 0) // if it has already a valid estimated depth, go to the next feature 
                continue;

            // otherwise extract as initialization depth the depth of the first bservation (anchor point)
            it_per_id.used_num = it_per_id.tree_per_frame.size(); // save the number of appearence of the feature in the used_num attribute
            if (it_per_id.used_num < 3) // if there are less then 4 observations of the feature skip and go to the next one
                continue;
            double depth = it_per_id.tree_per_frame[0].point.norm();
            if (depth > 0) // if it's >0 save it in the estimated depth term of the feature (if it's smaller then 0 there has been some error in the evaluation)
                it_per_id.estimated_depth = depth;
            else // else init it to INIT_DEPTH 
                it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex, set<int> &tree_outlierIndex)
{   
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nFM ro removing outliers\nnormal features: " << std::endl;

    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin(); // for all the features (but creating two pointer to the features)
         it != feature.end(); it = it_next) // 
    {
        it_next++; // get the pointer to the next feature ( in this way it will be a fof loop with it the i feature and it_next the i+1 feature)
        int index = it->feature_id; // get the feature id
        itSet = outlierIndex.find(index); 
        if(itSet != outlierIndex.end()) // if it's in the outlier list
        {
            feature.erase(it); // remove it
            oss << "    removing " << index << std::endl;
            //printf("remove outlier %d \n", index);
        }
    }
    oss << "tree features:" << std::endl;
    if (USE_TREE)
    {   
        std::set<int>::iterator it_t_Set;
        for (auto it = t_feature.begin(), it_next = t_feature.begin(); // for all the features (but creating two pointer to the features)
            it != t_feature.end(); it = it_next) // 
        {
            it_next++; // get the pointer to the next feature ( in this way it will be a fof loop with it the i feature and it_next the i+1 feature)
            int index = it->feature_id; // get the feature id
            it_t_Set = tree_outlierIndex.find(index); 
            if(it_t_Set != tree_outlierIndex.end()) // if it's in the outlier list
            {
                t_feature.erase(it); // remove it
                oss << "    removing " << index << std::endl;
                //printf("remove outlier %d \n", index);
            }
        }
        
    }
    logMessage(oss.str());
    ///// LOG /////
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{   
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove back shift depth before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point; // get the oldest position observed of the feature
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // delete the first observation
            if (it->feature_per_frame.size() < 2) // if you have less then 2 observation
            {
                normal_features_removed++;
                feature.erase(it); // delete the feature and process the next one
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth; // get the feature 3d position
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // project it in the new frame (which one?)
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // project it in the camera frame (i guess?)
                double dep_j = pts_j(2); // re-evaluate the depth
                if (dep_j > 0) // if valid assign it to the estimated depth
                    it->estimated_depth = dep_j;
                else // otherwise give it the INIT_DEPTH value
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    if (USE_TREE)
    {
        for (auto it = t_feature.begin(), it_next = t_feature.begin();
            it != t_feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
        {
            it_next++;

            if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            {   
                it->start_frame--;
                for (auto& fpf : it->tree_per_frame) 
                    {
                        fpf.frame--; // decrease the frame attribute of each FeaturePerFrame
                    }
            }
            else
            {
                Eigen::Vector3d anchor_pt = it->tree_per_frame[0].point.normalized(); // get the oldest position observed of the tree feature
                it->tree_per_frame.erase(it->tree_per_frame.begin()); // delete the first appearence of the feature
                for (auto& fpf : it->tree_per_frame) 
                {
                    fpf.frame--; // decrease the frame attribute of each FeaturePerFrame
                }

                if (it->tree_per_frame.size() < 2) // if you have less then 2 observation
                {
                    tree_features_removed++;
                    t_feature.erase(it); // delete the feature and process the next one
                    continue;
                }
                else
                {
                    Eigen::Vector3d pts_i = anchor_pt * it->estimated_depth; // get the feature 3d position
                    Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // project it in the new frame 
                    Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // project it in the camera frame (i guess?)
                    double dep_j = pts_j(2); // re-evaluate the depth
                    if (dep_j > 0) // if valid assign it to the estimated depth
                        it->estimated_depth = dep_j;
                    else // otherwise give it the INIT_DEPTH value
                        it->estimated_depth = INIT_DEPTH;

                    it->start_frame = it->tree_per_frame[0].frame;
                }
            }
        }
    }

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove back shift depth removed\nnormal features " <<  normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "-------------------------------------------------------------------------\nFM remove back shift depth after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::removeBack()
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove back before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // delete the first appearence of the feature
            if (it->feature_per_frame.size() == 0) // if you don't have observations of the feature
            {
                normal_features_removed++;
                feature.erase(it); // delete the whole feature
            }
        }
    }

    if (USE_TREE)
    {
        for (auto it = t_feature.begin(), it_next = t_feature.begin();
            it != t_feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
        {
            it_next++;

            if (it->start_frame != 0) // if the first feature first appearence is different from 0 decrease it
            {
                it->start_frame--;
                for (auto& fpf : it->tree_per_frame) 
                {
                    fpf.frame--; // ? decrease the frame attribute of each FeaturePerFrame
                }
            }
            else
            {
                it->tree_per_frame.erase(it->tree_per_frame.begin()); // delete the first appearence of the feature
                for (auto& fpf : it->tree_per_frame) 
                {
                    fpf.frame--; // decrease the frame attribute of each FeaturePerFrame
                }
                if (it->tree_per_frame.size() == 0) // if you don't have observations of the feature
                {
                    t_feature.erase(it); // delete the whole feature
                    tree_features_removed++;
                }
                else
                {
                    it->start_frame = it->tree_per_frame[0].frame; // reset the start frame (in the case the feature has been seen for one frame yes, one not and one yes)
                }
            }
        }
    }

    ///// LOG /////
    std::ostringstream oss1;
    oss1 << "=========================================================================\nFM remove back removed\nnormal features " <<  normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    // oss1 << "-------------------------------------------------------------------------\nFM remove back after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss1.str());
    ///// LOG /////
}

void FeatureManager::removeFront(int frame_count)
{
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "=========================================================================\nFM remove front before\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    // logMessage(oss.str());
    ///// LOG /////
    int normal_features_removed = 0;
    int tree_features_removed = 0;
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
    {
        it_next++;

        if (it->start_frame == frame_count) // if the start frame is equal to the frame count 
        {
            it->start_frame--; // decrease it
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame; // get the interval between the first apperence of the feature and the first element of the window
            if (it->endFrame() < frame_count - 1) // if the feature last frame apperence is smaller then the second to last frame count ? skip it
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // otherwise erase the feature observation 
            if (it->feature_per_frame.size() == 0) // if you no longer have observations of the feature
            {
                feature.erase(it); // delete it
                normal_features_removed++;
            }
        }
    }
    vector<int> removed_id;
    if (USE_TREE)
    {   
        for (auto it = t_feature.begin(), it_next = t_feature.begin(); it != t_feature.end(); it = it_next) // for all the features, init two iterator pointing to the first and second feature 
        {
            it_next++;

            if (it->start_frame == frame_count) // if the start frame is equal to the frame count 
            {
                it->start_frame--; // decrease it
                for (auto& fpf : it->tree_per_frame) 
                {
                    fpf.frame--; // ? decrease the frame attribute of each FeaturePerFrame
                }
            }
            else
            {   
                if (it->endFrame() < frame_count - 1) // if the feature last frame apperence is smaller then the second to last frame count ? skip it
                    continue;
                if(it->has_frame(WINDOW_SIZE - 1))
                {
                    auto fpf_it = std::find_if(it->tree_per_frame.begin(), it->tree_per_frame.end(),
                                [WINDOW_SIZE](const TreePerFrame& item) {
                                    return item.frame == WINDOW_SIZE - 1;
                                });
                    if (fpf_it != it->tree_per_frame.end())
                    {                          
                        it->tree_per_frame.erase(fpf_it); // Erase by iterator the frame WINDOW_SIZE -1
                    }
                }
                // decrease any feature observation with frame > WINDOW_SIZE - 1 by 1
                for(auto& obs : it->tree_per_frame) 
                {
                    if(obs.frame > WINDOW_SIZE - 1)
                    {
                        obs.frame -= 1;
                    }
                }
                if (it->tree_per_frame.size() == 0){ // if you no longer have observations of the feature
                    removed_id.push_back(it->feature_id);
                    t_feature.erase(it); // delete it

                    tree_features_removed++;
                }
                else if(it->start_frame != it->tree_per_frame[0].frame)
                {
                    std::cout << "REMOVE FRONT ERROR prev start frame " << it->start_frame << " new one " << it->tree_per_frame[0].frame << std::endl;
                    it->start_frame = it->tree_per_frame[0].frame;
                }
            }
        }
        std::cout << std::endl;
    }
    ///// LOG /////
    std::ostringstream oss1;
    // oss1 << "-------------------------------------------------------------------------\nFM tree features removed " << std::endl;
    // for(const auto& r_id : removed_id)
    // {
    //     oss1 << r_id << std::endl;
    // }
    oss1 << "-------------------------------------------------------------------------\nFM remove front removed\nnormal features " << normal_features_removed << "\ntree features " << tree_features_removed << std::endl;
    
    // oss1 << "-------------------------------------------------------------------------\nFM remove front after\nnormal features: " << std::endl;
    // for (auto &it_per_id : feature) // for all the visual features
    // {
    //     oss1 << "    feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " estimated depth " << it_per_id.estimated_depth << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        L pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " R pos " << it_per_frame.pointRight.x() << " " << it_per_frame.pointRight.y() << " " << it_per_frame.pointRight.z() << std::endl;
    //     }
    // }
    // oss1 << "tree features:" << std::endl;
    // for (auto &it_per_id : t_feature) // for all the visual features
    // {
    //     oss1 << "    tree feature " << it_per_id.feature_id << " start frame " << it_per_id.start_frame  << " solver flag " << it_per_id.solve_flag << " obs:" << std::endl;
    //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
    //     {  
    //         oss1 << "        pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " " << it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << std::endl;
    //     }
    // }
    logMessage(oss1.str());
    ///// LOG /////
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // get the feature informations (x, y, x, p_u, p_v, v_x, v_y) of the time before the last time we encountered the feature
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // get the feature informations (x, y, x, p_u, p_v, v_x, v_y) of the last time we encountered the feature (would be current frame when teh function is called)

    double ans = 0;
    Vector3d p_j = frame_j.point; // get the x, y, z of j

    double u_j = p_j(0); // get the undistorted x
    double v_j = p_j(1); // get the undistorted y

    Vector3d p_i = frame_i.point; // get the x, y, z of i
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2); // get the z of i
    double u_i = p_i(0) / dep_i; // normalize
    double v_i = p_i(1) / dep_i; // normalize
    double du = u_i - u_j, dv = v_i - v_j; // evaluate the difference in position of the feature point in the image between last two apperence of the feature

    double dep_i_comp = p_i_comp(2); // same as above ?
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp))); // get minimum distance in the image of the feature point in the last two frame in which it appears

    return ans;
}

double FeatureManager::compensated_tree_Parallax2(const TreePerId &it_per_id, int frame_count)
{
    // get T_tree_lcam
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

    Eigen::VectorXd K_vec(9);
    K_vec << ref_frame["tree_camera"].k[0], 0.0, ref_frame["tree_camera"].k[2], 0.0, ref_frame["tree_camera"].k[4], ref_frame["tree_camera"].k[5], 0.0, 0.0, 1.0;
    Eigen::Matrix3d K_mat = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_vec.data());

    //parallax betwwen seconde last frame and third last frame
    Eigen::Vector2d p_img_i, p_img_j;
    for(const auto& it_per_frame : it_per_id.tree_per_frame)
    {
        if(it_per_frame.frame == frame_count - 2)
        {
            // evaluate the point position in the image 
            // get the 3d position
            Eigen::Vector4d p_3d_h;
            p_3d_h << it_per_frame.point.x(), it_per_frame.point.y(), it_per_frame.point.z(), 1;
                                
            // project the point in the tree view
            Eigen::Vector4d p_3dt_h = T_lcam_tree * p_3d_h;
            Eigen::Vector3d p_3dt = p_3dt_h.block<3, 1>(0, 0);
            
            // project in the image plane
            Eigen::Vector3d p_uvs = K_mat * p_3dt;
            Eigen::Vector2d p_img;
            p_img_i << p_uvs[0] / p_uvs[2], p_uvs[1] / p_uvs[2];
        }
        else if(it_per_frame.frame == frame_count - 1)
        {
            // evaluate the point position in the image 
            // get the 3d position
            Eigen::Vector4d p_3d_h;
            p_3d_h << it_per_frame.point.x(), it_per_frame.point.y(), it_per_frame.point.z(), 1;
                                
            // project the point in the tree view
            Eigen::Vector4d p_3dt_h = T_lcam_tree * p_3d_h;
            Eigen::Vector3d p_3dt = p_3dt_h.block<3, 1>(0, 0);
            
            // project in the image plane
            Eigen::Vector3d p_uvs = K_mat * p_3dt;
            Eigen::Vector2d p_img;
            p_img_j << p_uvs[0] / p_uvs[2], p_uvs[1] / p_uvs[2];
        }
    }
    
    double ans = 0;
    double du = p_img_i[0] - p_img_j[0], dv = p_img_i[1] - p_img_j[1];

    ans = max(ans, sqrt(du * du + dv * dv)); 

    return ans;
}