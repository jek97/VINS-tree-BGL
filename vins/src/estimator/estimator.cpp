/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/
#include <iostream>
#include "../vins_node.h"
#include "estimator.h"
#include "../utility/visualization.h"
#include "std_msgs/msg/string.hpp"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();
    if(USE_TREE)
    {
        while(!tree_featureBuf.empty())
            tree_featureBuf.pop();
    }

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    input_tree_Cnt = 0;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
            pre_integrations[i] = nullptr;
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
    {
        delete tmp_pre_integration;
        tmp_pre_integration = nullptr;
    }
    if (last_marginalization_info != nullptr)
    {
        delete last_marginalization_info;
        last_marginalization_info = nullptr;
    }

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::setParameter_topic()
{
    mProcess.lock();
    // set extrinsic parameters
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    
    if(NUM_OF_CAM == 1)
    {
        featureTracker.setIntrinsicParameter_topic(ref_frame["left_camera"], "left_camera");
        ROW = ref_frame["left_camera"].height;
        COL = ref_frame["left_camera"].width;
    }
    else if(NUM_OF_CAM == 2)
    {   
        vector<string> frames = {"left_camera", "right_camera"};
        for(string f : frames)
        {
            featureTracker.setIntrinsicParameter_topic(ref_frame[f], f);
            ROW = ref_frame[f].height;
            COL = ref_frame[f].width;
        }
    }

    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        if(USE_TOPIC)
            setParameter_topic();
        else
            setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    ///// LOG /////
    // std::ostringstream oss;
    // oss << "T " << t << std::endl;
    // logMessage(oss.str());
    ///// LOG /////
    if(_img1.empty()) // meaning if you have only the _img0 and so monocular case
        featureFrame = featureTracker.trackImage(t, _img); // track only the single image
    else // you are in the stereo case
        featureFrame = featureTracker.trackImage(t, _img, _img1); // track the image of left and right camera
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    if (SHOW_TRACK) // if we want to show the track of the points between previous and last frame
    {   
        cv::Mat imgTrack = featureTracker.getTrackImage(); // get the track 
        pubTrackImage(imgTrack, t); // publish it
    }
    if(MULTIPLE_THREAD) // if we are in the multiple thread case
    {     
        if(inputImageCnt % 2 == 0) // if the image is a pair one ( meaning that we push in the feature buffer one feature yes and one not)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame)); // push the feature in the buffer
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame)); // add the feature to the buffer
        mBuf.unlock();
        TicToc processTime;
        processMeasurements(); // process the measurements
        printf("process time: %f\n", processTime.toc());
    }
    
}
void Estimator::logMessage(const std::string& message) 
{
    const std::string LOG_FILE_PATH = "/home/glugano/Desktop/log.txt";

    std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
    if (!logFile) {
        std::cerr << "Error: Unable to open log file." << std::endl;
        return;
    }

    logFile << message << std::endl;
}

void Estimator::inputForest(double t, std::pair<bool, ObservedForest> &forest)
{
    input_tree_Cnt++;
    pair<bool, pair<double, vector<pair<int, ObservedTree>>>> t_featureFrame;
    if(forest.first) // if we received a valid forest
    {
        t_featureFrame.first = true;
        {
            std::scoped_lock lk(Mlmodel, featureTracker.Mlmodel);
            featureTracker.last_model_forest = last_model_forest;
        }
        featureTracker.last_R   = Rs[WINDOW_SIZE];
        featureTracker.last_P   = Ps[WINDOW_SIZE];
        featureTracker.last_ric = ric[0];
        featureTracker.last_tic = tic[0];
        t_featureFrame.second = featureTracker.trackForest(t, forest.second);
        auto [joinedImage, cam_info, match_time] = featureTracker.getTreeMatch();
        pubTreeMatchImage(joinedImage, cam_info, match_time);
    }
    else
    {
        t_featureFrame.first = false;
        t_featureFrame.second.first = t;
    }

    if(MULTIPLE_THREAD) // if we are in the multiple thread case
    {     
        if(input_tree_Cnt % 2 == 0) // if the image is a pair one ( meaning that we push in the feature buffer one feature yes and one not)
        {
            mBuf.lock();
            tree_featureBuf.push(t_featureFrame); // add the feature to the buffer
            mBuf.unlock();
            
        }
    }
    else
    {   
        mBuf.lock();
        tree_featureBuf.push(t_featureFrame); // add the feature to the buffer
        mBuf.unlock();
        processMeasurements();

    }
    
    return;
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    //ROS_ERROR("deprecated at VINS-Fusion");
    //assert(0);
    ROS_WARN("estimator load feature");
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector) // t0 = prev time, t1 = curr time
{   
    if(accBuf.empty()) // if i don't have accelleration measures quit
    {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first) // if the current time is smaller then the newest accelleration obtained from the imu
    {
        while (accBuf.front().first <= t0) // as long as prev time is bigger then the oldest accelleration measure in the buffer, delete the acceleration and gyro measurements older then prev time
        {
            // std::cout << "t_imu: " << std::fixed << accBuf.front().first << "  t_0: " << std::fixed << t0 << "   gyr_buf size: " << gyrBuf.size() << std::endl;
            // std::cout << "1) acc pop" << std::endl;
            accBuf.pop();
            // std::cout << "1) gyr pop" << std::endl;
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1) // as long as curr time is bigger then the oldest accelleration measure in the buffer, add at the end of the accvector and gyrvector the relative measurement, 
        {
            accVector.push_back(accBuf.front());
            // std::cout << "2) acc pop" << std::endl;
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            // std::cout << "2) gyr pop" << std::endl;
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{   
    if(!accBuf.empty() && t <= accBuf.back().first)
    {
        return true;
    }
    else
        return false;
}

void Estimator::processMeasurements()
{   
    while (1)
    {   
        if (USE_TREE)
        {   
            // if you have both stereoimages features and tree features relative to the same time
            if ((!featureBuf.empty() && !tree_featureBuf.empty()) && ((featureBuf.front().first + td) == (tree_featureBuf.front().second.first + td)))// if we have both trees and image informations in the buffer
            {   
                if(tree_featureBuf.front().first) // if it's a valid forest
                {
                    // instantiate variables to keep the time step data
                    pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
                    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;

                    // extract time and stereoimages features
                    feature = featureBuf.front(); // get the first feature in the buffer
                    curTime = feature.first + td; // get the first element of the feature (its time)

                    // extract tree features
                    pair<double, vector<pair<int, ObservedTree>>> t_feature = tree_featureBuf.front().second;
                    ///// LOG /////
                    std::ostringstream oss;
                    oss << "=========================================================================\nE pm forest at time " << std::setprecision(15) << curTime << std::endl;
                    for (size_t i = 0; i < t_feature.second.size(); ++i) {
                        const ObservedTree& tree = t_feature.second[i].second;
                        oss << "  tree " << i << " (prev_idx=" << t_feature.second[i].first << ")\n";
                        for (int v = 0; v < (int)boost::num_vertices(tree); ++v) {
                            const ObservedNode& n = tree[v];
                            oss << "    node " << n.id << " pos " << n.x << " " << n.y << " " << n.z << " track cnt " << n.track_cnt << "\n";
                        }
                    }
                    logMessage(oss.str());
                    ///// LOG /////

                    while(1)
                    {
                        if ((!USE_IMU  || IMUAvailable(feature.first + td))) // if we set to not use the imu or there are imu measurements "yunger" then the frame time exit
                            break;
                        else // else wait for an imu measurements
                        {
                            printf("wait for imu ... \n");
                            if (! MULTIPLE_THREAD)
                                return;
                            std::chrono::milliseconds dura(5);
                            std::this_thread::sleep_for(dura);
                        }
                    }
                    // cout << "2" << endl;
                    mBuf.lock();
                    if(USE_IMU)
                    {
                        // cout << "2-1)" << endl;
                        getIMUInterval(prevTime, curTime, accVector, gyrVector); // basically it set accvector and gyrvector as all the acc and gyro measurements between prev_time and curr_time
                        // cout << "2-2)" << endl;
                    }

                    featureBuf.pop(); // remove newest frame from the feature buffer
                    tree_featureBuf.pop(); // remove newest frame from the tree feature buffer
                    mBuf.unlock();

                    // cout << "3" << endl;
                    if(USE_IMU)
                    {
                        if(!initFirstPoseFlag)
                            initFirstIMUPose(accVector);
                        for(size_t i = 0; i < accVector.size(); i++) // for every accelleration obtained in th accelleration vector (a series of samples from the imu)
                        {
                            double dt;
                            if(i == 0) // if first
                                dt = accVector[i].first - prevTime; // get as delta time the time between prev time and the sample
                            else if (i == accVector.size() - 1) // if last 
                                dt = curTime - accVector[i - 1].first; // get as delta time the time between the last sample and current time
                            else // else if you are in between the samples
                                dt = accVector[i].first - accVector[i - 1].first; // get as delta time the time between the two samples
                            processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // basically it obtain an unbiased estimation of position (Ps), velocity (Vs), rotation matrix (Rs) between the prev_time and curr_time
                        }
                    }
                    // cout << "4" << endl;
                    
                    mProcess.lock();
                    
                    // cout << "feature second" << feature.second;
                    processImage_tree(feature.first, feature.second, t_feature);
                    
                    prevTime = curTime; // update variable for next iteration

                    // cout << "5" << endl;
                    // print statistics
                    printStatistics(*this, 0);
                    
                    // create header to form the messages
                    std_msgs::msg::Header header;
                    header.frame_id = "world";

                    int sec_ts = (int)feature.first;
                    uint nsec_ts = (uint)((feature.first - sec_ts) * 1e9);
                    header.stamp.sec = sec_ts;
                    header.stamp.nanosec = nsec_ts;
                    
                    // publish the messages
                    pubOdometry(*this, header);
                    // cout << "5-1" << endl;
                    //pubKeyPoses(*this, header);
                    // cout << "5-2" << endl;
                    pubCameraPose(*this, header);
                    // cout << "5-3" << endl;
                    pubPointCloud(*this, header);
                    // cout << "5-4" << endl;
                    pubKeyframe(*this);
                    // cout << "5-5" << endl;
                    pubTF(*this, header);
                    // cout << "5-6" << endl;
                    mProcess.unlock();


                    // cout << "6" << endl;

                    // assert(0);
                }
                else
                {
                    // remove the last frame since we don't have trees
                    mBuf.lock();
                    featureBuf.pop(); // remove newest frame from the feature buffer
                    tree_featureBuf.pop(); // remove newest frame from the tree feature buffer
                    mBuf.unlock();
                }
            }
            
            if (! MULTIPLE_THREAD)
            {
                break;
            }
            
            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
        }
        else
        {
            pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
            vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
            if(!featureBuf.empty()) // if there are features
            {
                // cout << "1" << endl;
                feature = featureBuf.front(); // get the first feature in the buffer
                curTime = feature.first + td; // get the first element of the feature (its time)
                // std::cout << "t0: " << std::fixed << curTime << std::endl;
                while(1)
                {
                    if ((!USE_IMU  || IMUAvailable(feature.first + td))) // if we set to not use the imu or there are imu measurements "yunger" then the frame time exit
                        break;
                    else // else wait for an imu measurements
                    {
                        printf("wait for imu ... \n");
                        if (! MULTIPLE_THREAD)
                            return;
                        std::chrono::milliseconds dura(5);
                        std::this_thread::sleep_for(dura);
                    }
                }
                // cout << "2" << endl;
                mBuf.lock();
                if(USE_IMU)
                {
                    // cout << "2-1)" << endl;
                    getIMUInterval(prevTime, curTime, accVector, gyrVector); // basically it set accvector and gyrvector as all the acc and gyro measurements between prev_time and curr_time
                    // cout << "2-2)" << endl;
                }

                featureBuf.pop(); // remove newest frame from the feature buffer
                mBuf.unlock();

                // cout << "3" << endl;
                if(USE_IMU)
                {
                    if(!initFirstPoseFlag)
                        initFirstIMUPose(accVector);
                    for(size_t i = 0; i < accVector.size(); i++) // for every accelleration obtained in th accelleration vector (a series of samples from the imu)
                    {
                        double dt;
                        if(i == 0) // if first
                            dt = accVector[i].first - prevTime; // get as delta time the time between prev time and the sample
                        else if (i == accVector.size() - 1) // if last 
                            dt = curTime - accVector[i - 1].first; // get as delta time the time between the last sample and current time
                        else // else if you are in between the samples
                            dt = accVector[i].first - accVector[i - 1].first; // get as delta time the time between the two samples
                        processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // basically it obtain an unbiased estimation of position (Ps), velocity (Vs), rotation matrix (Rs) between the prev_time and curr_time
                    }
                }
                // cout << "4" << endl;

                mProcess.lock();
                // cout << "feature second" << feature.second;
                processImage(feature.second, feature.first); // process all the measurements
                prevTime = curTime; // update variable for next iteration

                // cout << "5" << endl;
                // print statistics
                printStatistics(*this, 0);
                
                // create header to form the messages
                std_msgs::msg::Header header;
                header.frame_id = "world";

                int sec_ts = (int)feature.first;
                uint nsec_ts = (uint)((feature.first - sec_ts) * 1e9);
                header.stamp.sec = sec_ts;
                header.stamp.nanosec = nsec_ts;

                // publish the messages
                pubOdometry(*this, header);
                // cout << "5-1" << endl;
                pubKeyPoses(*this, header);
                // cout << "5-2" << endl;
                pubCameraPose(*this, header);
                // cout << "5-3" << endl;
                pubPointCloud(*this, header);
                // cout << "5-4" << endl;
                pubKeyframe(*this);
                // cout << "5-5" << endl;
                pubTF(*this, header);
                // cout << "5-6" << endl;
                mProcess.unlock();


                // cout << "6" << endl;

                // assert(0);
            }

            // cout << "[processMeasurements]  loop - end" << endl;

            if (! MULTIPLE_THREAD)
                break;

            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
        }
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) //accVector[i].first, dt, accVector[i].second, gyrVector[i].second
{
    if (!first_imu) // if you are right after the clear state (meaning this is the first time you process the IMU)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); // save in new vector
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // save in new variable

        dt_buf[frame_count].push_back(dt); // save in new vector
        linear_acceleration_buf[frame_count].push_back(linear_acceleration); // save in new vector
        angular_velocity_buf[frame_count].push_back(angular_velocity); // save in new vector

        // NOTE: Bas = accellerometr bias, Bgs = gyroscope bias
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // rotate the previous accelleration to the new framse (SUPPOSE) 
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); // evaluate rotation matrix between prev_time and curr_time frame (globally)
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // average between previous and new accelleration
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; // formula for position update from prev_time to curr_time (globally)
        Vs[j] += dt * un_acc; // formula for velocity update from prev_time to curr_time (globally)
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header) // note image is actually the featureFrame arguments
{
    //cout << std::fixed << header << endl;

    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // check if it's a keyframe or not and in case set the corresponding flag
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
        ///// LOG /////
        // std::ostringstream oss;
        // oss << "keyframe" << std::endl;
        // logMessage(oss.str());
        ///// LOG /////
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
        ///// LOG /////
        // std::ostringstream oss;
        // oss << "no keyframe" << std::endl;
        // logMessage(oss.str());
        ///// LOG /////
    }
    
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header; // associate the time of the image received to the actual frame_count and save it in the Headers vector

    ImageFrame imageframe(image, header); // re-associate the feature frame to its time, actually this variable has the same datastructure of the feature
    imageframe.pre_integration = tmp_pre_integration; // add to the imageframe variable the accellerations preintegreted evaluated before
    all_image_frame.insert(make_pair(header, imageframe)); // add the imageframe to all_image_frame variable
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // reset tmp_pre_integration for next iterations

    if(ESTIMATE_EXTRINSIC == 2) // if the extrinsic parameters are not calibrated, calibrate it
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) // if it's not the first frame
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }


    if (solver_flag == INITIAL) // if we are in initialization phase (i will skip this part for now ?)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE) //if the frame count (that start from 0) is as big as the WINDOW_SIZE (this means that probably we got there after a clear_state call)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) // if we have the extrinsic parameters and the message we received is at least 0.1 sec farther from the last clear state  
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric); // if not using the imu get the IMU pose estimate by vision methods
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric); // obtain the depth of all the features

        // optimization
        TicToc t_solve;
        optimization(); // perform the non linear optimization with the bundle adjustment
        //ROS_INFO("solver costs: %f [ms]", t_solve.toc());

        set<int> removeIndex;
        set<int> empty_set;
        outliersRejection(removeIndex, empty_set); // remove the outliers based on the average reprojection error
        f_manager.removeOutlier(removeIndex, empty_set); // remove the outlier from the features
        if (! MULTIPLE_THREAD) // in case of single thred
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame(); // predict the points in the next frame and add them to the predicted point for their matching with the next frame features
        }
            
        
        if (failureDetection()) // if you get any failure restart the system
        {      
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            if(USE_TOPIC)
                setParameter_topic();
            else
                setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        ///// LOG /////
        // std::ostringstream oss;
        // oss << "----------------------------------------------------------------------------------------------------------------------------------------------------------------- " << std::endl;
        // oss << "Time " <<  header << " frame count: " << frame_count << std::endl << "feature data struct: " << std::endl;
        // for (auto &it_per_id : f_manager.feature) // for all the visual features
        // {
        //     oss << "    id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag << " observations:" << std::endl;

        //     for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
        //     {
        //         oss << "        obs pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << std::endl;
        //     }
        // }

        // oss << "tree feaure data struct: " << std::endl;
        // for (auto &it_per_id : f_manager.t_feature) // for all the visual features
        // {
        //     oss << "    id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag << " observations:" << std::endl;

        //     for (auto &it_per_frame : it_per_id.tree_per_frame) // for all the observations of the feature
        //     {
        //         oss << "        obs pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << " frame: " << it_per_frame.frame << std::endl;
        //     }
        // }

        // logMessage(oss.str());
        ///// LOG /////
        slideWindow(); // slide all the quantities in the window
        f_manager.removeFailures(); // delete features for which wasn't possible to do the optimization?
        // prepare output of VINS
        key_poses.clear(); // clear all the keypose
        for (int i = 0; i <= WINDOW_SIZE; i++) // for all the element in the window
            key_poses.push_back(Ps[i]); // add the new pose in the keypose buffer
        
        // update initial and finale orientation, position for next iteration 
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates(); // update state of the estimator
    }  
}

void Estimator::rebuildLastModelForest()
{
    std::lock_guard<std::mutex> lk(Mlmodel);
    last_model_forest.clear();

    std::ostringstream oss_lmf;
    oss_lmf << "=========================================================================\nrebuildLastModelForest\n";

    size_t tree_idx = 0;
    for (const auto &model_tree : f_manager.t_feature)
    {
        ObservedTree obs_tree;
        const int nv = (int)boost::num_vertices(model_tree);
        oss_lmf << "  model_tree " << tree_idx++ << "  nodes=" << nv
                << "  edges=" << boost::num_edges(model_tree) << "\n";

        // Pass 1: add vertices in the same index order as model_tree (vecS indices
        // are stable 0..nv-1, so vertex descriptors map 1-to-1 between the two graphs).
        for (int v = 0; v < nv; ++v)
        {
            const ModelNode &node = model_tree[v];
            ObservedNode obs;
            obs.id        = node.feature_id;
            obs.ex_id     = std::to_string(node.feature_id); // needed by isomorphism / matching
            obs.track_cnt = (int)node.tree_per_frame.size();

            int anchor_frame = -1;
            if (node.estimated_depth > 0 && !node.tree_per_frame.empty())
            {
                anchor_frame = node.start_frame;
                const Vector3d &pt0 = node.tree_per_frame[0].point;
                Vector3d pts_imu = ric[0] * (node.estimated_depth * pt0.normalized()) + tic[0];
                Vector3d pts_w   = Rs[anchor_frame] * pts_imu + Ps[anchor_frame];
                obs.x = pts_w.x(); obs.y = pts_w.y(); obs.z = pts_w.z();
                obs.timestamp = Headers[anchor_frame];
            }
            else if (!node.tree_per_frame.empty())
            {
                const TreePerFrame &latest = node.tree_per_frame.back();
                anchor_frame = latest.frame;
                Vector3d pts_imu = ric[0] * latest.point + tic[0];
                Vector3d pts_w   = Rs[anchor_frame] * pts_imu + Ps[anchor_frame];
                obs.x = pts_w.x(); obs.y = pts_w.y(); obs.z = pts_w.z();
                obs.timestamp = Headers[anchor_frame];
            }

            // Log anchor frame, rotation matrix, and translation vector for this node.
            oss_lmf << "    node id=" << node.feature_id << " anchor_frame=" << anchor_frame << "\n";
            if (anchor_frame >= 0)
            {
                const Matrix3d &R = Rs[anchor_frame];
                const Vector3d &P = Ps[anchor_frame];
                oss_lmf << "      R=[ [" << R(0,0) << ", " << R(0,1) << ", " << R(0,2) << "],\n"
                        << "          [" << R(1,0) << ", " << R(1,1) << ", " << R(1,2) << "],\n"
                        << "          [" << R(2,0) << ", " << R(2,1) << ", " << R(2,2) << "] ]\n"
                        << "      t=[ " << P(0) << ", " << P(1) << ", " << P(2) << " ]\n";
            }

            boost::add_vertex(obs, obs_tree);
        }

        // Pass 2: mirror all directed edges (parent→child) from model_tree to obs_tree.
        // Iterate vertex-by-vertex through out_edges so the copy is explicit and does not
        // rely on the global boost::edges() ordering for bidirectionalS graphs.
        for (int v = 0; v < nv; ++v)
            for (auto e : boost::make_iterator_range(boost::out_edges(v, model_tree)))
                boost::add_edge(v, (int)boost::target(e, model_tree), obs_tree);

        last_model_forest.push_back(std::move(obs_tree));
    }

    oss_lmf << "---------  obs_trees (last_model_forest)  ---------\n";
    for (size_t ti = 0; ti < last_model_forest.size(); ++ti)
    {
        const ObservedTree& t = last_model_forest[ti];
        oss_lmf << "  tree " << ti << "  nodes=" << boost::num_vertices(t)
                << "  edges=" << boost::num_edges(t) << "\n";
        for (int v = 0; v < (int)boost::num_vertices(t); ++v)
        {
            const ObservedNode& n = t[v];
            std::string parent_exid = "none";
            auto [ie, ie_end] = boost::in_edges(v, t);
            if (ie != ie_end)
                parent_exid = t[boost::source(*ie, t)].ex_id;
            std::string sons_str;
            for (auto e : boost::make_iterator_range(boost::out_edges(v, t)))
                sons_str += t[boost::target(e, t)].ex_id + " ";
            if (sons_str.empty()) sons_str = "none";
            oss_lmf << "    node id=" << n.id << " ex_id=" << n.ex_id
                    << " pos=(" << n.x << ", " << n.y << ", " << n.z << ")"
                    << " parent=" << parent_exid << " sons=[" << sons_str << "]\n";
        }
    }
    logMessage(oss_lmf.str());
}

void Estimator::processImage_tree(const double header, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const pair<double, vector<pair<int, ObservedTree>>> &tree)
{   
         
    ROS_DEBUG("new tree and image coming ------------------------------------------");
    ROS_DEBUG("Adding image feature points %lu", image.size());
    ROS_DEBUG("Adding tree feature points %lu", tree.second.size());
    
    if (f_manager.addFeatureTreeCheckParallax(frame_count, header, image, tree, td, Rs, Ps, ric[0], tic[0])) // check if it's a keyframe or not and in case set the corresponding flag
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    {
        std::ostringstream oss;
        oss << "=========================================================================\n"
            << "processImage_tree after addFeatureTreeCheckParallax"
            << "  frame=" << frame_count
            << "  trees=" << f_manager.t_feature.size() << "\n";
        for (size_t ti = 0; ti < f_manager.t_feature.size(); ++ti)
        {
            const ModelTree &mt = f_manager.t_feature[ti];
            const int nv = (int)boost::num_vertices(mt);
            oss << "  tree " << ti << "  nodes=" << nv
                << "  edges=" << boost::num_edges(mt) << "\n";
            for (int v = 0; v < nv; ++v)
            {
                const ModelNode &node = mt[v];
                oss << "    node id=" << node.feature_id
                    << " depth=" << node.estimated_depth
                    << " obs=" << node.tree_per_frame.size();
                if (!node.tree_per_frame.empty())
                {
                    const Vector3d &pt = node.tree_per_frame.back().point;
                    oss << " latest_frame=" << node.tree_per_frame.back().frame
                        << " point=(" << pt.x() << ", " << pt.y() << ", " << pt.z() << ")";
                }
                oss << "\n";
            }
        }
        logMessage(oss.str());
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header; // associate the time of the image received to the actual frame_count and save it in the Headers vector

    ///// LOG /////
    std::ostringstream oss;
    oss << "-------------------------------------------------------------------------\nE pit frame count " << frame_count << " is time " << std::setprecision(15) << header << std::endl;
    logMessage(oss.str());
    ///// LOG /////

    ImageFrame imageframe(image, header); // re-associate the feature frame to its time, actually this variable has the same datastructure of the feature
    imageframe.pre_integration = tmp_pre_integration; // add to the imageframe variable the accellerations preintegreted evaluated before
    all_image_frame.insert(make_pair(header, imageframe)); // add the imageframe to all_image_frame variable
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // reset tmp_pre_integration for next iterations

    if(ESTIMATE_EXTRINSIC == 2) // if the extrinsic parameters are not calibrated, calibrate it
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) // if it's not the first frame
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
    if (solver_flag == INITIAL) // if we are in initialization phase (i will skip this part for now ?)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE) //if the frame count (that start from 0) is as big as the WINDOW_SIZE (this means that probably we got there after a clear_state call)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) // if we have the extrinsic parameters and the message we received is at least 0.1 sec farther from the last clear state
                {
                    result = initialStructure();
                    initial_timestamp = header;
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    rebuildLastModelForest();
                    ROS_INFO("Initialization finish!");
                }
                else
                {
                    slideWindow();
                    rebuildLastModelForest();
                }
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);


            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                rebuildLastModelForest();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();
            
            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                rebuildLastModelForest();
                ROS_INFO("Initialization finish!");
            }
            else
            {
                rebuildLastModelForest();
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        if(!USE_IMU)
        {   
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric); // if not using the imu get the IMU pose estimate by vision methods
        }
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric); // obtain the depth of all the features

        // optimization
        TicToc t_solve;

        optimization(); // perform the non linear optimization with the bundle adjustment
        
        //ROS_INFO("solver costs: %f [ms]", t_solve.toc());


        set<int> removeIndex;
        set<int> remove_tree_Index;
        outliersRejection(removeIndex, remove_tree_Index); // remove the outliers based on the average reprojection error 
        f_manager.removeOutlier(removeIndex, remove_tree_Index); // remove the outlier from the features 
        if (! MULTIPLE_THREAD) // in case of single thred
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame(); // predict the points in the next frame and add them to the predicted point for their matching with the next frame features TODO FOR TREE
        }
                
        
        if (failureDetection()) // if you get any failure restart the system
        {      
            ///// LOG /////
            std::ostringstream oss;
            oss << "################################################ PREV DATA ###############################################################" << std::endl; 
            oss << "frame " << frame_count << std::endl;
            oss << "---------------------------------------------- feature buffer ------------------------------------------------------------" << std::endl;
            for (auto &it_per_id : prev_deb_feature) // for all the visual features
            {
                oss << "id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag << " estimated depth " << it_per_id.estimated_depth << " observations:" << std::endl;
                int count = 0;
                for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
                {
                    oss << "    obs 3d pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << " 2d pos " << it_per_frame.uv.x() << " " << it_per_frame.uv.y() << " velocity " << it_per_frame.velocity.x() << " " << it_per_frame.velocity.y() << " td " << it_per_frame.cur_td << " frame " << it_per_id.start_frame + count << std::endl;
                    count += 1;
                }
            }
            logMessage(oss.str());
            oss.str("");
            oss.clear();
            oss << "---------------------------------------------- tree buffer ------------------------------------------------------------" << std::endl;
            for (auto &_mt : prev_deb_t_feature)
            for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
            {
                auto &it_per_id = _mt[_v];
                oss << "id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag <<  " observations:" << std::endl;
                for (auto &it_per_frame : it_per_id.tree_per_frame)
                {
                    oss << "    obs 3d pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << " velocity " << it_per_frame.velocity.x() << " " << it_per_frame.velocity.y() << " " << it_per_frame.velocity.z()<< " td " << it_per_frame.cur_td << " frame " << it_per_frame.frame << std::endl;
                }
            }
            logMessage(oss.str());
            oss.str("");
            oss.clear();
            oss << "---------------------------------------------- position ------------------------------------------------------------" << std::endl;
            for(auto p : prev_deb_Ps){
                oss << p << std::endl << std::endl;
            }
            oss << "---------------------------------------------- orientation ------------------------------------------------------------" << std::endl;
            for(auto r : prev_deb_Rs){
                oss << r << std::endl << std::endl;
            } 
            logMessage(oss.str());
            oss.str("");
            oss.clear();
            oss << "################################################ CUR DATA ###############################################################" << std::endl; 
            oss << "---------------------------------------------- feature buffer ------------------------------------------------------------" << std::endl;
            for (auto &it_per_id : f_manager.feature) // for all the visual features
            {
                oss << "id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag << " estimated depth " << it_per_id.estimated_depth << " observations:" << std::endl;
                int count = 0;
                for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
                {
                    oss << "    obs 3d pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << " 2d pos " << it_per_frame.uv.x() << " " << it_per_frame.uv.y() << " velocity " << it_per_frame.velocity.x() << " " << it_per_frame.velocity.y() << " td " << it_per_frame.cur_td << " frame " << it_per_id.start_frame + count << std::endl;
                    count += 1;
                }
            }
            logMessage(oss.str());
            oss.str("");
            oss.clear();
            oss << "---------------------------------------------- tree buffer ------------------------------------------------------------" << std::endl;
            for (auto &_mt : f_manager.t_feature)
            for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
            {
                auto &it_per_id = _mt[_v];
                oss << "id " << it_per_id.feature_id << " start frame " << it_per_id.start_frame << " solver falg: " << it_per_id.solve_flag <<  " observations:" << std::endl;
                for (auto &it_per_frame : it_per_id.tree_per_frame)
                {
                    oss << "    obs 3d pos " << it_per_frame.point.x() << " " << it_per_frame.point.y() << " "<< it_per_frame.point.z() << " normal " << it_per_frame.n.x() << " " << it_per_frame.n.y() << " " << it_per_frame.n.z() << " velocity " << it_per_frame.velocity.x() << " " << it_per_frame.velocity.y() << " " << it_per_frame.velocity.z()<< " td " << it_per_frame.cur_td << " frame " << it_per_frame.frame << std::endl;
                }
            }
            logMessage(oss.str());
            oss.str("");
            oss.clear();
            oss << "---------------------------------------------- position ------------------------------------------------------------" << std::endl;
            for(auto p : Ps){
                oss << p << std::endl << std::endl;
            }
            oss << "---------------------------------------------- orientation ------------------------------------------------------------" << std::endl;
            for(auto r : Rs){
                oss << r << std::endl << std::endl;
            } 
            logMessage(oss.str());
            ///// LOG /////
            // ROS_WARN("failure detection!");
            // failure_occur = 1;
            // clearState();
            // if(USE_TOPIC)
            //     setParameter_topic();
            // else
            //     setParameter();
            // ROS_WARN("system reboot!");
            // return;
        }

        /////// DEBUG //////////
        for(int i = 0; i <= WINDOW_SIZE; ++i){
            prev_deb_Ps[i] = Ps[i];
        }
        for(int i = 0; i <= WINDOW_SIZE; ++i){
            prev_deb_Vs[i] = Vs[i];
        }
        for(int i = 0; i <= WINDOW_SIZE; ++i){
            prev_deb_Rs[i] = Rs[i];
        }
        for(int i = 0; i <= WINDOW_SIZE; ++i){
            prev_deb_Bas[i] = Bas[i];
        }
        for(int i = 0; i <= WINDOW_SIZE; ++i){
            prev_deb_Bgs[i] = Bgs[i];
        }
        
        prev_deb_feature.clear();
        for(const auto& f : f_manager.feature){
            prev_deb_feature.push_back(f);
        }
        prev_deb_t_feature = f_manager.t_feature;
        /////// DEBUG //////////
        
        slideWindow(); // slide all the quantities in the window
        f_manager.removeFailures(); // delete features for which wasn't possible to do the optimization?

        rebuildLastModelForest();

        // prepare output of VINS
        key_poses.clear(); // clear all the keypose
        for (int i = 0; i <= WINDOW_SIZE; i++) // for all the element in the window
        {
            try{
                key_poses.push_back(Ps[i]); // add the new pose in the keypose buffer
            }
            catch(int err){
                std::cout << "keypose error " << err << std::endl;
            }
        }
            
        // update initial and finale orientation, position for next iteration 
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates(); // update state of the estimator
    } 
}
    


bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g; // the displacement between the second and last frame in all_image_frame
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) // for all the frames in all_image_frame, starting from the second one
        {
            double dt = frame_it->second.pre_integration->sum_dt; // get the delta time [related to what?]
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // get the delta velocity [related to what?] and devide it by the delta time, obtaining a displacement
            sum_g += tmp_g; // sum the displacemeent to obtain the displacement between the second and last frame in all_image_frame
        }
        Vector3d aver_g; // the average over the evaluated displacements
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); //evaluate the average over the evaluated displacements
        double var = 0; // the variance over the displacements
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); // get the variance over the displacements
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // finalize the variance evaluation
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25) // if the variance is too low
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature) // for all the features
    {
        int imu_j = it_per_id.start_frame - 1; // get the feature first appearence (in frame_count terms)
        SFMFeature tmp_feature;
        tmp_feature.state = false; // initialize the structure from motion datastructure
        tmp_feature.id = it_per_id.feature_id; // pass it the feature id
        for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the samples of the features (x, y, z, u, v, v_x, v_y)
        {
            imu_j++; // increase the imu counter (counting the number of appearences)
            Vector3d pts_j = it_per_frame.point; // get the undistorted position
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()})); // add the combination of counter and undistorted position of the feature in the datasrtuct
        }
        sfm_f.push_back(tmp_feature); // add the datastruct on the strucure from motion one 
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l)) // if the estimation of the transformation matrix is not good enough between a choosen frame in the window and the last one
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm; // initialize the structure from motion 
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points)) // create the instance of the structure from motion and solve the related bundle adjustment problem to obtain the pose of the camera
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++) // for all the frames
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) // iterate oveer the window_size
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // get all the features that has been present at least between frame i and WINDOW_SIZE, with their undistorted position at these istants
        if (corres.size() > 20) // if you obtain more then 20 features
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++) // for all the features in corres
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm(); // evaluate the position difference f the feature in the frames i and WINDOW_SIZE
                sum_parallax = sum_parallax + parallax; // increase the global parallax 

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size()); // average it over the number of features
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) // if the average parallax is sufficient between frame i and frame WINDOW_SIZE and the quality of the relative transformation matrix is good enough
            {
                l = i; // choose the frame i as the target frame together with the newest one to triangulate the whole structure 
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++) // for all the frame in the window convert the quantities from vectors to arrays
    {
        // cout << Ps[i].x() << " " << Ps[i].y() << " " << Ps[i].z() << endl;
        // cout << "--------" << endl;
        // position of base_link framer in world frame
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};

        // orientation (in quaternion factorization) of base_link frame in world frame
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {   
            // velocity
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();
            // speed bias
            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();
            // angular velocity bias
            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++) // for every camera
    {   
        // camera position in base link frame
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();

        // camera orientation in base link frame
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector(); // obtain the vector with one element per feature in which each element is equal to 1/depth of the feature
    for (int i = 0; i < f_manager.getFeatureCount(); i++) // for all the feature
    {   
        para_Feature[i][0] = dep(i); // save this depth in the para_Feature vector
    }
    if(USE_TREE)
    {
        VectorXd t_dep = f_manager.get_tree_DepthVector(); // obtain the vector with one element per feature in which each element is equal to 1/depth of the feature
        for (int i = 0; i < f_manager.get_tree_FeatureCount(); i++) // for all the tree feature
        {   
            para_tree_Features[i][0] = t_dep(i);; // save this depth in the para_tree_Feature vector
        }
    }
    para_Td[0][0] = td; // save the time delay in the relative variable
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++){
        dep(i) = para_Feature[i][0];
    }
    f_manager.setDepth(dep);

    if(USE_TREE)
    {
        VectorXd t_dep = f_manager.get_tree_DepthVector();
        for (int i = 0; i < f_manager.get_tree_FeatureCount(); i++){
            t_dep(i) = para_tree_Features[i][0];
        }
        f_manager.set_tree_Depth(t_dep);
    }
    
    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    //return false;
    if (f_manager.last_track_num < 2)
    {   
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
        return false;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        //return true;
        return false;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        //return true;
        return false;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {   
        ///// LOG /////
        std::ostringstream oss;
        oss << "JUMP " << std::setprecision(9) << Headers[frame_count] << std::endl;
        ROS_INFO(" jump");
        
        logMessage(oss.str());
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        //return true;
        return false;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::optimization()
{
    // Helper: log all model trees with the raw coordinates stored in the model
    // (TreePerFrame.point = camera-frame x/y/z as received, no projection applied).
    auto log_model_trees = [&](std::ostringstream &o, const char *label)
    {
        o << "=========================================================================\n"
          << label << "  optimization()  trees=" << f_manager.t_feature.size() << "\n";
        for (size_t ti = 0; ti < f_manager.t_feature.size(); ++ti)
        {
            const ModelTree &mt = f_manager.t_feature[ti];
            const int nv = (int)boost::num_vertices(mt);
            o << "  tree " << ti << "  nodes=" << nv
              << "  edges=" << boost::num_edges(mt) << "\n";
            for (int v = 0; v < nv; ++v)
            {
                const ModelNode &node = mt[v];
                o << "    node id=" << node.feature_id
                  << " depth=" << node.estimated_depth
                  << " obs=" << node.tree_per_frame.size();
                if (!node.tree_per_frame.empty())
                {
                    const Vector3d &pt = node.tree_per_frame.back().point;
                    o << " latest_frame=" << node.tree_per_frame.back().frame
                      << " point=(" << pt.x() << ", " << pt.y() << ", " << pt.z() << ")";
                }
                o << "\n";
            }
        }
    };

    {
        std::ostringstream oss;
        log_model_trees(oss, "PRE");
        logMessage(oss.str());
    }

    TicToc t_whole, t_prepare;
    vector2double(); // save all the needed quantities in the correct format

    ceres::Problem problem; // create an instance of a ceres problem
    
    //loss_function = NULL;
    ceres::LossFunction* image_loss = new ceres::HuberLoss(1.0);
    ceres::LossFunction* tree_loss_function = new ceres::HuberLoss(100.0);
    // tree_loss_function = new ceres::ScaledLoss(icp_loss, 1.0, ceres::TAKE_OWNERSHIP);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    //// LOG ////
    double avg_n_obs = 0;
    int count = 0;
    int deb_c = 0;
        for (auto &it_per_id : f_manager.feature) // for all the visual features
    {   
        deb_c += 1;
        int n_obs = 0;
        //std::cout << "start " << it_per_id.start_frame << " n obs " << it_per_id.feature_per_frame.size() << " condition " << (it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1 == frame_count) << std::endl;
        if((it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1 == frame_count) && (it_per_id.feature_per_frame.size() >= 4)){
            count++;
            avg_n_obs += (it_per_id.feature_per_frame.size() - avg_n_obs) / count;
        }
            
    }
    double avg_t_n_obs = 0;
    int t_count = 0;
    int t_deb_c = 0;
    for (auto &_mt : f_manager.t_feature)
    for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
    {
        auto &it_per_id = _mt[_v];
        t_deb_c += 1;
        if (it_per_id.has_frame(frame_count) && it_per_id.tree_per_frame.size() >= 3) {
            t_count++;
            avg_t_n_obs += (it_per_id.tree_per_frame.size() - avg_t_n_obs) / t_count;
        }
    }
    std::cout << "FEATURE: avg " << avg_n_obs << " n " << count << "/" << deb_c << "                      TREE: avg " << avg_t_n_obs << " n " << t_count << "/" << t_deb_c <<  std::endl;
    ///// LOG /////
    std::ostringstream oss2;
    oss2 << "=========================================================================\nE o forest tracking stats at time " << std::setprecision(15) << Headers[frame_count] << std::endl;
    oss2 << "FEATURE: avg " << avg_n_obs << " n " << count << "/" << deb_c << "\nTREE: avg " << avg_t_n_obs << " n " << t_count << "/" << t_deb_c <<  std::endl;
    logMessage(oss2.str());
    //// LOG ////

    ///////////////////////////////////////////// BALANCING ICP AND VISUAL RESIDUALS BY TUNING THE ICP FACTOR ////////////////////////////////////////////////

    // Evaluate visual feature residuals
    double visual_residual_sum_bef = 0.0;
    int visual_residual_count_bef = 0;

    int feature_index1 = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index1;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(
                    pts_i, pts_j, 
                    it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                
                // Evaluate residual
                double residual[2];
                double *parameters[5] = {
                    para_Pose[imu_i], 
                    para_Pose[imu_j], 
                    para_Ex_Pose[0], 
                    para_Feature[feature_index1], 
                    para_Td[0]
                };
                f_td->Evaluate(parameters, residual, nullptr);
                
                // Compute squared norm s = r? r
                double s = residual[0]*residual[0] +
                        residual[1]*residual[1];

                // Apply loss function
                double rho[2];
                image_loss->Evaluate(s, rho);

                // Final cost is 0.5 * rho[0]
                double cost = 0.5 * rho[0];

                // Accumulate (residual is already multiplied by sqrt_info in the factor)
                visual_residual_sum_bef += cost;
                visual_residual_count_bef += 1; // 2D residual
                
                delete f_td;
            }

            if(STEREO && it_per_frame.is_stereo) // in the case of stereovision
            {                
                Vector3d pts_j_right = it_per_frame.pointRight; // get the right observation of the feature repeat the same processing but including both the camera poses
                if(imu_i != imu_j) // if it appeared for more then one time
                {   
                    
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    
                    // Evaluate residual
                    double residual[2];
                    double *parameters[6] = {
                        para_Pose[imu_i], 
                        para_Pose[imu_j], 
                        para_Ex_Pose[0],
                        para_Ex_Pose[1], 
                        para_Feature[feature_index1], 
                        para_Td[0]
                    };
                    f->Evaluate(parameters, residual, nullptr);
                    // Compute squared norm s = r? r
                    double s = residual[0]*residual[0] +
                            residual[1]*residual[1];

                    // Apply loss function
                    double rho[2];
                    image_loss->Evaluate(s, rho);

                    // Final cost is 0.5 * rho[0]
                    double cost = 0.5 * rho[0];
                        
                    // Accumulate (residual is already multiplied by sqrt_info in the factor)
                    visual_residual_sum_bef += cost;
                    visual_residual_count_bef += 1; // 2D residual
                    
                    delete f;
                }
                else // handle the case in which the feature has been observed only in one frame (the right one)
                {   
                    
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    
                    // Evaluate residual
                    double residual[2];
                    double *parameters[4] = {
                        para_Ex_Pose[0],
                        para_Ex_Pose[1], 
                        para_Feature[feature_index1], 
                        para_Td[0]
                    };
                    f->Evaluate(parameters, residual, nullptr);
                    
                    // Compute squared norm s = r? r
                    double s = residual[0]*residual[0] +
                            residual[1]*residual[1];

                    // Apply loss function
                    double rho[2];
                    image_loss->Evaluate(s, rho);

                    // Final cost is 0.5 * rho[0]
                    double cost = 0.5 * rho[0];
                    // Accumulate (residual is already multiplied by sqrt_info in the factor)
                    visual_residual_sum_bef += cost;
                    visual_residual_count_bef += 1; // 2D residual
                    
                    delete f;
                }
            }
        }
    }

    // Evaluate IMU residuals
    double imu_residual_sum_bef = 0.0;
    int imu_residual_count_bef = 0;
    if(USE_IMU) 
    {
        for (int i = 0; i < frame_count; i++) // for every frame 
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0) // if the imu pre-integratio sum_dt is too big skip it
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]); // create a new IMU gactor
            // Evaluate residual
            double residual[15]; // IMU has 15 residuals (3 pos + 3 rot + 3 vel + 3 ba + 3 bg)
            double *parameters[4] = {
                para_Pose[i],       // pose_i (7 params)
                para_SpeedBias[i],  // speedbias_i (9 params)
                para_Pose[j],       // pose_j (7 params)
                para_SpeedBias[j]   // speedbias_j (9 params)
            };
            imu_factor->Evaluate(parameters, residual, nullptr);
            
            // Compute squared norm s = r^T * r
            double s = 0.0;
            for (int k = 0; k < 15; k++) {
                s += residual[k] * residual[k];
            }

            // Note: IMU factor has NULL loss function in your code
            // So the cost is just 0.5 * s (no robust loss applied)
            double cost = 0.5 * s;
            
            imu_residual_sum_bef += cost;
            imu_residual_count_bef += 1; // Count as 1 residual block (or use 15 if you want per-component)
            
            delete imu_factor;
        }
    }

    // Evaluate ICP residuals
    double icp_residual_sum_bef = 0.0;
    int icp_residual_count_bef = 0;

    if (USE_TREE)
    {
        int t_feature_index = -1;
        for (auto &_mt : f_manager.t_feature)
        for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
        {
            auto &it_per_id = _mt[_v];
            if (it_per_id.used_num < 3)
                continue;
            ++t_feature_index;

            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.tree_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.tree_per_frame)
            {
                if (imu_i != it_per_frame.frame)
                {
                    Vector3d pts_j = it_per_frame.point;

                    if(ICP_P2L)
                    {
                        Vector3d n_j = it_per_frame.n;
                        ceres::CostFunction* f_icp = ICPCostFunction_p2l::Create(
                            pts_i, it_per_id.tree_per_frame[0].velocity,
                            it_per_id.tree_per_frame[0].cur_td,
                            it_per_id.tree_per_frame[0].n,
                            pts_j, it_per_frame.velocity, it_per_frame.cur_td);

                        double residual[3];
                        double *parameters[5] = {
                            para_Pose[imu_i],
                            para_Pose[it_per_frame.frame],
                            para_Ex_Pose[0],
                            para_tree_Features[t_feature_index],
                            para_Td[0]
                        };
                        f_icp->Evaluate(parameters, residual, nullptr);
                        double s = residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2];
                        double rho[3];
                        tree_loss_function->Evaluate(s, rho);
                        icp_residual_sum_bef += 0.5 * rho[0];
                        icp_residual_count_bef += 1;
                        delete f_icp;
                    }
                    else
                    {
                        ceres::CostFunction* f_icp = ICPCostFunction_p2p::Create(
                            pts_i, it_per_id.tree_per_frame[0].velocity,
                            it_per_id.tree_per_frame[0].cur_td,
                            pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                        double residual[3];
                        double *parameters[5] = {
                            para_Pose[imu_i],
                            para_Pose[it_per_frame.frame],
                            para_Ex_Pose[0],
                            para_tree_Features[t_feature_index],
                            para_Td[0]
                        };
                        f_icp->Evaluate(parameters, residual, nullptr);
                        double s = residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2];
                        double rho[3];
                        tree_loss_function->Evaluate(s, rho);
                        icp_residual_sum_bef += 0.5 * rho[0];
                        icp_residual_count_bef += 1;
                        delete f_icp;
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////// END BALANCING INFO /////////////////////////////////////////////////////////////////////
    
    for (int i = 0; i < frame_count + 1; i++) // for every frame
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization(); // assign the local parametrization, used for the update in solving the problem
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // add a parameter block related to the frame pose
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); // add a parameter block related to the velocities (linear and angular) and IMU biases
    }
    if(!USE_IMU) // otherwise 
        problem.SetParameterBlockConstant(para_Pose[0]); // add a constant block (ceres will not change it in the optimization) with the pose of the first frame in the sliding window in the case of not using the IMU
    
    for (int i = 0; i < NUM_OF_CAM; i++) // for all the cameras (handle also the stereo case)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); // add a parameter block related to the pose fo the camera
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation) // in this case the extrinsic parameters has been estimated
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else // otherwise 
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); // add a a constant parameter block, meaning that the camera poses will not be optimized
        }
    }
    
    problem.AddParameterBlock(para_Td[0], 1); // add a parameter block related to the time offset
    
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2) // if we didn't estimated it or the initial velocity of the camera is too low
        problem.SetParameterBlockConstant(para_Td[0]); // set it constant
    
    if (last_marginalization_info && last_marginalization_info->valid) // if there are some valid marginalization info
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info); // create a new marginalization factor with the necessary informations
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks); // add a residual block related to the marginalization to the problem
    }

    if(USE_IMU) 
    {
        for (int i = 0; i < frame_count; i++) // for every frame 
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0) // if the imu pre-integratio sum_dt is too big skip it
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]); // create a new IMU gactor
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]); // add it to the problem in a residual block with the camera pose, the speed and IMU biases, the successive pose and successive speed and IMU biases
        }
    }
    
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nE opt adding features to optimization problem at time " << std::setprecision(15) << Headers[frame_count] << "\nnormal features: " << std::endl;

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature) // for all the visual features
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4) // if the feature has been observed less then 4 times skip it
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // get a counter from the first appearence 
        int n_obs_added = 0;
        int n_obs_added_s = 0;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point; // get the first observation image position of the feature
        for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observations of the feature
        {   
            imu_j++;
            if (imu_i != imu_j) // if the feature has been observed for more then once
            {   
                n_obs_added += 1;
                Vector3d pts_j = it_per_frame.point; // get the image position of the next appearence of the feature
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td); // initialize the factor for the projection of a feature observed in two frames
                problem.AddResidualBlock(f_td, image_loss, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]); // add a residual block with the factor to add, the loss function, the camera poses to optimize for those frames, the extrinsic parameter of the camera, the feature 3d position, the time offset 
            }

            if(STEREO && it_per_frame.is_stereo) // in the case of stereovision
            {                
                Vector3d pts_j_right = it_per_frame.pointRight; // get the right observation of the feature repeat the same processing but including both the camera poses
                if(imu_i != imu_j) // if it appeared for more then one time
                {   
                    n_obs_added_s += 1;
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, image_loss, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else // handle the case in which the feature has been observed only in one frame (the right one)
                {   
                    n_obs_added_s += 1;
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, image_loss, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++; // increment the counter of processed features
        }
        if((n_obs_added > 0) || (n_obs_added_s > 0)) 
        {
            oss << "    feature " << it_per_id.feature_id << " with " << n_obs_added << " observations and " << n_obs_added_s << " stereo observations " << std::endl;
        }
    }
    
    oss << "tree features:" << std::endl;
    int f_m_t_cnt = 0;
    int t_feature_index = -1;
    int curr_tree_features = 0;
    
    if (USE_TREE) 
    {   
        for (auto &_mt : f_manager.t_feature)
        for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
        {
            auto &it_per_id = _mt[_v];
            it_per_id.used_num = it_per_id.tree_per_frame.size();
            if (it_per_id.used_num < 3)
                continue;

            ++t_feature_index;

            int imu_i = it_per_id.start_frame;
            int n_obs_added = 0;
            Vector3d pts_i = it_per_id.tree_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.tree_per_frame)
            {
                if (imu_i != it_per_frame.frame)
                {
                    Vector3d pts_j = it_per_frame.point;
                    if(ICP_P2L)
                    {
                        Vector3d n_j = it_per_frame.n;
                        n_obs_added += 1;
                        ceres::CostFunction* f_icp = ICPCostFunction_p2l::Create(pts_i, it_per_id.tree_per_frame[0].velocity, it_per_id.tree_per_frame[0].cur_td, it_per_id.tree_per_frame[0].n, pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                        problem.AddResidualBlock(f_icp, tree_loss_function, para_Pose[imu_i], para_Pose[it_per_frame.frame], para_Ex_Pose[0], para_tree_Features[t_feature_index], para_Td[0]);
                    }
                    else
                    {
                        n_obs_added += 1;
                        ceres::CostFunction* f_icp = ICPCostFunction_p2p::Create(pts_i, it_per_id.tree_per_frame[0].velocity, it_per_id.tree_per_frame[0].cur_td, pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                        problem.AddResidualBlock(f_icp, tree_loss_function, para_Pose[imu_i], para_Pose[it_per_frame.frame], para_Ex_Pose[0], para_tree_Features[t_feature_index], para_Td[0]);
                    }
                }
                f_m_t_cnt++;
            }
            if (n_obs_added > 0)
                oss << "    tree feature " << it_per_id.feature_id << " with " << n_obs_added << " observations " << std::endl;
        }
    }
    
    logMessage(oss.str());
    ///// LOG /////

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options; // initialize the solveroptions

    if (USE_GPU_CERES)
        // std::cout << "1" << endl;
        options.dense_linear_algebra_library_type = ceres::CUDA; // set ceres to use cuda
    else
        // std::cout << "2" << endl;
        options.linear_solver_type = ceres::DENSE_SCHUR; // otherwise use different solver type

    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG; // solving parameter
    options.max_num_iterations = NUM_ITERATIONS; // set maximum number of iterations
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;

    // configure solver maximum runtime based on marginalization informations
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary; // structure to hold a summary of the problem solution
    ceres::Solve(options, &problem, &summary); // solve the problem
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    ///////////////////////////////////////////// BALANCING ICP AND VISUAL RESIDUALS BY TUNING THE ICP FACTOR ////////////////////////////////////////////////

    // Evaluate visual feature residuals
    double visual_residual_sum_aft = 0.0;
    int visual_residual_count_aft = 0;

    int feature_index2 = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index2;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(
                    pts_i, pts_j, 
                    it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                
                // Evaluate residual
                double residual[2];
                double *parameters[5] = {
                    para_Pose[imu_i], 
                    para_Pose[imu_j], 
                    para_Ex_Pose[0], 
                    para_Feature[feature_index2], 
                    para_Td[0]
                };
                f_td->Evaluate(parameters, residual, nullptr);
                
                // Compute squared norm s = r? r
                double s = residual[0]*residual[0] +
                        residual[1]*residual[1];

                // Apply loss function
                double rho[2];
                image_loss->Evaluate(s, rho);

                // Final cost is 0.5 * rho[0]
                double cost = 0.5 * rho[0];

                // Accumulate (residual is already multiplied by sqrt_info in the factor)
                visual_residual_sum_aft += cost;
                visual_residual_count_aft += 1; // 2D residual
                
                delete f_td;
            }

            if(STEREO && it_per_frame.is_stereo) // in the case of stereovision
            {                
                Vector3d pts_j_right = it_per_frame.pointRight; // get the right observation of the feature repeat the same processing but including both the camera poses
                if(imu_i != imu_j) // if it appeared for more then one time
                {   
                    
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    
                    // Evaluate residual
                    double residual[2];
                    double *parameters[6] = {
                        para_Pose[imu_i], 
                        para_Pose[imu_j], 
                        para_Ex_Pose[0],
                        para_Ex_Pose[1], 
                        para_Feature[feature_index2], 
                        para_Td[0]
                    };
                    f->Evaluate(parameters, residual, nullptr);
                    // Compute squared norm s = r? r
                    double s = residual[0]*residual[0] +
                            residual[1]*residual[1];

                    // Apply loss function
                    double rho[2];
                    image_loss->Evaluate(s, rho);

                    // Final cost is 0.5 * rho[0]
                    double cost = 0.5 * rho[0];
                        
                    // Accumulate (residual is already multiplied by sqrt_info in the factor)
                    visual_residual_sum_aft += cost;
                    visual_residual_count_aft += 1; // 2D residual
                    
                    delete f;
                }
                else // handle the case in which the feature has been observed only in one frame (the right one)
                {   
                    
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    
                    // Evaluate residual
                    double residual[2];
                    double *parameters[4] = {
                        para_Ex_Pose[0],
                        para_Ex_Pose[1], 
                        para_Feature[feature_index2], 
                        para_Td[0]
                    };
                    f->Evaluate(parameters, residual, nullptr);
                    
                    // Compute squared norm s = r? r
                    double s = residual[0]*residual[0] +
                            residual[1]*residual[1];

                    // Apply loss function
                    double rho[2];
                    image_loss->Evaluate(s, rho);

                    // Final cost is 0.5 * rho[0]
                    double cost = 0.5 * rho[0];
                    // Accumulate (residual is already multiplied by sqrt_info in the factor)
                    visual_residual_sum_aft += cost;
                    visual_residual_count_aft += 1; // 2D residual
                    
                    delete f;
                }
            }
        }
    }

    // Evaluate IMU residuals
    double imu_residual_sum_aft = 0.0;
    int imu_residual_count_aft = 0;
    if(USE_IMU) 
    {
        for (int i = 0; i < frame_count; i++) // for every frame 
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0) // if the imu pre-integratio sum_dt is too big skip it
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]); // create a new IMU gactor
            // Evaluate residual
            double residual[15]; // IMU has 15 residuals (3 pos + 3 rot + 3 vel + 3 ba + 3 bg)
            double *parameters[4] = {
                para_Pose[i],       // pose_i (7 params)
                para_SpeedBias[i],  // speedbias_i (9 params)
                para_Pose[j],       // pose_j (7 params)
                para_SpeedBias[j]   // speedbias_j (9 params)
            };
            imu_factor->Evaluate(parameters, residual, nullptr);
            
            // Compute squared norm s = r^T * r
            double s = 0.0;
            for (int k = 0; k < 15; k++) {
                s += residual[k] * residual[k];
            }

            // Note: IMU factor has NULL loss function in your code
            // So the cost is just 0.5 * s (no robust loss applied)
            double cost = 0.5 * s;
            
            imu_residual_sum_aft += cost;
            imu_residual_count_aft += 1; // Count as 1 residual block (or use 15 if you want per-component)
            
            delete imu_factor;
        }
    }

    // Evaluate ICP residuals
    double icp_residual_sum_aft = 0.0;
    int icp_residual_count_aft = 0;

    if (USE_TREE)
    {
        int t_feature_index = -1;
        for (auto &_mt : f_manager.t_feature)
        for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
        {
            auto &it_per_id = _mt[_v];
            if (it_per_id.used_num < 3)
                continue;
            ++t_feature_index;

            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.tree_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.tree_per_frame)
            {
                if (imu_i != it_per_frame.frame)
                {
                    Vector3d pts_j = it_per_frame.point;
                    if(ICP_P2L)
                    {
                        Vector3d n_j = it_per_frame.n;
                        ceres::CostFunction* f_icp = ICPCostFunction_p2l::Create(
                            pts_i, it_per_id.tree_per_frame[0].velocity,
                            it_per_id.tree_per_frame[0].cur_td,
                            it_per_id.tree_per_frame[0].n,
                            pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                        double residual[3];
                        double *parameters[5] = {
                            para_Pose[imu_i],
                            para_Pose[it_per_frame.frame],
                            para_Ex_Pose[0],
                            para_tree_Features[t_feature_index],
                            para_Td[0]
                        };
                        f_icp->Evaluate(parameters, residual, nullptr);
                        double s = residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2];
                        double rho[3];
                        tree_loss_function->Evaluate(s, rho);
                        icp_residual_sum_aft += 0.5 * rho[0];
                        icp_residual_count_aft += 1;
                        delete f_icp;
                    }
                    else
                    {
                        ceres::CostFunction* f_icp = ICPCostFunction_p2p::Create(
                            pts_i, it_per_id.tree_per_frame[0].velocity,
                            it_per_id.tree_per_frame[0].cur_td,
                            pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                        double residual[3];
                        double *parameters[5] = {
                            para_Pose[imu_i],
                            para_Pose[it_per_frame.frame],
                            para_Ex_Pose[0],
                            para_tree_Features[t_feature_index],
                            para_Td[0]
                        };
                        f_icp->Evaluate(parameters, residual, nullptr);
                        double s = residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2];
                        double rho[3];
                        tree_loss_function->Evaluate(s, rho);
                        icp_residual_sum_aft += 0.5 * rho[0];
                        icp_residual_count_aft += 1;
                        delete f_icp;
                    }
                }
            }
        }
    }
    ///// LOG /////
    std::ostringstream oss_r;
    oss_r << "=========================================================================\nE opt residuals at time " << std::setprecision(15) << Headers[frame_count] << std::endl;
    oss_r << "Visual residual before: " << visual_residual_sum_bef << " after: " << visual_residual_sum_aft << " (count: " << visual_residual_count_aft << ")" << std::endl;
    oss_r << "IMU residual before: " << imu_residual_sum_bef << " after: " << imu_residual_sum_aft << " (count: " << imu_residual_count_aft << ")" << std::endl;
    oss_r << "ICP residual before: " << icp_residual_sum_bef << " after: " << icp_residual_sum_aft << " (count: " << icp_residual_count_aft << ")" << std::endl;

    logMessage(oss_r.str());
    ///// LOG /////
    ////////////////////////////////////////////////////// END BALANCING INFO /////////////////////////////////////////////////////////////////////

    double2vector(); // reset the obtained results in the relative vectors
    //printf("frame_count: %d \n", frame_count);
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) // if it is a keyframe
    {
        //std::cout << "keyframe " <<std::endl;
        MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // initialize the marginalization factors
        vector2double(); // reconvert the optimized data from vectors to arrays

        if (last_marginalization_info && last_marginalization_info->valid) // if you have previous informations about the marginalization, and they are valid
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) // for all the parameters of the last marginalization
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) // if the relative parameter is equal to the first pose of the window size or it is equal to the velocities and IMU biasses
                    drop_set.push_back(i); // add the index i of the last marginalization block to the droup out
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set); // create the marginalization residual block
            marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to the marginalization problem
        }

        if(USE_IMU) 
        {
            if (pre_integrations[1]->sum_dt < 10.0) // if the sum_dt (time required for the evaluation of the residual summed ofver all the features) is less then 10 
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]); // create a new factor
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); //create the IMU residual block
                marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to the marginalization problem
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature) // for all the features
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size(); // set as used number the number of appearence of the feature
                if (it_per_id.used_num < 4) // if it's smaller then 4 skip the feature
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // get the index for the first and second appearence of the feature
                if (imu_i != 0) // ? if it didn't appear at the frame_count 0 skip the feature?
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point; // get the 3d position of the feature in its first appearence

                for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the appearence of the feature
                {
                    imu_j++;
                    if(imu_i != imu_j) // if the feature has been observed for more then once
                    {
                        Vector3d pts_j = it_per_frame.point; // get its new 3d position
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td); // set the factor of the residual block
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, image_loss,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3}); // create the residual block
                        marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to the marginalization problem
                    }
                    if(STEREO && it_per_frame.is_stereo) // if we are in the stereo case, and we have the right observation of the feature
                    {   
                        Vector3d pts_j_right = it_per_frame.pointRight; // get the right observation
                        if(imu_i != imu_j) // if the feature has been observed for more then once
                        {   
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td); // create the factor of the residual block
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, image_loss,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4}); // create the residual block
                            marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to the marginalization problem
                        }
                        else // if it has been observed only once
                        {   
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td); // create the factor of the residual block (same as above)
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, image_loss,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2}); // create the residual block (optimizing only the position of the feature and the extrinsic parameters)
                            marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to the marginalization problem
                        }
                    }
                }
            }
            
            if (USE_TREE)
            {
                int t_feature_index = -1;
                for (auto &_mt : f_manager.t_feature)
                for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
                {
                    auto &it_per_id = _mt[_v];
                    it_per_id.used_num = it_per_id.tree_per_frame.size();
                    if (it_per_id.used_num < 3)
                        continue;

                    ++t_feature_index;

                    int imu_i = it_per_id.start_frame;
                    if (imu_i != 0)
                        continue;

                    Vector3d pts_i = it_per_id.tree_per_frame[0].point;

                    for (auto &it_per_frame : it_per_id.tree_per_frame)
                    {
                        if (imu_i != it_per_frame.frame)
                        {
                            Vector3d pts_j = it_per_frame.point;
                            if(ICP_P2L)
                            {
                                Vector3d n_j = it_per_frame.n;
                                ceres::CostFunction* f_icp = ICPCostFunction_p2l::Create(pts_i, it_per_id.tree_per_frame[0].velocity, it_per_id.tree_per_frame[0].cur_td, it_per_id.tree_per_frame[0].n, pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_icp, tree_loss_function,
                                    vector<double *>{para_Pose[imu_i], para_Pose[it_per_frame.frame], para_Ex_Pose[0], para_tree_Features[t_feature_index], para_Td[0]},
                                    vector<int>{0, 3});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                            else
                            {
                                ceres::CostFunction* f_icp = ICPCostFunction_p2p::Create(pts_i, it_per_id.tree_per_frame[0].velocity, it_per_id.tree_per_frame[0].cur_td, pts_j, it_per_frame.velocity, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_icp, tree_loss_function,
                                    vector<double *>{para_Pose[imu_i], para_Pose[it_per_frame.frame], para_Ex_Pose[0], para_tree_Features[t_feature_index], para_Td[0]},
                                    vector<int>{0, 3});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize(); // operate a pre marginalization
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize(); // operate the marginalization
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
        
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) // for all the element in the window
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; // associate the previous parameter block related to the position to the current one
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1]; // associate the previous parameter block related to the velocities and IMU biases to the current one
        }
        for (int i = 0; i < NUM_OF_CAM; i++) // cover both the case of monocular and stereocameras
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i]; // keep constant the parameter block without linking to the previous ones

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0]; // keep constant the parameter block without linking to the previous ones
        
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift); // get the parameter for which we performed the marginalization 

        // update marginalization informations
        if (last_marginalization_info)
        {
            delete last_marginalization_info;
            last_marginalization_info = nullptr;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else // if it's not a keyframe
    {   
        
        //std::cout << "no key frame" << std::endl;
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) // if we have marginalization info from the last run and the last pose of the window size is among the parameter block
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // create a new marginalization problem
            vector2double(); // convert the information from vector to the correct arrays
            if (last_marginalization_info && last_marginalization_info->valid) // if there are marginalization information from the previous run and they are valid
            {
                vector<int> drop_set; 
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) // for all the last marginalization parameters
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]); // check that they are different from the last velocities and IMU biasses in the window
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1]) // if they are equal to the last pose in the window
                        drop_set.push_back(i); // add them to the drop set
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set); // create the residual block

                marginalization_info->addResidualBlockInfo(residual_block_info); // add the residual block to marginalization problem
            }
            
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize(); // operate the pre marginalization
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize(); // operate the marginalization
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++) // for all the elements in the slinding window
            {
                if (i == WINDOW_SIZE - 1) // if it's the second to last skip it
                    continue;
                else if (i == WINDOW_SIZE) // if it is the last element of the window size
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; // the last pose is shifted to map to the previous parameters
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1]; // the last velocities and IMU biasses is shifted to map to the previous parameters
                }
                else 
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i]; // assign each block to itself (not modified over the marginalization)
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i]; // assign each block to itself (not modified over the marginalization)
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++) // for all the cameras (handling also the stereo case)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i]; // assign the extrinsic parameter block to itself (not modified over the marginalization)

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0]; // assign the time offset block to itself (not modified over the marginalization)

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift); // get the marginalization info
            // update marginalization informations
            if (last_marginalization_info)
            {
                delete last_marginalization_info;
                last_marginalization_info = nullptr;
            }
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());

    {
        std::ostringstream oss;
        log_model_trees(oss, "POST");
        logMessage(oss.str());
    }
}

void Estimator::slideWindow()
{
    
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) // if the last frame was a keyframe
    {   
        double t_0 = Headers[0]; // get the time of the first element in the window
        back_R0 = Rs[0]; // get the orientation of the first element in the window
        back_P0 = Ps[0]; // get the position of the first element in the window
        if (frame_count == WINDOW_SIZE) // if frame count is equal to the winwod size
        {
            for (int i = 0; i < WINDOW_SIZE; i++) // for all the element of the window
            {
                Headers[i] = Headers[i + 1]; // slide the header
                Rs[i].swap(Rs[i + 1]); // slide the oprientation
                Ps[i].swap(Ps[i + 1]); // slide the position
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]); // slide the preintegration

                    dt_buf[i].swap(dt_buf[i + 1]); // slide the time differences
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]); // slide the linear accellerations
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]); // slide the angular velocities

                    Vs[i].swap(Vs[i + 1]); // slide the frame general velocity
                    Bas[i].swap(Bas[i + 1]); // slide the accellerometer biasses
                    Bgs[i].swap(Bgs[i + 1]); // slide the gyroscope biasses
                }
            }
            // handle the last element of the header, position and orientation differently
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1]; 
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1]; 
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            // same for IMU velocities, accellerometer biasses,  gyroscope biasses, preintegrations, delta times, linear accelleration, angular accelleration
            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = nullptr;
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL) // if we are at the initial case
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0); // find the header in the all_image_frame list
                delete it_0->second.pre_integration; // erase its pre_integration
                it_0->second.pre_integration = nullptr; // assign instead a null pointer
                all_image_frame.erase(all_image_frame.begin(), it_0); // remove from the container the element corresponding to the first frame specified by it_0
            }
            slideWindowOld(); // modify the feature removing oservations and features that went out of the window with the sliding
        }
    }
    else // if it wasn't a keyframe
    {
        if (frame_count == WINDOW_SIZE) // if the frame count is equal to the size of the window
        {   
            // slide the last element of header, position and orientatiion
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) // for all the preintegration interals
                {
                    double tmp_dt = dt_buf[frame_count][i]; // get the time interval
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i]; // get the linear accelleration
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i]; // get the angular velocity

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity); // put them at the end of the pre_integration list

                    dt_buf[frame_count - 1].push_back(tmp_dt); // slide the time interval
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration); // slide the linear accellerations
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity); // slide the angular velocities
                }
                // slide velocities and IMU biasses
                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE]; // delete the last pre_integration
                pre_integrations[WINDOW_SIZE] = nullptr; // assign it a null pointer
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]}; // and finally fill it with the value obtained in the last iteration

                dt_buf[WINDOW_SIZE].clear(); // delete the last time interval
                linear_acceleration_buf[WINDOW_SIZE].clear(); // delete the last linear accelleration
                angular_velocity_buf[WINDOW_SIZE].clear(); // delete the last angular velocity
            }
            slideWindowNew(); // modify the feature incrementing the start frame and removing the front feature?
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count); // remove the front feature?
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false; // if we are not in initialization
    if (shift_depth)
    {   
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0]; // evaluate the new orientation of the old first frame (i guess?)
        R1 = Rs[0] * ric[0]; // evaluate the new orientation of the new first frame (i guess?)
        P0 = back_P0 + back_R0 * tic[0]; // evaluate the new position of the old first frame (i guess?)
        P1 = Ps[0] + Rs[0] * tic[0]; // evaluate the new position of the new first frame (i guess?)
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); // decrease all last feature start frame and remove the last one checking for the depth of the last feature and delete possible observations of the feature before the window ?
    }
    else
    {
        f_manager.removeBack(); // decrease the start frame for all the frame and delete possible observations of the feature before the window ?
        
    }
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT); // get current pose?
    getPoseInWorldFrame(frame_count - 1, prevT); // get the pose of the last frame
    nextT = curT * (prevT.inverse() * curT); //evaluate the next pose under a linear velocity assumption
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature) // for all the features
    {
        if(it_per_id.estimated_depth > 0) // if the depth has been estimated
        {
            int firstIndex = it_per_id.start_frame; // get their first appearence
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1; // get their last appearence
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) // if the feature has been observed for more then 2 times, and the last time was in the current frame_count
            {
                double depth = it_per_id.estimated_depth; //get the depth
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0]; // obtain the point position in the first appearence transformed in the right frame (which one?)
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex]; // get its orientation
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3)); // get its position in the enxt frame in one reference frame
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]); // get it in the camera reference frame
                int ptsIndex = it_per_id.feature_id; // get it id
                predictPts[ptsIndex] = pts_cam; // add the predicted point to the list
            }
        }
    }
    featureTracker.setPrediction(predictPts); // set the prediction in the feature tracker, where it will be used
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

double Estimator::icp_p2p_error(Matrix3d &Ri, Vector3d &Pi, Matrix3d &ric, Vector3d &tic,
                            Matrix3d &Rj, Vector3d &Pj, double depth, Vector3d &pts_i, Vector3d &pts_j) // evaluate the ICP error Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], pts_i, pts_j
{
    // evaluate icp error
    // get the anchor point position using the estimated depth
    Eigen::Vector3d est_pts_i = pts_i.normalized() * depth;

    // Transform to IMU frame
    Eigen::Vector3d pts_imu_i = ric * est_pts_i + tic;
    Eigen::Vector3d pts_imu_j = ric * pts_j + tic;

    // Transform to world frame
    Eigen::Vector3d pts_w_i = Ri * pts_imu_i + Pi;
    Eigen::Vector3d pts_w_j = Rj * pts_imu_j + Pj;

    // Compute residuals
    Eigen::Vector3d r = pts_w_i - pts_w_j;

    return r.norm();
}

double Estimator::icp_p2l_error(Matrix3d &Ri, Vector3d &Pi, Matrix3d &ric, Vector3d &tic,
                            Matrix3d &Rj, Vector3d &Pj, double depth, Vector3d &pts_i, Vector3d &normal_i, Vector3d &pts_j) // evaluate the ICP error Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], pts_i, pts_j
{
    // evaluate icp error
    // get the anchor point position using the estimated depth
    Eigen::Vector3d est_pts_i = pts_i.normalized() * depth;

    // project into body (imu) frame at time i
    Eigen::Vector3d pts_imu_i = ric * est_pts_i + tic;
    // project in world frame at time i
    Eigen::Vector3d pts_w_i = Ri * pts_imu_i + Pi;

    // evaluate inverces
    Eigen::Matrix3d R_bw_j = Rj.transpose();
    Eigen::Vector3d t_bw_j = -R_bw_j * Pj;

    Eigen::Matrix3d R_cb = ric.transpose();
    Eigen::Vector3d t_cb = -R_cb * tic;

    // project point i in body (imu) j
    Eigen::Vector3d pts_i_bj = R_bw_j * pts_w_i + t_bw_j;

    // project point i in camera frame at time j
    Eigen::Vector3d pts_i_cj = R_cb * pts_i_bj + t_cb;

    // reproject normal
    Eigen::Vector3d n_bi = ric * normal_i + tic;
    Eigen::Vector3d n_wi = Ri * n_bi + Pi;
    Eigen::Vector3d n_bj = R_bw_j * n_wi + t_bw_j; 
    Eigen::Vector3d n_cj = R_cb * n_bj + t_cb;

    Eigen::Vector3d diff = pts_j - pts_i_cj;

    double e0 = diff.dot(n_cj);
    Eigen::Vector3d e0_v = e0 * n_cj;
    Eigen::Vector3d proj_diff = diff - e0_v;

    return proj_diff.norm();
}

void Estimator::outliersRejection(set<int> &removeIndex, set<int> &remove_tree_Index)
{   
    ///// LOG /////
    std::ostringstream oss;
    oss << "=========================================================================\nE or selecting outliers at time " << std::setprecision(15) << Headers[frame_count] << "\nnormal features: " << std::endl;

    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature) // for all the feature
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // set as used number the number of observations of the feature
        if (it_per_id.used_num < 4) // if it's less then 4 skip it
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // get the feature count of the first appearence of the feature and the second one
        Vector3d pts_i = it_per_id.feature_per_frame[0].point; // get the 3D position of the feature in its first appearence
        double depth = it_per_id.estimated_depth; // get its depth
        for (auto &it_per_frame : it_per_id.feature_per_frame) // for all the observation of the feature
        {
            imu_j++;
            if (imu_i != imu_j) // if it has been observed for more then once
            {
                Vector3d pts_j = it_per_frame.point; // get the new observation 3D position of the feature
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j); // evaluate the reprojection error
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo) // if we are in the stereo case and we have a right camera correspondence
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight; // get the 3D position of the feature from the right camera
                if(imu_i != imu_j) // if it has been observed for more than once
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right); // evaluate the reprojection error between last appearence and right appearence 
                    err += tmp_error;
                    errCnt++; 
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else 
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }

        double ave_err = err / errCnt; // evaluate the average reprojection error for the given feature over its observations
        if(ave_err * FOCAL_LENGTH > 3){ // if this condition hold
            removeIndex.insert(it_per_id.feature_id); // add the feature among the one to remove
            oss << "    adding " << it_per_id.feature_id << std::endl;
        }
        

    }
    oss << "tree features:" << std::endl;
    if (USE_TREE)
    {   
        for (auto &_mt : f_manager.t_feature)
        for (int _v = 0; _v < (int)boost::num_vertices(_mt); ++_v)
        {
            auto &it_per_id = _mt[_v];
            double t_err = 0;
            int t_errCnt = 0;

            it_per_id.used_num = it_per_id.tree_per_frame.size();
            if (it_per_id.used_num < 3)
                continue;

            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.tree_per_frame[0].point;
            double depth = it_per_id.estimated_depth;
            for (auto &it_per_frame : it_per_id.tree_per_frame)
            {
                if (imu_i != it_per_frame.frame)
                {
                    double tmp_error;
                    Vector3d pts_j = it_per_frame.point;
                    if(ICP_P2L)
                    {
                        Vector3d n_i = it_per_id.tree_per_frame[0].n;
                        tmp_error = icp_p2l_error(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                  Rs[it_per_frame.frame], Ps[it_per_frame.frame], depth, pts_i, n_i, pts_j);
                    }
                    else
                    {
                        tmp_error = icp_p2p_error(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                  Rs[it_per_frame.frame], Ps[it_per_frame.frame], depth, pts_i, pts_j);
                    }
                    t_err += tmp_error;
                    t_errCnt++;
                }
            }

            double ave_err = t_err / t_errCnt;
            if (ave_err > TREE_OUTLIERS_TRESH) {
                remove_tree_Index.insert(it_per_id.feature_id);
                oss << "    adding " << it_per_id.feature_id << std::endl;
            }
        }
    }
    logMessage(oss.str());
    ///// LOG /////
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{   
    mPropagate.lock(); // lock the threads
    // update the state of the estimator
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    // get last measurements of accellerations and angular velocities
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr); // predict new IMU readings
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
