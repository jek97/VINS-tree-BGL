void Estimator::processMeasurements()
{
    while (1)
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