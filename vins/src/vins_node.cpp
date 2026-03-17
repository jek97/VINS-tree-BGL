#include "vins_node.h"

using namespace std::chrono_literals;
using namespace message_filters;

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;

Estimator estimator;

queue<sensor_msgs::msg::Imu::ConstPtr> imu_buf;
queue<sensor_msgs::msg::PointCloud::ConstPtr> feature_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img0_buf;
queue<sensor_msgs::msg::Image::ConstPtr> img1_buf;
std::mutex m_buf;

struct DetectionBox {
    cv::Rect box;
    float confidence;
    int classId;
    std::vector<float> maskCoeffs;
};

class TensorRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

struct TensorRTRuntime {
    TensorRTLogger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
            
    // GPU buffers (must persist)
    void* d_input = nullptr;
    void* d_output = nullptr;
    void* d_protos = nullptr;
    cudaStream_t stream;
            
    int inputIndex = -1;
    int outputIndex = -1;
    int protosIndex = -1;
            
    size_t inputSize;
    size_t outputSize;
    size_t protosSize;
            
    int numClasses;
            
    ~TensorRTRuntime() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_protos) cudaFree(d_protos);
        if (stream) cudaStreamDestroy(stream);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};

struct ONNXRuntime {
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    int numClasses;
            
    ONNXRuntime() : env(ORT_LOGGING_LEVEL_WARNING, "YOLO11_ONNX") {}
            
    ~ONNXRuntime() {
        for (auto name : inputNames) free(const_cast<char*>(name));
        for (auto name : outputNames) free(const_cast<char*>(name));
    }

};
// define publishers

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_color_skels;
rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_color_skels_info;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_tree_match;
rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_tree_match_info;

double old_header = 0;

class VinsNode : public rclcpp::Node, public std::enable_shared_from_this<VinsNode>
{   
    public:
        VinsNode() : Node("vins_estimator")
        {   
            // declare parameters
            this->declare_parameter("queue_size", 50); // sync queue size
            this->declare_parameter("delay_threshold", 0.01); // sync delay threshold
            this->declare_parameter("vins_config_file", " "); // vins config file
            
            // get parameters
            int64_t queue_size = this->get_parameter("queue_size").as_int();
            double_t delay_threshold = this->get_parameter("delay_threshold").as_double();
            string config_file = this->get_parameter("vins_config_file").as_string();
            
            cam0_info_ft = true;
            cam1_info_ft = true;
            cam_tree_info_ft = true;
            cam_info_set_flag = 0; 

            struct stat buffer;
            if(!stat (config_file.c_str(), &buffer) == 0)
            {
                printf("config file doesn't exists \n"
                "please intput: rosrun vins vins_node [config file] \n"
                "for example: rosrun vins vins_node "
                "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
            }

            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
            T_lcam_tree.setIdentity(); 
            tree_lcam_flag = false; 
            tree_baselink_flag= false; 

            printf("config_file: %s\n", config_file.c_str());
            readParameters(config_file);
            if(!USE_TOPIC)
            {
                estimator.setParameter();
            }
            if(USE_TREE)
            {
                loadYOLOModel();
            }

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
            ROS_WARN("waiting for image and imu...");
            
            // create subscribers:
            rclcpp::QoS tree_qos = rclcpp::QoS(100)
                .reliability(rclcpp::ReliabilityPolicy::Reliable)
                .durability(rclcpp::DurabilityPolicy::TransientLocal)
                .history(rclcpp::HistoryPolicy::KeepLast);
            
            rclcpp::QoS camera_info_qos = rclcpp::QoS(1)  // depth = 1
                .reliability(rclcpp::ReliabilityPolicy::Reliable)
                .durability(rclcpp::DurabilityPolicy::Volatile)
                .history(rclcpp::HistoryPolicy::KeepLast);

            rclcpp::QoS image_qos = rclcpp::QoS(10)  // depth = 10
                .reliability(rclcpp::ReliabilityPolicy::Reliable)
                .durability(rclcpp::DurabilityPolicy::Volatile)
                .history(rclcpp::HistoryPolicy::KeepLast);
            
            rclcpp::QoS imu_qos = rclcpp::QoS(10)  // depth = 10
                .reliability(rclcpp::ReliabilityPolicy::Reliable)
                .durability(rclcpp::DurabilityPolicy::Volatile)
                .history(rclcpp::HistoryPolicy::KeepLast)
                .keep_last(10);
            
            sub_feature = this->create_subscription<sensor_msgs::msg::PointCloud>("/feature_tracker/feature", tree_qos, std::bind(&VinsNode::feature_callback, this, _1));
            
            if(USE_TREE)
            {   
                if(STEREO)
                {
                    sub_img0.subscribe(this, IMAGE0_TOPIC, image_qos.get_rmw_qos_profile()); 
                    sub_img1.subscribe(this, IMAGE1_TOPIC, image_qos.get_rmw_qos_profile()); 
                    sub_imgc.subscribe(this, TREE_COLOR_TOPIC, image_qos.get_rmw_qos_profile()); 
                    sub_imgd.subscribe(this, TREE_DEPTH_TOPIC, image_qos.get_rmw_qos_profile()); 
                    stereo_img_tree_sync = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
                        ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(
                        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>(queue_size), sub_img0, sub_img1, sub_imgc, sub_imgd);

                    stereo_img_tree_sync->setAgePenalty(delay_threshold);
                    stereo_img_tree_sync->registerCallback(std::bind(&VinsNode::stereo_img_tree_sync_cb, this, _1, _2, _3, _4));

                    if(USE_TOPIC){
                        sub_img0_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE0_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img0_info_callback, this, _1));
                        sub_img1_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE1_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img1_info_callback, this, _1));
                        sub_tree_imgd_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(TREE_DEPTH_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::tree_imgd_info_callback, this, _1));
                    }
                }
                else
                {   
                    sub_img0_only_tr.subscribe(this, IMAGE0_TOPIC, image_qos.get_rmw_qos_profile()); 
                    sub_imgc.subscribe(this, TREE_COLOR_TOPIC, image_qos.get_rmw_qos_profile()); 
                    sub_imgd.subscribe(this, TREE_DEPTH_TOPIC, image_qos.get_rmw_qos_profile()); 
                    stereo_img_mono_tree_sync = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
                        ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(
                        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                        sensor_msgs::msg::Image, sensor_msgs::msg::Image>(queue_size), sub_img0_only_tr, sub_imgc, sub_imgd);
                    
                    stereo_img_mono_tree_sync->setAgePenalty(delay_threshold);
                    stereo_img_mono_tree_sync->registerCallback(std::bind(&VinsNode::stereo_img_mono_tree_sync_cb, this, _1, _2, _3));
                    
                    if(USE_TOPIC){
                        sub_img0_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE0_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img0_info_callback, this, _1));
                        sub_tree_imgd_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(TREE_DEPTH_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::tree_imgd_info_callback, this, _1));
                    }
                }
            }
            
            if(USE_IMU)
            {
                sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, imu_qos, std::bind(&VinsNode::imu_callback, this, _1));
            }
            
            if(STEREO && !USE_TREE)
            {   
                sub_img0.subscribe(this, IMAGE0_TOPIC, image_qos.get_rmw_qos_profile()); 
                sub_img1.subscribe(this, IMAGE1_TOPIC, image_qos.get_rmw_qos_profile()); 
                stereo_img_sync = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
                    ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(
                    message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                    sensor_msgs::msg::Image>(queue_size), sub_img0, sub_img1);

                stereo_img_sync->setAgePenalty(delay_threshold);
                stereo_img_sync->registerCallback(std::bind(&VinsNode::stereo_img_sync_cb, this, _1, _2));

                if(USE_TOPIC)
                {   
                    sub_img0_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE0_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img0_info_callback, this, _1));
                    sub_img1_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE1_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img1_info_callback, this, _1));
                }
            }
            else if(!STEREO && !USE_TREE)
            {
                auto sub_img0_only = this->create_subscription<sensor_msgs::msg::Image>(IMAGE0_TOPIC, image_qos, std::bind(&VinsNode::img0_only_callback, this, _1));
                
                if(USE_TOPIC)
                {   
                    sub_img0_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(IMAGE0_INFO_TOPIC, camera_info_qos, std::bind(&VinsNode::img0_info_callback, this, _1));
                }
            }       
            
            auto sub_restart = this->create_subscription<std_msgs::msg::Bool>("/vins_restart", camera_info_qos, std::bind(&VinsNode::restart_callback, this, _1));
            auto sub_imu_switch = this->create_subscription<std_msgs::msg::Bool>("/vins_imu_switch", camera_info_qos, std::bind(&VinsNode::imu_switch_callback, this, _1));
            auto sub_cam_switch = this->create_subscription<std_msgs::msg::Bool>("/vins_cam_switch", camera_info_qos, std::bind(&VinsNode::cam_switch_callback, this, _1));

            // create publishers
            pub_image_track = this->create_publisher<sensor_msgs::msg::Image>("/image_track", 1000);
            pub_latest_odometry = this->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 1000);
            pub_path = this->create_publisher<nav_msgs::msg::Path>("path", 1000);
            pub_odometry = this->create_publisher<nav_msgs::msg::Odometry>("vins_odometry", 1000);
            pub_point_cloud = this->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 1000);
            pub_margin_cloud = this->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud", 1000);
            pub_key_poses = this->create_publisher<visualization_msgs::msg::Marker>("key_poses", 1000);
            pub_camera_pose = this->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 1000);
            pub_camera_pose_visual = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
            pub_keyframe_pose = this->create_publisher<nav_msgs::msg::Odometry>("keyframe_pose", 1000);
            pub_keyframe_point = this->create_publisher<sensor_msgs::msg::PointCloud>("keyframe_point", 1000);
            pub_extrinsic = this->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 1000);
            
            if (USE_TREE)
            {
                pub_color_skels = this->create_publisher<sensor_msgs::msg::Image>("/color_skeletons/image", 1000); // to remove only for debug
                pub_color_skels_info = this->create_publisher<sensor_msgs::msg::CameraInfo>("/color_skeletons/camera_info", 1000); // to remove only for debug

                pub_tree_match = this->create_publisher<sensor_msgs::msg::Image>("tree_match/image", 1000);
                pub_tree_match_info = this->create_publisher<sensor_msgs::msg::CameraInfo>("tree_match/camera_info", 1000);
            }

            // launch processing threads
            image_thread = std::thread(&VinsNode::image_processing, this);
            if(USE_TREE)
            {
                tree_thread = std::thread(&VinsNode::tree_processing, this);
            }
        }

        ~VinsNode()
        {
            // destructor
            if (image_thread.joinable())
                image_thread.join();
            
            if (tree_thread.joinable())
                tree_thread.join();
        }

    private:
        int msg_count = 40; // counter to ignore the first messages that are noisy 
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        Eigen::Matrix3d R_bodylink_tree;
        cv::Mat prev_depth_;
        std::mutex prev_depth_mutex_;  
        ONNXRuntime onnx_;
        TensorRTRuntime trt_;
        

        // subscriber definition:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr sub_feature;

        // tree case
        // stereo
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_imgc; 
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_imgd;
        std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> stereo_img_tree_sync;
        // mono
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_img0_only_tr;
        std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> stereo_img_mono_tree_sync;
        
        // use imu case
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
        Eigen::Vector3d acc_old, gyr_old;
        int acc_filter_f = true;

        // stereo case
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_img0;
        message_filters::Subscriber<sensor_msgs::msg::Image> sub_img1;
        std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> stereo_img_sync;

        // mono case
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_only = NULL;

        // use topic case
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_img0_info;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_img1_info;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_tree_imgd_info;
        bool cam0_info_ft;
        bool cam1_info_ft;
        bool cam_tree_info_ft;
        std::atomic<int> cam_info_set_flag; 

        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_restart;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_imu_switch;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_cam_switch;  

        // mutlithread case
        std::queue<std::pair<double, std::pair<cv::Mat, cv::Mat>>> images_input_buf;
        std::queue<std::pair<double, std::pair<cv::Mat, cv::Mat>>> tree_input_buf;
        std::mutex m_image_buf;
        std::mutex m_tree_buf;
        std::thread image_thread;
        std::thread tree_thread;
        double last_msg_t = 0;

        void image_processing()
        {
            while(rclcpp::ok())
            {
                if(!images_input_buf.empty())
                {
                    cv::Mat image0, image1;
                    double time = 0;
                    bool flag = false;

                    // extract data
                    if(USE_TOPIC)
                    {
                        if(USE_TREE)
                        {
                            if((NUM_OF_CAM + 1 - cam_info_set_flag) == 0)
                            {   
                                flag = true;
                                {
                                    std::lock_guard<std::mutex> lock(m_image_buf);
                                    auto data = images_input_buf.front();
                                    images_input_buf.pop();

                                    time = data.first;
                                    image0 = data.second.first;
                                    image1 = data.second.second;

                                }
                            }
                        }
                        else
                        {
                            if(((NUM_OF_CAM - cam_info_set_flag) == 0))
                            {   
                                flag = true;
                                {
                                    std::lock_guard<std::mutex> lock(m_image_buf);
                                    auto data = images_input_buf.front();
                                    images_input_buf.pop();

                                    time = data.first;
                                    image0 = data.second.first;
                                    image1 = data.second.second;

                                }
                            }
                        }
                    }
                    else
                    {
                        flag = true;
                        {
                            std::lock_guard<std::mutex> lock(m_image_buf);
                            auto data = images_input_buf.front();
                            images_input_buf.pop();

                            time = data.first;
                            image0 = data.second.first;
                            image1 = data.second.second;

                        }
                    }
                    
                    // if you get data process them
                    if(flag)
                    {
                        if(image1.empty())
                        {
                            estimator.inputImage(time, image0);
                        }
                        else
                        {
                            estimator.inputImage(time, image0, image1);
                        }
                    }

                }
                std::chrono::milliseconds dura(2);
                std::this_thread::sleep_for(dura);
            }
        }

        void tree_processing()
        {
            while(rclcpp::ok())
            {
                if(!tree_input_buf.empty())
                {
                    cv::Mat color_image, depth_image;
                    double time = 0;
                    bool flag = false;

                    // extract data
                    if(USE_TOPIC)
                    {   
                        if((NUM_OF_CAM + 1 - cam_info_set_flag) == 0)
                        {   
                            flag = true;
                            {
                                std::lock_guard<std::mutex> lock(m_tree_buf);
                                auto data = tree_input_buf.front();
                                tree_input_buf.pop();

                                time = data.first;
                                color_image = data.second.first;
                                depth_image = data.second.second;
                            }
                        }
                    }
                    else
                    {
                        flag = true;
                        {
                            std::lock_guard<std::mutex> lock(m_tree_buf);
                            auto data = tree_input_buf.front();
                            tree_input_buf.pop();

                            time = data.first;
                            color_image = data.second.first;
                            depth_image = data.second.second;
                        }
                    }

                    // process data
                    if(flag)
                    {   
                        // segment trees
                        std::vector<cv::Mat> masks;
                        masks = treeSegmentation(color_image);

                        if(masks.empty())
                        {
                            // load empty forest
                            std::pair<bool, std::vector<std::vector<Ex_TreeNode>>> out_forest;
                            out_forest.first = false;
                            estimator.inputForest(time, out_forest);
                            continue;
                        }
                        
                        // pre process depth image
                        cv::Mat processed_depth_image;
                        processed_depth_image = preprocessDepthImage(depth_image, masks);

                        // horizontal clustering
                        std::vector<cv::Mat> hclustered_masks;
                        hclustered_masks = horizontalClustering(processed_depth_image, masks);

                        
                        // skeletonization
                        std::vector<std::vector<Ex_TreeNode>> forest;
                        forest = skeletonize(hclustered_masks, processed_depth_image, color_image);
                        
                        // visualize skeletons
                        cv::Mat color_skeletons;
                        color_skeletons = draw_forest(color_image, hclustered_masks, forest);
                        
                        // publish skeleton visualization
                        // image
                        std_msgs::msg::Header header = ref_frame["tree_camera"].header;
                        header.frame_id = "oakd_rgb_camera_optical_frame"; // TO MODIFY AFTER TF IS CORRECT
                        builtin_interfaces::msg::Time stamp;
                        stamp.sec  = static_cast<int32_t>(time);
                        stamp.nanosec = static_cast<uint32_t>((time - stamp.sec) * 1e9);
                        header.stamp = stamp;
                        
                        cv::Mat rgb_image;
                        cv::cvtColor(color_skeletons, rgb_image, cv::COLOR_BGR2RGB);
                        auto img_msg = cv_bridge::CvImage(
                            header, "bgr8", rgb_image).toImageMsg();
                        pub_color_skels->publish(*img_msg);

                        // camera info
                        sensor_msgs::msg::CameraInfo cam_info = ref_frame["tree_camera"];
                        cam_info.header.stamp = header.stamp;
                        cam_info.header.frame_id = "oakd_rgb_camera_optical_frame"; // TO MODIFY AFTER TF IS CORRECT

                        pub_color_skels_info->publish(cam_info);
                        
                        // debug
                        ///// LOG /////
                        std::ostringstream oss;
                        oss << "=========================================================================\nVN forest at time " << std::setprecision(15) << time << std::endl;
                        for(size_t i = 0; i < forest.size(); ++i){
                            oss << "tree " << i << "\n--------------------------------------------------------------------------" << std::endl; 
                            for(size_t j = 0; j < forest[i].size(); ++j){
                                oss << "    node " << forest[i][j].ex_id << " pos " << forest[i][j].x << " " << forest[i][j].y << " " << forest[i][j].z << " parent " << forest[i][j].ex_parent << " sons ";
                                for(const auto& s : forest[i][j].ex_sons){
                                    oss << s << " ";
                                }
                                oss << "fd ";
                                for(const auto& fdi : forest[i][j].fd_brief){
                                    oss << static_cast<int>(fdi) << " ";
                                }
                                oss << std::endl;
                            }
                        }
                        logMessage(oss.str());
                        ///// LOG /////

                        // load forest
                        std::pair<bool, std::vector<std::vector<Ex_TreeNode>>> out_forest;
                        out_forest.first = true;
                        out_forest.second = forest;
                        estimator.inputForest(time, out_forest);
                    }
                }
                std::chrono::milliseconds dura(2);
                std::this_thread::sleep_for(dura);
            }
        }

        void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
        {
            if(USE_TOPIC)
            {   
                if((NUM_OF_CAM - cam_info_set_flag) == 0)
                {
                    std::cout << "feature cb" << std::endl;
                    std::cout << "Feature: " << feature_msg->points.size() << std::endl;
                    
                    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
                    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
                    {   
                        int feature_id = feature_msg->channels[0].values[i];
                        int camera_id = feature_msg->channels[1].values[i];
                        double x = feature_msg->points[i].x;
                        double y = feature_msg->points[i].y;
                        double z = feature_msg->points[i].z;
                        double p_u = feature_msg->channels[2].values[i];
                        double p_v = feature_msg->channels[3].values[i];
                        double velocity_x = feature_msg->channels[4].values[i];
                        double velocity_y = feature_msg->channels[5].values[i];
                        
                        if(feature_msg->channels.size() > 6)
                        {
                            double gx = feature_msg->channels[6].values[i];
                            double gy = feature_msg->channels[7].values[i];
                            double gz = feature_msg->channels[8].values[i];
                            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
                            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
                        }
                        assert(z == 1);
                        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
                    }
                    double t = feature_msg->header.stamp.sec + feature_msg->header.stamp.nanosec * (1e-9);
                    estimator.inputFeature(t, featureFrame);
                    return;
                }
            }
            else
            {
                std::cout << "feature cb" << std::endl;
                std::cout << "Feature: " << feature_msg->points.size() << std::endl;
                
                map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
                for (unsigned int i = 0; i < feature_msg->points.size(); i++)
                {   
                    int feature_id = feature_msg->channels[0].values[i];
                    int camera_id = feature_msg->channels[1].values[i];
                    double x = feature_msg->points[i].x;
                    double y = feature_msg->points[i].y;
                    double z = feature_msg->points[i].z;
                    double p_u = feature_msg->channels[2].values[i];
                    double p_v = feature_msg->channels[3].values[i];
                    double velocity_x = feature_msg->channels[4].values[i];
                    double velocity_y = feature_msg->channels[5].values[i];
                    
                    if(feature_msg->channels.size() > 6)
                    {
                        double gx = feature_msg->channels[6].values[i];
                        double gy = feature_msg->channels[7].values[i];
                        double gz = feature_msg->channels[8].values[i];
                        pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
                        //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
                    }
                    assert(z == 1);
                    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                    featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
                }
                double t = feature_msg->header.stamp.sec + feature_msg->header.stamp.nanosec * (1e-9);
                estimator.inputFeature(t, featureFrame);
                return;
            }
        }

        void logMessage(const std::string& message) 
        {
            const std::string LOG_FILE_PATH = "/home/glugano/Desktop/log.txt";

            std::ofstream logFile(LOG_FILE_PATH, std::ios::app);
            if (!logFile) {
                std::cerr << "Error: Unable to open log file." << std::endl;
                return;
            }

            logFile << message << std::endl;
        }

        // Helper function to calculate median of cv::Mat with NaN values
        double nanMedian(const cv::Mat& depth_float) 
        {
            std::vector<float> valid_values;
            valid_values.reserve(depth_float.total());
            
            for (int i = 0; i < depth_float.rows; i++) {
                for (int j = 0; j < depth_float.cols; j++) {
                    float val = depth_float.at<float>(i, j);
                    if (!std::isnan(val)) {
                        valid_values.push_back(val);
                    }
                }
            }
            
            if (valid_values.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            
            std::nth_element(valid_values.begin(), 
                            valid_values.begin() + valid_values.size() / 2, 
                            valid_values.end());
            return valid_values[valid_values.size() / 2];
        }

        cv::Mat fillDepthHoles(const cv::Mat& depth_image) {
            /**
            * Fill small holes in depth image using interpolation
            * Works directly with float32 to avoid precision loss
            */
            
            cv::Mat depth_filled = depth_image.clone();
            cv::Mat mask = cv::Mat::zeros(depth_image.size(), CV_8U);
            
            // Identify holes and convert NaN to 0
            for (int i = 0; i < depth_image.rows; i++) {
                for (int j = 0; j < depth_image.cols; j++) {
                    float val = depth_image.at<float>(i, j);
                    if (std::isnan(val) || val == 0.0f) {
                        depth_filled.at<float>(i, j) = 0.0f;
                        mask.at<uchar>(i, j) = 255;
                    }
                }
            }
            
            if (cv::countNonZero(mask) == 0) {
                return depth_filled;
            }
            
            // Inpaint directly on float32 (OpenCV supports this)
            cv::Mat inpainted;
            cv::inpaint(depth_filled, mask, inpainted, 3, cv::INPAINT_TELEA);
            
            // Only replace hole values, keep original valid values
            for (int i = 0; i < depth_image.rows; i++) {
                for (int j = 0; j < depth_image.cols; j++) {
                    float original = depth_image.at<float>(i, j);
                    if (std::isnan(original) || original == 0.0f) {
                        // This was a hole, use inpainted value
                        depth_filled.at<float>(i, j) = inpainted.at<float>(i, j);
                    } else {
                        // Keep original valid value
                        depth_filled.at<float>(i, j) = original;
                    }
                }
            }
            
            return depth_filled;
        }

        cv::Mat preprocessDepthImage(const cv::Mat& depth_image, const std::vector<cv::Mat>& masks) 
        {
            // Create a working copy immediately
            cv::Mat depth_working = depth_image.clone();
            
            // 0. Remove obviously invalid values
            for (int i = 0; i < depth_working.rows; i++) {
                for (int j = 0; j < depth_working.cols; j++) {
                    float& val = depth_working.at<float>(i, j);
                    if (val == 0.0f) {
                        val = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }

            // 1. Hole filling (interpolate small gaps) ( to apply before the masking)
            if (TREE_DPP_HF_FILTER_F) {
                depth_working = fillDepthHoles(depth_working);
            }

            // 2. threshold on distance
            for (int i = 0; i < depth_working.rows; i++) {
                for (int j = 0; j < depth_working.cols; j++) {
                    float& val = depth_working.at<float>(i, j);
                    if (val < TREE_DPP_MIN_D || val > TREE_DPP_MAX_D) {
                        val = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }

            // 3. Combine all masks into a single mask (OR operation)
            cv::Mat combined_mask = cv::Mat::zeros(depth_working.size(), CV_8U);
            for (const auto& mask : masks) {
                if (mask.size() == depth_working.size()) {
                    cv::bitwise_or(combined_mask, mask, combined_mask);
                }
            }
            
            // Apply combined mask: set masked-out pixels to NaN
            for (int i = 0; i < depth_working.rows; i++) {
                for (int j = 0; j < depth_working.cols; j++) {
                    if (combined_mask.at<uchar>(i, j) < 1) {
                        depth_working.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            
            
            // 2. Handle sunlight interference (very common in orchards)
            // Remove values that are too close (likely sunlight saturation)
            double median_depth = nanMedian(depth_working);
            if (!std::isnan(median_depth)) {
                // If many pixels are much closer than median, likely sunlight interference
                double close_threshold = median_depth * 0.1;
                int too_close_count = 0;
                
                for (int i = 0; i < depth_working.rows; i++) {
                    for (int j = 0; j < depth_working.cols; j++) {
                        float val = depth_working.at<float>(i, j);
                        if (!std::isnan(val) && val < close_threshold) {
                            too_close_count++;
                        }
                    }
                }
                
                // If >30% too close
                if (too_close_count > (depth_working.total() * 0.3)) {
                    for (int i = 0; i < depth_working.rows; i++) {
                        for (int j = 0; j < depth_working.cols; j++) {
                            float& val = depth_working.at<float>(i, j);
                            if (!std::isnan(val) && val < close_threshold) {
                                val = std::numeric_limits<float>::quiet_NaN();
                            }
                        }
                    }
                }
            }
            
            // 3. Bilateral filtering (preserves edges while reducing noise)
            if (TREE_DPP_B_FILTER_F) {
                // Convert nan to 0 for filtering
                cv::Mat depth_for_filter = depth_working.clone();
                cv::Mat nan_mask = cv::Mat::zeros(depth_working.size(), CV_8U);
                
                bool has_valid = false;
                for (int i = 0; i < depth_working.rows; i++) {
                    for (int j = 0; j < depth_working.cols; j++) {
                        if (std::isnan(depth_working.at<float>(i, j))) {
                            depth_for_filter.at<float>(i, j) = 0.0f;
                            nan_mask.at<uchar>(i, j) = 1;
                        } else {
                            has_valid = true;
                        }
                    }
                }
                
                if (has_valid) {
                    // Scale up for bilateral filter
                    depth_for_filter *= 1000.0f;
                    
                    cv::Mat filtered;
                    cv::bilateralFilter(depth_for_filter, filtered, 9, 50, 50);
                    
                    // Scale back and restore NaN
                    filtered /= 1000.0f;
                    for (int i = 0; i < filtered.rows; i++) {
                        for (int j = 0; j < filtered.cols; j++) {
                            if (nan_mask.at<uchar>(i, j) == 1) {
                                filtered.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
                            }
                        }
                    }
                    depth_working = filtered;
                }
            }
            
            // 4. Temporal filtering (thread-safe)
            if (TREE_DPP_T_FILTER_F) {
                std::lock_guard<std::mutex> lock(prev_depth_mutex_);
                
                if (!prev_depth_.empty() && prev_depth_.size() == depth_working.size()) {
                    for (int i = 0; i < depth_working.rows; i++) {
                        for (int j = 0; j < depth_working.cols; j++) {
                            float curr_val = depth_working.at<float>(i, j);
                            float prev_val = prev_depth_.at<float>(i, j);
                            
                            // Only blend if BOTH current and previous values are valid
                            if (!std::isnan(curr_val) && !std::isnan(prev_val)) {
                                depth_working.at<float>(i, j) = 
                                    TREE_DPP_T_FILTER_A * curr_val + 
                                    (1.0 - TREE_DPP_T_FILTER_A) * prev_val;
                            }
                            // If either is NaN, keep the current value (NaN or valid)
                        }
                    }
                }
                
                prev_depth_ = depth_working.clone();
            }
            
            return depth_working;
        }

        void initTensorRTModel() 
        {
            // set class count
            trt_.numClasses = 1;
            
            // Load engine
            std::ifstream engineFile(YOLO_MODEL_PATH, std::ios::binary);
            if (!engineFile.good()) {
                RCLCPP_ERROR(this->get_logger(), "[TensorRT] Engine file not found!");
                throw std::runtime_error("TensorRT engine not found");
            }
            
            engineFile.seekg(0, std::ios::end);
            size_t size = engineFile.tellg();
            engineFile.seekg(0, std::ios::beg);
            
            std::vector<char> engineData(size);
            engineFile.read(engineData.data(), size);
            engineFile.close();
            
            // Deserialize engine
            trt_.runtime = nvinfer1::createInferRuntime(trt_.logger);
            trt_.engine = trt_.runtime->deserializeCudaEngine(engineData.data(), size);
            
            if (!trt_.engine) {
                RCLCPP_ERROR(this->get_logger(), "[TensorRT] Failed to deserialize engine");
                throw std::runtime_error("Failed to deserialize TensorRT engine");
            }
            
            trt_.context = trt_.engine->createExecutionContext();
            
            const int numIOTensors = trt_.engine->getNbIOTensors();
            
            RCLCPP_INFO(this->get_logger(), "[TensorRT] Number of I/O tensors: %d", numIOTensors);
            
            for (int i = 0; i < numIOTensors; i++) {
                const char* tensorName = trt_.engine->getIOTensorName(i);
                nvinfer1::TensorIOMode ioMode = trt_.engine->getTensorIOMode(tensorName);
                auto dims = trt_.engine->getTensorShape(tensorName);
                
                // Calculate tensor size
                size_t tensorSize = 1;
                for (int j = 0; j < dims.nbDims; j++) {
                    tensorSize *= dims.d[j];
                }
                
                size_t bytes = tensorSize * sizeof(float);
                
                // RCLCPP_INFO(this->get_logger(), "[TensorRT] Tensor %d: %s, Mode: %s", 
                //             i, tensorName, 
                //             (ioMode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT"));
                
                // // Print dimensions
                // std::string dimStr = "[";
                // for (int j = 0; j < dims.nbDims; j++) {
                //     dimStr += std::to_string(dims.d[j]);
                //     if (j < dims.nbDims - 1) dimStr += ", ";
                // }
                // dimStr += "]";
                // RCLCPP_INFO(this->get_logger(), "  Shape: %s, Size: %zu", dimStr.c_str(), tensorSize);
                
                if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
                    trt_.inputIndex = i;
                    trt_.inputSize = tensorSize;
                    cudaMalloc(&trt_.d_input, bytes);
                    
                    // Set input tensor address for TensorRT 10.x
                    trt_.context->setTensorAddress(tensorName, trt_.d_input);
                    
                    // RCLCPP_INFO(this->get_logger(), "  -> Allocated INPUT buffer: %zu bytes", bytes);
                } 
                else 
                {
                    // Handle multiple outputs
                    if (trt_.outputIndex == -1) {
                        trt_.outputIndex = i;
                        trt_.outputSize = tensorSize;
                        cudaMalloc(&trt_.d_output, bytes);
                        
                        // Set output tensor address
                        trt_.context->setTensorAddress(tensorName, trt_.d_output);
                        
                        // RCLCPP_INFO(this->get_logger(), "  -> Allocated OUTPUT buffer: %zu bytes", bytes);
                    } 
                    else 
                    {
                        trt_.protosIndex = i;
                        trt_.protosSize = tensorSize;
                        cudaMalloc(&trt_.d_protos, bytes);
                        
                        // Set protos tensor address
                        trt_.context->setTensorAddress(tensorName, trt_.d_protos);
                        
                        // RCLCPP_INFO(this->get_logger(), "  -> Allocated PROTOS buffer: %zu bytes", bytes);
                    }
                }
            }
            
            cudaStreamCreate(&trt_.stream);
            
            RCLCPP_INFO(this->get_logger(), "[TensorRT] Model loaded successfully: %d classes", trt_.numClasses);
        }

        void initONNXModel() 
        {
            // set class count
            onnx_.numClasses = 1;
            
            // Session options
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // Create session
            onnx_.session = std::make_unique<Ort::Session>(
                onnx_.env, YOLO_MODEL_PATH.c_str(), sessionOptions
            );
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            
            size_t numInputNodes = onnx_.session->GetInputCount();
            onnx_.inputNames.resize(numInputNodes);
            for (size_t i = 0; i < numInputNodes; i++) {
                auto inputName = onnx_.session->GetInputNameAllocated(i, allocator);
                onnx_.inputNames[i] = strdup(inputName.get());
            }
            
            size_t numOutputNodes = onnx_.session->GetOutputCount();
            onnx_.outputNames.resize(numOutputNodes);
            for (size_t i = 0; i < numOutputNodes; i++) {
                auto outputName = onnx_.session->GetOutputNameAllocated(i, allocator);
                onnx_.outputNames[i] = strdup(outputName.get());
            }
            
            RCLCPP_INFO(this->get_logger(), "[ONNX] Model loaded: %d classes", onnx_.numClasses);
        }

        void loadYOLOModel() {
            if(USE_GPU) // if use gpuset up the tensorrt model
            {
                initTensorRTModel();
            }
            else //else set up the onnx model
            {
                initONNXModel();
            }
        }

        // Letterbox preprocessing
        cv::Mat letterbox(const cv::Mat& image, int targetWidth, int targetHeight) 
        {
            float ratio = std::min(static_cast<float>(targetWidth) / image.cols,
                                static_cast<float>(targetHeight) / image.rows);
            
            int newWidth = static_cast<int>(image.cols * ratio);
            int newHeight = static_cast<int>(image.rows * ratio);
            
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(newWidth, newHeight));
            
            int dw = targetWidth - newWidth;
            int dh = targetHeight - newHeight;
            
            cv::Mat output;
            cv::copyMakeBorder(resized, output, 
                            dh / 2, dh - dh / 2,
                            dw / 2, dw - dw / 2,
                            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            
            return output;
        }

        // Apply Non-Maximum Suppression
        void applyNMS(std::vector<DetectionBox>& detections) 
        {
            std::sort(detections.begin(), detections.end(),
                    [](const DetectionBox& a, const DetectionBox& b) {
                        return a.confidence > b.confidence;
                    });
            
            std::vector<bool> suppressed(detections.size(), false);
            
            for (size_t i = 0; i < detections.size(); i++) {
                if (suppressed[i]) continue;
                
                for (size_t j = i + 1; j < detections.size(); j++) {
                    if (suppressed[j] || detections[i].classId != detections[j].classId) 
                        continue;
                    
                    float intersection = (detections[i].box & detections[j].box).area();
                    float unionArea = (detections[i].box | detections[j].box).area();
                    float iou = intersection / unionArea;
                    
                    if (iou > TS_IOU_T) {
                        suppressed[j] = true;
                    }
                }
            }
            
            detections.erase(
                std::remove_if(detections.begin(), detections.end(),
                            [&suppressed, &detections](const DetectionBox& det) {
                                size_t idx = &det - &detections[0];
                                return suppressed[idx];
                            }),
                detections.end());
        }

        // Parse YOLO output to detections
        std::vector<DetectionBox> parseDetections(float* output, const cv::Size& originalSize, const cv::Size& inputSize, int numProposals = 8400) {
            std::vector<DetectionBox> detections;
            
            // YOLO11-seg output: [1, 116, 8400]
            // 116 = 4 (bbox) + 80 (classes) + 32 (mask coeffs)
            int numMasks = 32;
            
            // Calculate letterbox parameters
            float scale = std::min(
                static_cast<float>(inputSize.width) / originalSize.width,
                static_cast<float>(inputSize.height) / originalSize.height
            );

            int scaledWidth = static_cast<int>(originalSize.width * scale);
            int scaledHeight = static_cast<int>(originalSize.height * scale);

            int padLeft = (inputSize.width - scaledWidth) / 2;
            int padTop = (inputSize.height - scaledHeight) / 2;

            for (int i = 0; i < numProposals; i++) {
                // Get best class
                float maxScore = 0.0f;
                int classId = 0;
                
                for (int c = 0; c < 1; c++) {
                    float score = output[(4 + c) * numProposals + i];
                    if (score > maxScore) {
                        maxScore = score;
                        classId = c;
                    }
                }
                
                if (maxScore < TS_CONFIDENCE_T) continue;
                
                // Get bbox (center format)
                float cx = output[0 * numProposals + i];
                float cy = output[1 * numProposals + i];
                float w = output[2 * numProposals + i];
                float h = output[3 * numProposals + i];
                
                // convert in original image size
                float cx_o = cx - padLeft;
                float cy_o = cy - padTop;
                
                // Convert to corner format
                int x1 = static_cast<int>((cx_o - w / 2) / scale);
                int y1 = static_cast<int>((cy_o - h / 2) / scale);
                int x2 = static_cast<int>((cx_o + w / 2) / scale);
                int y2 = static_cast<int>((cy_o + h / 2) / scale);
                
                // Clamp
                x1 = std::max(0, std::min(x1, originalSize.width - 1));
                y1 = std::max(0, std::min(y1, originalSize.height - 1));
                x2 = std::max(0, std::min(x2, originalSize.width - 1));
                y2 = std::max(0, std::min(y2, originalSize.height - 1));
                
                // Get mask coefficients
                std::vector<float> maskCoeffs(numMasks);
                for (int m = 0; m < numMasks; m++) {
                    maskCoeffs[m] = output[(4 + 1 + m) * numProposals + i];
                }
                
                DetectionBox det;
                det.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                det.confidence = maxScore;
                det.classId = classId;
                det.maskCoeffs = maskCoeffs;
                
                detections.push_back(det);
            }
            
            applyNMS(detections);
            return detections;
        }

        // Process single mask from coefficients
        cv::Mat processMask(const std::vector<float>& maskCoeffs, const cv::Mat& protos, 
                            const cv::Rect& box, const cv::Size& originalSize, 
                            const cv::Size& inputSize) 
        {
            // protos shape: [32, 160, 160]
            int numMasks = protos.size[0];
            int protoH = protos.size[1];
            int protoW = protos.size[2];
            
            // Coefficients as Mat
            cv::Mat coeffsMat(1, numMasks, CV_32F);
            for (int i = 0; i < numMasks; i++) {
                coeffsMat.at<float>(0, i) = maskCoeffs[i];
            }
            
            // Reshape protos to [32, 160*160]
            cv::Mat protosReshaped = protos.reshape(1, numMasks);
            
            // Matrix multiply: [1, 32] @ [32, 160*160] = [1, 160*160]
            cv::Mat mask = coeffsMat * protosReshaped;
            mask = mask.reshape(1, protoH);
            
            // Apply sigmoid
            cv::exp(-mask, mask);
            mask = 1.0 / (1.0 + mask);
            
            // Resize to input size (160x160 -> 640x640)
            cv::Mat maskResized;
            cv::resize(mask, maskResized, inputSize, 0, 0, cv::INTER_LINEAR);
            
            // Calculate letterbox parameters
            // Letterbox uses uniform scaling + padding to maintain aspect ratio
            float scale = std::min(
                static_cast<float>(inputSize.width) / originalSize.width,
                static_cast<float>(inputSize.height) / originalSize.height
            );
            
            // Calculate the actual scaled size (before padding)
            int scaledWidth = static_cast<int>(originalSize.width * scale);
            int scaledHeight = static_cast<int>(originalSize.height * scale);
            
            // Calculate padding offsets
            int padLeft = (inputSize.width - scaledWidth) / 2;
            int padTop = (inputSize.height - scaledHeight) / 2;
            
            // Scale box coordinates using the uniform scale + offset
            cv::Rect boxScaled(
                static_cast<int>(box.x * scale) + padLeft,
                static_cast<int>(box.y * scale) + padTop,
                static_cast<int>(box.width * scale),
                static_cast<int>(box.height * scale)
            );
            
            // Clamp box to mask bounds
            boxScaled.x = std::max(0, std::min(boxScaled.x, inputSize.width - 1));
            boxScaled.y = std::max(0, std::min(boxScaled.y, inputSize.height - 1));
            boxScaled.width = std::min(boxScaled.width, inputSize.width - boxScaled.x);
            boxScaled.height = std::min(boxScaled.height, inputSize.height - boxScaled.y);
            
            // Crop mask to box
            cv::Mat maskCropped = maskResized(boxScaled);
            
            // Resize to original box size
            cv::Mat maskFinal;
            cv::resize(maskCropped, maskFinal, cv::Size(box.width, box.height), 
                    0, 0, cv::INTER_LINEAR);
            
            // Threshold to binary
            cv::threshold(maskFinal, maskFinal, TS_MASK_T, 1.0, cv::THRESH_BINARY);
            maskFinal.convertTo(maskFinal, CV_8U);
            
            return maskFinal;
        }

        // Generate all binary masks from detections
        std::vector<cv::Mat> generateMasks(const std::vector<DetectionBox>& detections, float* protosData, const cv::Size& originalSize, const cv::Size& inputSize, int protoH = 160, int protoW = 160) {
            std::vector<cv::Mat> masks;
            
            // Create prototypes Mat: [32, 160, 160]
            int numMasks = 32;
            int protoDims[] = {numMasks, protoH, protoW};
            cv::Mat protos(3, protoDims, CV_32F, protosData);
            
            for (const auto& det : detections) {
                // Process mask
                cv::Mat mask = processMask(det.maskCoeffs, protos, det.box, 
                                        originalSize, inputSize);
                
                // Create full-size binary mask
                cv::Mat fullMask = cv::Mat::zeros(originalSize, CV_8U);
                
                // Place mask in correct position
                cv::Rect roi = det.box & cv::Rect(0, 0, originalSize.width, originalSize.height);
                if (roi.width > 0 && roi.height > 0 && 
                    mask.rows == roi.height && mask.cols == roi.width) {
                    mask.copyTo(fullMask(roi));
                }
                
                masks.push_back(fullMask);
            }
            
            return masks;
        }

        std::vector<cv::Mat> inferAndPostprocessTensorRT(TensorRTRuntime& trt, const cv::Mat& image) 
        {
            cv::Size originalSize = image.size();
            
            // Preprocess
            cv::Mat processed = letterbox(image, 640, 640);
            cv::Mat blob;
            cv::dnn::blobFromImage(processed, blob, 1.0 / 255.0, 
                                cv::Size(640, 640),
                                cv::Scalar(), true, false);
            
            // Copy input to GPU
            cudaMemcpyAsync(trt.d_input, blob.ptr<float>(), 
                        trt.inputSize * sizeof(float),
                        cudaMemcpyHostToDevice, trt.stream);
            
            // Run inference
            bool success = trt.context->enqueueV3(trt.stream);
            
            if (!success) {
                RCLCPP_ERROR(this->get_logger(), "[TensorRT] Inference failed!");
                return {};
            }

            // Allocate local host memory for outputs
            std::vector<float> h_output(trt.outputSize);
            std::vector<float> h_protos(trt.protosSize);
            
            // Copy results back to CPU
            cudaMemcpyAsync(h_output.data(), trt.d_output,
                        trt.outputSize * sizeof(float),
                        cudaMemcpyDeviceToHost, trt.stream);
            
            cudaMemcpyAsync(h_protos.data(), trt.d_protos,
                        trt.protosSize * sizeof(float),
                        cudaMemcpyDeviceToHost, trt.stream);
            
            cudaStreamSynchronize(trt.stream);
            
            // Postprocess: parse detections
            cv::Size inputSize(640, 640);
            std::vector<DetectionBox> detections = parseDetections(h_output.data(), originalSize, inputSize);
            
            // Generate and return binary masks
            return generateMasks(detections, h_protos.data(), originalSize, inputSize);
        }

        std::vector<cv::Mat> inferAndPostprocessONNX(ONNXRuntime& onnx, const cv::Mat& image) 
        {
            cv::Size originalSize = image.size();
            
            // Preprocess
            cv::Mat processed = letterbox(image, 640, 640);
            cv::Mat blob;
            cv::dnn::blobFromImage(processed, blob, 1.0 / 255.0, 
                                cv::Size(640, 640),
                                cv::Scalar(), true, false);
            
            // Create input tensor
            std::vector<int64_t> inputShape = {1, 3, 640, 640};
            size_t inputTensorSize = 1 * 3 * 640 * 640;
            
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, blob.ptr<float>(), inputTensorSize,
                inputShape.data(), inputShape.size());
            
            // Run inference
            auto outputTensors = onnx.session->Run(
                Ort::RunOptions{nullptr},
                onnx.inputNames.data(), &inputTensor, 1,
                onnx.outputNames.data(), onnx.outputNames.size());
            
            // Get output sizes and copy data to local buffers
            auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
            size_t outputSize = 1;
            for (auto dim : outputShape) outputSize *= dim;
            
            auto protosShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
            size_t protosSize = 1;
            for (auto dim : protosShape) protosSize *= dim;
            
            // Copy to local buffers (safer but uses more memory)
            std::vector<float> h_output(outputSize);
            std::vector<float> h_protos(protosSize);
            
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            float* protosData = outputTensors[1].GetTensorMutableData<float>();
            
            std::memcpy(h_output.data(), outputData, outputSize * sizeof(float));
            std::memcpy(h_protos.data(), protosData, protosSize * sizeof(float));
            
            // Postprocess: parse detections
            cv::Size inputSize(640, 640);
            std::vector<DetectionBox> detections = parseDetections(h_output.data(), originalSize, inputSize);
            
            // Generate and return binary masks
            return generateMasks(detections, h_protos.data(), originalSize, inputSize);
        }

        std::vector<cv::Mat> treeSegmentation(const cv::Mat color_image)
        {   
            if (color_image.empty()) {
                RCLCPP_WARN(this->get_logger(), "[TreeSeg] Empty input image");
                return {};
            }

            std::vector<cv::Mat> masks;
        
            try {
                if (USE_GPU) {
                    masks = inferAndPostprocessTensorRT(trt_, color_image);
                } else {
                    masks = inferAndPostprocessONNX(onnx_, color_image);
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "[TreeSeg] Inference failed: %s", e.what());
                return {};
            }
            
            if (masks.empty()) {
                RCLCPP_DEBUG(this->get_logger(), "[TreeSeg] No trees detected");
            } else {
                RCLCPP_DEBUG(this->get_logger(), "[TreeSeg] Detected %zu tree masks", masks.size());
            }

            return masks;
            
        }

        std::vector<cv::Mat> horizontalClustering(const cv::Mat& depth_image, const std::vector<cv::Mat>& masks)
        {
            if (masks.empty()) return {};
            
            const auto& camera_info = ref_frame["tree_camera"];
            const double fx = camera_info.k[0];
            const double fy = camera_info.k[4];
            const double cx = camera_info.k[2];
            const double cy = camera_info.k[5];
            
            // Structure to hold only essential mask data
            struct MaskData {
                Eigen::Vector2d barycenter_2d;
            };
            
            std::vector<MaskData> mask_data;
            mask_data.reserve(masks.size());
            
            // Step 1: Compute horizontal barycenters for each mask
            for (const auto& mask : masks) {
                double sum_x = 0, sum_y = 0;
                int valid_points = 0;
                
                // Get pointer to mask and depth data for faster access
                const uchar* mask_ptr = mask.ptr<uchar>();
                const float* depth_ptr = depth_image.ptr<float>();
                
                const int total_pixels = mask.rows * mask.cols;
                
                for (int idx = 0; idx < total_pixels; ++idx) {
                    if (mask_ptr[idx] > 0) {
                        float depth = depth_ptr[idx];
                        
                        if (std::isfinite(depth) && depth > 0) {
                            // Convert linear index back to u, v
                            int v = idx / mask.cols;
                            int u = idx % mask.cols;
                            
                            // Convert to 3D point
                            double z = depth;
                            double x = (u - cx) * z / fx;
                            double y = (v - cy) * z / fy;
                            
                            // Project directly during accumulation
                            Eigen::Vector3d point(x, y, z);

                            // Apply full rotation from camera to base_link
                            Eigen::Vector3d point_in_baselink = R_bodylink_tree * point; 

                            // Take only horizontal components (x, y in base_link frame)
                            sum_x += point_in_baselink(0);  // First element
                            sum_y += point_in_baselink(1);  // Second element
                            valid_points++;
                        }
                    }
                }
                
                if (valid_points == 0) {
                    // Use a sentinel value or skip - we'll handle this in clustering
                    mask_data.push_back({Eigen::Vector2d(0, 0)});
                    continue;
                }
                
                Eigen::Vector2d barycenter_2d(sum_x / valid_points, sum_y / valid_points);
                mask_data.push_back({barycenter_2d});
            }
            
            int n = mask_data.size();
            
            // Step 2: Union-Find with path compression
            // Initialize: each mask starts in its own cluster (parent[i] = i)
            std::vector<int> parent(n);
            std::iota(parent.begin(), parent.end(), 0);  // Same as: for(int i=0; i<n; i++) parent[i]=i;
            
            // Find: returns the root/leader of the cluster that x belongs to
            std::function<int(int)> find = [&](int x) {
                if (parent[x] != x) {
                    parent[x] = find(parent[x]);  // Path compression optimization
                }
                return parent[x];
            };
            
            // Unite: merges the clusters containing x and y
            auto unite = [&](int x, int y) {
                int px = find(x);  // Find cluster leader of x
                int py = find(y);  // Find cluster leader of y
                if (px != py) {
                    parent[px] = py;  // Make one leader point to the other
                }
            };
            
            // Step 3: Unite masks on-the-fly (no distance matrix storage)
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    // Use squared distance to avoid sqrt
                    double dx = mask_data[i].barycenter_2d.x() - mask_data[j].barycenter_2d.x();
                    double dy = mask_data[i].barycenter_2d.y() - mask_data[j].barycenter_2d.y();
                    double dist_sq = dx * dx + dy * dy;
                    if (dist_sq < H_CLUSTERING_T_SQ) {
                        unite(i, j);  // Join these two masks into the same cluster
                    }
                }
            }
            
            // Step 4: Count clusters and build index mapping
            std::unordered_map<int, int> root_to_cluster;
            int cluster_count = 0;
            
            for (int i = 0; i < n; ++i) {
                int root = find(i);
                if (root_to_cluster.find(root) == root_to_cluster.end()) {
                    root_to_cluster[root] = cluster_count++;
                }
            }
            
            // Step 5: Merge masks directly into output vector
            std::vector<cv::Mat> merged_masks(cluster_count);
            for (int i = 0; i < cluster_count; ++i) {
                merged_masks[i] = cv::Mat::zeros(masks[0].size(), masks[0].type());
            }
            
            for (int i = 0; i < n; ++i) {
                int root = find(i);
                int cluster_id = root_to_cluster[root];
                cv::bitwise_or(merged_masks[cluster_id], masks[i], merged_masks[cluster_id]);
            }
            
            return merged_masks;
        }

        cv::Point findRoot(const cv::Mat& mask) 
        {
            int lowestY = -1;
            cv::Point lowestEndpoint(-1, -1);
            
            // Iterate through the image
            for (int y = 0; y < mask.rows; y++) {
                for (int x = 0; x < mask.cols; x++) {
                    // Check if current pixel is white
                    if (mask.at<uchar>(y, x) > 0) {
                        // Count adjacent white pixels (8-connectivity)
                        int adjacentCount = 0;
                        
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue; // Skip center pixel
                                
                                int ny = y + dy;
                                int nx = x + dx;
                                
                                // Check bounds
                                if (ny >= 0 && ny < mask.rows && nx >= 0 && nx < mask.cols) {
                                    if (mask.at<uchar>(ny, nx) > 0) {
                                        adjacentCount++;
                                    }
                                }
                            }
                        }
                        
                        // If this is an endpoint (only 1 neighbor) and it's lower than previous
                        if (adjacentCount == 1) {
                            if (lowestY == -1 || y > lowestY) {
                                lowestY = y;
                                lowestEndpoint = cv::Point(x, y);
                            }
                        }
                    }
                }
            }
            
            return lowestEndpoint;
        }

        
        // Hash function for cv::Point to use in unordered_set/map
        struct PointHash {
            std::size_t operator()(const cv::Point& p) const {
                return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
            }
        };

        // Helper function to get 8-directional neighbors
        std::vector<cv::Point> getNeighborsFast(const cv::Point& node, 
                                                const cv::Mat& mask,
                                                const std::unordered_set<cv::Point, PointHash>& skeleton_set) {
            std::vector<cv::Point> neighbors;
            
            // 8 directions
            static const int dirs[8][2] = {
                {-1, -1}, {-1, 0}, {-1, 1},
                {0, -1},           {0, 1},
                {1, -1},  {1, 0},  {1, 1}
            };
            
            for (int i = 0; i < 8; i++) {
                int ny = node.y + dirs[i][0];
                int nx = node.x + dirs[i][1];
                
                // Check bounds
                if (ny >= 0 && ny < mask.rows && nx >= 0 && nx < mask.cols) {
                    cv::Point neighbor(nx, ny);
                    if (skeleton_set.find(neighbor) != skeleton_set.end()) {
                        neighbors.push_back(neighbor);
                    }
                }
            }
            
            return neighbors;
        }

        std::vector<uint8_t> evaluate_visual_fd(const cv::Mat& color_image, const cv::Point& pt)
        {
            // Convert to grayscale
            cv::Mat gray_image;
            cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
            
            // Create a KeyPoint from your point
            std::vector<cv::KeyPoint> keypoints;
            keypoints.push_back(cv::KeyPoint(pt.x, pt.y, 31));
            
            // Create BRIEF descriptor extractor
            cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = 
                cv::xfeatures2d::BriefDescriptorExtractor::create();
            
            // Compute descriptor
            cv::Mat descriptors;
            brief->compute(gray_image, keypoints, descriptors);
            
            // Convert to vector<uint8_t>
            std::vector<uint8_t> descriptor_vec;
            if (!descriptors.empty()) {
                descriptor_vec.assign(descriptors.data, 
                                    descriptors.data + descriptors.cols);
            } else {
                //std::cout << "Failed to compute descriptor" << std::endl;
                descriptor_vec.assign(32, 0);
            }

            return descriptor_vec;
            
        }

        std::unordered_map<cv::Point, Ex_TreeNode_Skel, PointHash> skeletonize2d(const cv::Point& root, const cv::Mat& component_mask) 
        {
            int every_k_point = DOWN_SKEL_K;
            // Pre-compute skeleton set for fast lookup
            std::unordered_set<cv::Point, PointHash> skeleton_set;
            for (int y = 0; y < component_mask.rows; y++) {
                for (int x = 0; x < component_mask.cols; x++) {
                    if (component_mask.at<uchar>(y, x) > 0) {
                        skeleton_set.insert(cv::Point(x, y));
                    }
                }
            }

            // class for the node_to_expand datastructure
            struct expand_class {
                int count = 0;          
                cv::Point point;        
            };

            // none point
            cv::Point none(-1, -1);
            
            // Graph structure: map from point to node data
            std::unordered_map<cv::Point, Ex_TreeNode_Skel, PointHash> out_graph;

            // nodes to expand in the skeletonization
            std::unordered_map<cv::Point, expand_class, PointHash> node_to_expand;

            // nodes visited in the graph
            std::unordered_set<cv::Point, PointHash> visited;

            // kickstart
            // node to expand
            expand_class e_root;
            e_root.point = root;
            node_to_expand[root] = e_root;

            // visited
            visited.insert(root);

            // out_graph
            Ex_TreeNode_Skel root_node;
            root_node.ex_id = root;
            root_node.ex_parent = none; 
            out_graph[root] = root_node;

            while(!node_to_expand.empty())
            {   
                // extract all nodes of current iteration
                std::vector<cv::Point> current_nodes;
                for (const auto& [key, value] : node_to_expand) {
                    current_nodes.push_back(key);
                }
                
                for(size_t i = 0; i < current_nodes.size(); i++)
                {   
                    cv::Point processing_node = current_nodes[i];
                    
                    // get count and parent of processing node
                    auto& exp_data = node_to_expand[processing_node];
                    int count = exp_data.count;
                    cv::Point parent = exp_data.point;  
                    
                    // get all unvisited neighbors
                    std::vector<cv::Point> neighbors;
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (dx == 0 && dy == 0) continue;

                            cv::Point pt(processing_node.x + dx, processing_node.y + dy);
                            if (skeleton_set.find(pt) != skeleton_set.end() && visited.find(pt) == visited.end())
                            {   
                                neighbors.emplace_back(pt);
                            }
                        }
                    }
                    
                    // 45 degrees path special handling: basically we keep the 45 degree connection only when it is the only connection path between the two pixels, if there is a 90 degree path 
                    // (N-S, E-W) that connect the two pixels we suppress the 45 degree paths, avoiding parallel branches with multiple tips on long 45 degrees branches. also right now we check 
                    // such a 90 degree path on both unvisited and visited nodes, this sounds but check this if you have problems in the future. 
                    // for all the neighbors
                    for (size_t i = 0; i < neighbors.size(); ) {
                        // if it is a 45 degrees connection
                        double dx = neighbors[i].x - processing_node.x;
                        double dy = neighbors[i].y - processing_node.y;

                        double dx_abs = std::abs(dx);
                        double dy_abs = std::abs(dy);
                        bool deg45_f = (dx_abs > 0) && (dy_abs > 0);

                        // and there is a 90 degree pah from the current node to the neighbor
                        cv::Point ptx(processing_node.x + dx, processing_node.y);
                        cv::Point pty(processing_node.x, processing_node.y + dy);
                        bool deg90_f = (skeleton_set.find(ptx) != skeleton_set.end()) || (skeleton_set.find(pty) != skeleton_set.end());

                        if (deg45_f && deg90_f) {
                            neighbors.erase(neighbors.begin() + i);
                            // do NOT increment i, because erase shifts elements down
                        } else {
                            ++i;
                        }
                    }

                    if(neighbors.empty()) // tip node
                    {   
                        // add node to graph
                        Ex_TreeNode_Skel node;
                        node.ex_id = processing_node;
                        node.ex_parent = parent; 
                        out_graph[processing_node] = node;
                        // check parent - son relation
                        if((parent != none) && (out_graph.find(processing_node) != out_graph.end()) && (std::find(out_graph[parent].ex_sons.begin(), out_graph[parent].ex_sons.end(), processing_node) == out_graph[parent].ex_sons.end())) // if the parent is not none and the parent node is in the graph and has not the processing node as son
                        {
                            out_graph[parent].ex_sons.push_back(processing_node);
                        }
                    }
                    else
                    {
                        //Track if this node should be added to parent's sons (only once per node)
                        bool node_added_to_parent = false;
                        for(const auto& neighbor : neighbors)
                        {   
                            visited.insert(neighbor);
                            if(neighbors.size() > 1) // branch point
                            {   
                                if(!node_added_to_parent)
                                {   
                                    // add node to graph
                                    Ex_TreeNode_Skel node;
                                    node.ex_id = processing_node;
                                    node.ex_parent = parent; 
                                    out_graph[processing_node] = node;
                                    
                                    // check parent - son relation
                                    if((parent != none) && (out_graph.find(processing_node) != out_graph.end()) && (std::find(out_graph[parent].ex_sons.begin(), out_graph[parent].ex_sons.end(), processing_node) == out_graph[parent].ex_sons.end())) // if the parent is not node and the parent node is in the graph and has not the processing node as son
                                    {   
                                        out_graph[parent].ex_sons.push_back(processing_node);
                                    }
                                    node_added_to_parent = true;
                                }
                                
                                expand_class e;
                                e.point = processing_node;
                                node_to_expand[neighbor] = e;
                            }
                            else // single path
                            {   
                                if(count >= every_k_point) // downsample
                                {
                                    if(!node_added_to_parent)
                                    {   
                                        // add node to graph
                                        Ex_TreeNode_Skel node;
                                        node.ex_id = processing_node;
                                        node.ex_parent = parent; 
                                        out_graph[processing_node] = node;
                                        
                                        // check parent - son relation
                                        if((parent != none) && (out_graph.find(processing_node) != out_graph.end()) && (std::find(out_graph[parent].ex_sons.begin(), out_graph[parent].ex_sons.end(), processing_node) == out_graph[parent].ex_sons.end())) // if the parent is not node and the parent node is in the graph and has not the processing node as son
                                        {
                                            out_graph[parent].ex_sons.push_back(processing_node);
                                        }
                                        node_added_to_parent = true;
                                    }
                                    expand_class e;
                                    e.point = processing_node;
                                    node_to_expand[neighbor] = e;
                                }
                                else
                                {
                                    expand_class e;
                                    e.point = parent;
                                    e.count = count + 1;
                                    node_to_expand[neighbor] = e;
                                }
                            }
                        }
                    }
                    size_t removed = node_to_expand.erase(processing_node);
                }
            }
            
            return out_graph;
        }

        float get_node_depth(const cv::Mat& depth_image, const cv::Point& node, const cv::Mat& tree_mask, const cv::Mat& inscribed_circles)
        {
            // Radius is the distance at the center point
            float radius = inscribed_circles.at<float>(node);

            // obtain the depth as a gaussian kernel on the circle around the point
            float sigma = std::max(radius / 6.0f, 1.0f); // filter half width = 3 sigma
            int r = static_cast<int>(std::ceil(radius));
            float sumWeights = 0.0f;
            float weightedSum = 0.0f;

            for (int y = node.y - r; y <= node.y + r; ++y)
            {
                for (int x = node.x - r; x <= node.x + r; ++x)
                {
                    if (x < 0 || y < 0 || x >= depth_image.cols || y >= depth_image.rows)
                        continue;

                    float dx = x - node.x;
                    float dy = y - node.y;
                    float d2 = dx*dx + dy*dy;

                    if (d2 > radius * radius)
                        continue;

                    
                    float weight = std::exp(-d2 / (2.0f * sigma * sigma));
                    float value = depth_image.at<float>(y, x);
                    if (!std::isfinite(value) || value <= 0.0f)
                        continue;

                    weightedSum += value * weight;
                    sumWeights += weight;
                }
            }

            float gaussianAverage = (sumWeights > 0.0f) ? weightedSum / sumWeights : depth_image.at<float>(node);
            if(std::isnan(gaussianAverage)){
                return INIT_DEPTH;
            }

            return gaussianAverage;
        }

        std::vector<std::vector<Ex_TreeNode>> skeleton_conversion(const std::vector<std::pair<int, std::unordered_map<cv::Point, Ex_TreeNode_Skel, PointHash>>>& forest_2d, const std::vector<cv::Mat>& tree_masks, const cv::Mat& depth_image, const cv::Mat& color_image)
        {   
            // create an id lookup table
            std::unordered_map<cv::Point, int, PointHash> ids;  // FIXED: Added PointHash
            int id_i = 0;
            for(const auto& [component_id, tree_graph] : forest_2d) 
            {
                for(const auto& [point, node] : tree_graph)
                {
                    ids[point] = id_i;
                    id_i += 1;
                }
            }
            
            // convert forest in Ex_TreeNode format
            std::vector<std::vector<Ex_TreeNode>> forest;
            cv::Point none(-1, -1);  // FIXED: was 'node', should be 'none'
            
            // extract camera info data
            double fx = ref_frame["tree_camera"].k[0];
            double fy = ref_frame["tree_camera"].k[4];
            double cx = ref_frame["tree_camera"].k[2];
            double cy = ref_frame["tree_camera"].k[5];
            
            for (std::size_t i = 0; i < forest_2d.size(); ++i)
            {
                const auto& [component_id, tree_2d] = forest_2d[i];  

                // get the maximum radius inscripted in the mask and centered in the node position
                cv::Mat inscribed_circles;
                cv::distanceTransform(tree_masks[i], inscribed_circles, cv::DIST_L2, 5);

                std::vector<Ex_TreeNode> tree;
                for(const auto& [point, node2d] : tree_2d)
                {
                    // relation sons parent
                    Ex_TreeNode node;
                    node.ex_id = std::to_string(ids[point]);
                    
                    if(node2d.ex_parent == none)
                    {
                        node.ex_parent = "root";
                    }
                    else
                    {
                        node.ex_parent = std::to_string(ids[node2d.ex_parent]);  // FIXED: convert int to string
                    }
                    
                    if(node2d.ex_sons.size() == 0)
                    {
                        node.ex_sons.push_back("tip");
                    }
                    else
                    {
                        for(const auto& s2d : node2d.ex_sons)  // FIXED: was 'sd2', should be 's2d'
                        {
                            node.ex_sons.push_back(std::to_string(ids[s2d]));  // FIXED: convert int to string
                        }
                    }
                    
                    // component
                    node.component = component_id;
                    
                    // position
                    // get the 3d position
                    Eigen::Vector4d p_3d_h;

                    // get the depth as the gaussian average of the points inside the mask around the 2d node point
                    float node_depth = get_node_depth(depth_image, point, tree_masks[i], inscribed_circles);

                    p_3d_h << (point.x - cx) * node_depth / fx, (point.y - cy) * node_depth / fy, node_depth, 1;
                   
                    // project the point in the left camera view
                    Eigen::Vector4d p_3dt_h = T_lcam_tree * p_3d_h;
                    
                    node.x = p_3dt_h[0];  
                    node.y = p_3dt_h[1];  
                    node.z = p_3dt_h[2];
                    
                    // visual fd
                    node.fd_brief = evaluate_visual_fd(color_image, point);
                    
                    // track count
                    node.track_cnt = 1;
                    
                    tree.push_back(node);
                }
                if(tree.size() > 0)
                {
                    forest.push_back(tree);
                }
            }
            return forest;
        }

        std::vector<std::vector<Ex_TreeNode>> skeletonize(const std::vector<cv::Mat>& masks, const cv::Mat& depth_image, const cv::Mat& color_image)
        {   
            std::vector<std::pair<int, std::unordered_map<cv::Point, Ex_TreeNode_Skel, PointHash>>> forest_2d;
            std::vector<cv::Mat> tree_masks;
            std::vector<cv::Mat> img_skeletons;
            img_skeletons.reserve(masks.size());

            for (const auto& mask : masks) {
                cv::Mat skeleton;
                
                // Ensure mask is binary
                cv::Mat binary_mask;
                cv::threshold(mask, binary_mask, 0, 255, cv::THRESH_BINARY);
                
                // Use OpenCV's thinning function
                cv::ximgproc::thinning(binary_mask, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
                
                // get the lowest connected component
                cv::Mat labels;
                int num_components = cv::connectedComponents(skeleton, labels, 8, CV_32S);

                // Find the lowest component
                int lowest_comp_id = -1;
                int lowest_root_alt = 0;
                cv::Point lowest_root(0, 0);

                for (int comp_id = 1; comp_id < num_components; comp_id++) {
                    // find connected component root
                    cv::Mat component_mask = (labels == comp_id);
                    cv::Point root = findRoot(component_mask);
                    if((root.x == -1) && (root.y == -1))
                    {
                        continue;
                    }

                    // get altitude
                    double root_alt = static_cast<double>(root.y);
                    if (root_alt > lowest_root_alt) {
                        lowest_root_alt = root_alt;
                        lowest_comp_id = comp_id;
                        lowest_root = root;
                    }
                }
                if(lowest_comp_id == -1){
                    continue;
                }
                // get 3d downsample skeleton
                cv::Mat lowest_component_mask = (labels == lowest_comp_id);

                std::unordered_map<cv::Point, Ex_TreeNode_Skel, PointHash> tree_2d = skeletonize2d(lowest_root, lowest_component_mask);
                if(tree_2d.size() > 0)
                {
                    forest_2d.push_back(make_pair(lowest_comp_id, tree_2d));
                    tree_masks.push_back(lowest_component_mask);
                }
                
            }
            
            // convert in 3d forest of type Ex_TreeNode
            std::vector<std::vector<Ex_TreeNode>> forest = skeleton_conversion(forest_2d, tree_masks, depth_image, color_image);

            
            return forest;
        }

        cv::Mat draw_forest(const cv::Mat& color_image, const std::vector<cv::Mat>& masks, const std::vector<std::vector<Ex_TreeNode>>& forest)
        {   
            // use the color image in grey scale as background
            cv::Mat gray_image;
            cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
            cv::Mat result;
            cv::cvtColor(gray_image, result, cv::COLOR_GRAY2BGR);
            
            // Generate distinct colors for each mask
            std::vector<cv::Vec3b> colors;
            std::vector<cv::Vec3b> opposite_colors;
            std::vector<cv::Vec3b> dark_opposite_colors;
            if (masks.size() == forest.size()) 
            {
                for (size_t i = 0; i < masks.size(); ++i) 
                {
                    // Generate colors using HSV to ensure they're visually distinct
                    float hue = (i * 360.0f / masks.size());
                    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2, 255, 255)); // OpenCV hue is 0-180
                    cv::Mat bgr;
                    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                    
                    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
                    colors.push_back(color);
                    
                    // Generate opposite color (255 - each component)
                    cv::Vec3b opp_color(255 - color[0], 255 - color[1], 255 - color[2]);
                    opposite_colors.push_back(opp_color);
                    
                    // Generate darker opposite color (scale down by 50%)
                    cv::Vec3b dark_opp_color(
                        (uchar)((255 - color[0]) * 0.5),
                        (uchar)((255 - color[1]) * 0.5),
                        (uchar)((255 - color[2]) * 0.5)
                    );
                    dark_opposite_colors.push_back(dark_opp_color);
                }
            }
            else
            {
                for (size_t i = 0; i < masks.size(); ++i) 
                {
                    // Generate colors using HSV to ensure they're visually distinct
                    float hue = (i * 360.0f / masks.size());
                    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2, 255, 255)); // OpenCV hue is 0-180
                    cv::Mat bgr;
                    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                    
                    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
                    colors.push_back(color);
                    
                }

                for (size_t i = 0; i < forest.size(); ++i) 
                {
                    // Generate colors using HSV to ensure they're visually distinct
                    float hue = (i * 360.0f / forest.size());
                    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2, 255, 255)); // OpenCV hue is 0-180
                    cv::Mat bgr;
                    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                    
                    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
                    opposite_colors.push_back(color);

                    // Generate darker opposite color (scale down by 50%)
                    cv::Vec3b dark_opp_color(
                        (uchar)((255 - color[0]) * 0.5),
                        (uchar)((255 - color[1]) * 0.5),
                        (uchar)((255 - color[2]) * 0.5)
                    );
                    dark_opposite_colors.push_back(dark_opp_color);
                    
                }

            }
            
            // Overlay each mask with its corresponding color
            for (size_t i = 0; i < masks.size(); ++i) {
                if (masks[i].empty()) continue;
                
                // Create a colored overlay for this mask
                cv::Mat colored_mask = cv::Mat::zeros(masks[i].size(), CV_8UC3);
                colored_mask.setTo(colors[i], masks[i]);
                
                // Blend the colored mask with the result image
                // Using alpha blending: result = result * (1 - alpha) + colored_mask * alpha
                double alpha = 0.5; // Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
                cv::addWeighted(result, 1.0, colored_mask, alpha, 0.0, result, -1);
            }

            // draw tree graphs
            Eigen::VectorXd k_vec(9);
            k_vec << ref_frame["tree_camera"].k[0], 0.0, ref_frame["tree_camera"].k[2], 0.0, ref_frame["tree_camera"].k[4], ref_frame["tree_camera"].k[5], 0.0, 0.0, 1.0;
            Eigen::Matrix3d k = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(k_vec.data());

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

            for(size_t i = 0; i < forest.size(); i++)
            {
                // draw edge segments
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
                    Eigen::Vector3d p_uvs = k * p_3dt;
                    Eigen::Vector2d p_img;
                    p_img << p_uvs[0] / p_uvs[2], p_uvs[1] / p_uvs[2];

                    // convert point in cv format
                    cv::Point p_img_cv(static_cast<int>(p_img.x()), static_cast<int>(p_img.y()));

                    // draw also the edge to the sons nodes
                    for (const auto& son : node.ex_sons) // given the sons
                    { 
                        // find the son in the vector
                        auto it = std::find_if(forest[i].begin(), forest[i].end(), 
                        [&son](const Ex_TreeNode& n) { return n.ex_id == son; });

                        if (it != forest[i].end()) // If a matching node is found
                        {  
                            // evaluate the point position in the image 
                            // get the 3d position
                            Eigen::Vector4d p_3d_h_son;
                            p_3d_h_son << it->x, it->y, it->z, 1;

                            // project the point in the tree view
                            Eigen::Vector4d p_3dt_h_son = T_tree_lcam * p_3d_h_son;
                            Eigen::Vector3d p_3dt_son = p_3dt_h_son.block<3, 1>(0, 0);
                            
                            // project in the image plane
                            Eigen::Vector3d p_uvs_son = k * p_3dt_son;
                            Eigen::Vector2d p_img_son;
                            p_img_son << p_uvs_son[0] / p_uvs_son[2], p_uvs_son[1] / p_uvs_son[2];

                            // convert point in cv format
                            cv::Point p_img_cv_son(static_cast<int>(p_img_son.x()), static_cast<int>(p_img_son.y()));

                            // draw a line for the son
                            cv::line(result, p_img_cv, p_img_cv_son, opposite_colors[i], 2, cv::LINE_8);
                        }
                    }
                } 
                // draw node dots
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
                    Eigen::Vector3d p_uvs = k * p_3dt;
                    Eigen::Vector2d p_img;
                    p_img << p_uvs[0] / p_uvs[2], p_uvs[1] / p_uvs[2];

                    // convert point in cv format
                    cv::Point p_img_cv(static_cast<int>(p_img.x()), static_cast<int>(p_img.y()));
                    
                    // draw a circle for the node
                    cv::circle(result, p_img_cv, 3, dark_opposite_colors[i], -1);
                }
            }
            
            return result;
        }

        cv::Mat grey_img_preprocess(cv::Mat& image)
        {
            // Ensure input is grayscale
            CV_Assert(image.type() == CV_8UC1);
            
            cv::Mat result = image.clone();
            
            // Step 1: Calculate 1st and 99th percentile values
            std::vector<uchar> pixels;
            pixels.reserve(image.total());
            
            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    pixels.push_back(image.at<uchar>(i, j));
                }
            }
            
            std::sort(pixels.begin(), pixels.end());
            
            int idx_1st = static_cast<int>(pixels.size() * 0.05);
            int idx_99th = static_cast<int>(pixels.size() * 0.95);
            
            uchar min_val = pixels[idx_1st];
            uchar max_val = pixels[idx_99th];
            
            // Step 2: Clip values to percentile range and rescale to full range
            for (int i = 0; i < result.rows; i++) {
                for (int j = 0; j < result.cols; j++) {
                    uchar val = result.at<uchar>(i, j);
                    
                    // Clip to percentile range
                    if (val < min_val) 
                    {
                        val = min_val;
                    }
                    else if (val > max_val)
                    {                   
                        val = max_val;
                    }
                    else
                    {  
                        continue;
                    }
                    
                    // Rescale to 0-255 range
                    result.at<uchar>(i, j) = static_cast<uchar>(val);
                }
            }
            
            // Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(2.0);
            clahe->setTilesGridSize(cv::Size(8, 8));
            clahe->apply(result, result);
            
            // Step 4: Apply bilateral filter for smoothing
            cv::Mat smoothed;
            cv::bilateralFilter(result, smoothed, 5, 50, 50);
            
            return smoothed;
        }

        void stereo_img_mono_tree_sync_cb(const sensor_msgs::msg::Image::ConstSharedPtr & img0_msg, const sensor_msgs::msg::Image::ConstSharedPtr & imgc_msg, const sensor_msgs::msg::Image::ConstSharedPtr & imgd_msg)
        {   
            // process timestamps
            double time_0 = img0_msg->header.stamp.sec + img0_msg->header.stamp.nanosec * (1e-9);
            double time_c = imgc_msg->header.stamp.sec + imgc_msg->header.stamp.nanosec * (1e-9);
            double time_d = imgd_msg->header.stamp.sec + imgd_msg->header.stamp.nanosec * (1e-9);
            std::vector<double> time_vec = {time_0, time_c, time_d};

            // check for big delays
            // Get minimum and maximum value
            double max_val = *std::max_element(time_vec.begin(), time_vec.end());
            double min_val = *std::min_element(time_vec.begin(), time_vec.end());
            if((max_val - min_val) > 0.01)
            {
                double dt = max_val - min_val;
                ROS_WARN("big delay between stereocamera images and tree images: %f sec", dt);
            }

            // evaluate mean time
            double sum_t = std::accumulate(time_vec.begin(), time_vec.end(), 0.0);
            double avg_t = sum_t / time_vec.size();

            if((avg_t - last_msg_t) >= DOWNSAMPLE_P) // if the time distance between the last message retain and this one is bigger then the threshold, process the message
            {
                // reassign last_msg_t for future iterations
                last_msg_t = avg_t;
                // process stereo images
                // convert the messages in images
                cv::Mat image0;
                image0 = getImageFromMsg(img0_msg, true).clone();

                // process tree
                // convert the messages in images
                cv::Mat color_image, depth_image;
                color_image = getImageFromMsg(imgc_msg, false).clone();
                depth_image = convertDepthImage(imgd_msg).clone();

                // input data in the buffers
                // stereo images
                {
                    std::lock_guard<std::mutex> lock(m_image_buf);
                    images_input_buf.push(make_pair(avg_t, make_pair(image0, cv::Mat())));
                }

                // tree images
                {
                    std::lock_guard<std::mutex> lock(m_tree_buf);
                    tree_input_buf.push(make_pair(avg_t, make_pair(color_image, depth_image)));
                }
            }
            // else don't do anything and return

            return;
        }
        
        void stereo_img_tree_sync_cb(const sensor_msgs::msg::Image::ConstSharedPtr & img0_msg, const sensor_msgs::msg::Image::ConstSharedPtr & img1_msg, const sensor_msgs::msg::Image::ConstSharedPtr & imgc_msg, const sensor_msgs::msg::Image::ConstSharedPtr & imgd_msg)
        {   
            // process timestamps
            double time_0 = img0_msg->header.stamp.sec + img0_msg->header.stamp.nanosec * (1e-9);
            double time_1 = img1_msg->header.stamp.sec + img1_msg->header.stamp.nanosec * (1e-9);
            double time_c = imgc_msg->header.stamp.sec + imgc_msg->header.stamp.nanosec * (1e-9);
            double time_d = imgd_msg->header.stamp.sec + imgd_msg->header.stamp.nanosec * (1e-9);
            std::vector<double> time_vec = {time_0, time_1, time_c, time_d};

            // check for big delays
            // Get minimum and maximum value
            double max_val = *std::max_element(time_vec.begin(), time_vec.end());
            double min_val = *std::min_element(time_vec.begin(), time_vec.end());
            if((max_val - min_val) > 0.01)
            {
                double dt = max_val - min_val;
                ROS_WARN("big delay between stereocamera images and tree images: %f sec", dt);
            }

            // evaluate mean time
            double sum_t = std::accumulate(time_vec.begin(), time_vec.end(), 0.0);
            double avg_t = sum_t / time_vec.size();

            if((avg_t - last_msg_t) >= DOWNSAMPLE_P) // if the time distance between the last message retain and this one is bigger then the threshold, process the message
            {
                // reassign last_msg_t for future iterations
                last_msg_t = avg_t;

                // process stereo images
                // convert the messages in images
                cv::Mat image0, image1;
                image0 = getImageFromMsg(img0_msg, true).clone();
                image1 = getImageFromMsg(img1_msg, true).clone();

                // process tree
                // convert the messages in images
                cv::Mat color_image, depth_image;
                color_image = getImageFromMsg(imgc_msg, false).clone();
                depth_image = convertDepthImage(imgd_msg).clone();

                // input data in the buffers
                // stereo images
                {
                    std::lock_guard<std::mutex> lock(m_image_buf);
                    images_input_buf.push(make_pair(avg_t, make_pair(image0, image1)));
                }

                // tree images
                {
                    std::lock_guard<std::mutex> lock(m_tree_buf);
                    tree_input_buf.push(make_pair(avg_t, make_pair(color_image, depth_image)));
                }
            }
            
            // else don't do anything and return

            return;
        }

        void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
        {
            // std::cout << "IMU cb" << std::endl;
            if(USE_TOPIC)
            {   
                if(USE_TREE){
                    if((NUM_OF_CAM + 1 - cam_info_set_flag) == 0)
                    {   
                        double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
                        double dx = imu_msg->linear_acceleration.x;
                        double dy = imu_msg->linear_acceleration.y;
                        double dz = imu_msg->linear_acceleration.z;
                        double rx = imu_msg->angular_velocity.x;
                        double ry = imu_msg->angular_velocity.y;
                        double rz = imu_msg->angular_velocity.z;
                        
                        Vector3d acc, gyr;
                        if(IMU_FILTER)
                        {
                            Vector3d acc_new(dx, dy, dz);
                            Vector3d gyr_new(rx, ry, rz);

                            if(acc_filter_f) // if it's the first sample
                            {
                                acc = acc_new;
                                gyr = gyr_new;

                                acc_filter_f = false;
                            }
                            else
                            {
                                acc = IMU_FILTER_ALPHA * acc_new + (1 - IMU_FILTER_ALPHA) * acc_old;
                                gyr = IMU_FILTER_ALPHA * gyr_new + (1 - IMU_FILTER_ALPHA) * gyr_old;
                            }

                            acc_old = acc;
                            gyr_old = gyr;
                        }
                        else
                        {
                            acc << dx, dy, dz;
                            gyr << rx, ry, rz;
                        }

                        // std::cout << "got t_imu: " << std::fixed << t << endl;
                        estimator.inputIMU(t, acc, gyr);
                        return;
                    }
                }
                else
                {
                    if((NUM_OF_CAM - cam_info_set_flag) == 0)
                    {
                        double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
                        double dx = imu_msg->linear_acceleration.x;
                        double dy = imu_msg->linear_acceleration.y;
                        double dz = imu_msg->linear_acceleration.z;
                        double rx = imu_msg->angular_velocity.x;
                        double ry = imu_msg->angular_velocity.y;
                        double rz = imu_msg->angular_velocity.z;
                        
                        Vector3d acc, gyr;
                        if(IMU_FILTER)
                        {
                            Vector3d acc_new(dx, dy, dz);
                            Vector3d gyr_new(rx, ry, rz);

                            if(acc_filter_f) // if it's the first sample
                            {
                                acc = acc_new;
                                gyr = gyr_new;

                                acc_filter_f = false;
                            }
                            else
                            {
                                acc = IMU_FILTER_ALPHA * acc_new + (1 - IMU_FILTER_ALPHA) * acc_old;
                                gyr = IMU_FILTER_ALPHA * gyr_new + (1 - IMU_FILTER_ALPHA) * gyr_old;
                            }

                            acc_old = acc;
                            gyr_old = gyr;
                        }
                        else
                        {
                            acc << dx, dy, dz;
                            gyr << rx, ry, rz;
                        }

                        // std::cout << "got t_imu: " << std::fixed << t << endl;
                        estimator.inputIMU(t, acc, gyr);
                        return;
                    }
                }
            }
            else
            {
                double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * (1e-9);
                double dx = imu_msg->linear_acceleration.x;
                double dy = imu_msg->linear_acceleration.y;
                double dz = imu_msg->linear_acceleration.z;
                double rx = imu_msg->angular_velocity.x;
                double ry = imu_msg->angular_velocity.y;
                double rz = imu_msg->angular_velocity.z;
                
                Vector3d acc, gyr;
                if(IMU_FILTER)
                {
                    Vector3d acc_new(dx, dy, dz);
                    Vector3d gyr_new(rx, ry, rz);

                    if(acc_filter_f) // if it's the first sample
                    {
                        acc = acc_new;
                        gyr = gyr_new;

                        acc_filter_f = false;
                    }
                    else
                    {
                        acc = IMU_FILTER_ALPHA * acc_new + (1 - IMU_FILTER_ALPHA) * acc_old;
                        gyr = IMU_FILTER_ALPHA * gyr_new + (1 - IMU_FILTER_ALPHA) * gyr_old;
                    }
                    acc_old = acc;
                    gyr_old = gyr;
                }
                else
                {
                    acc << dx, dy, dz;
                    gyr << rx, ry, rz;
                }

                // std::cout << "got t_imu: " << std::fixed << t << endl;
                estimator.inputIMU(t, acc, gyr);
                return;
            }
        }

        void stereo_img_sync_cb(const sensor_msgs::msg::Image::ConstSharedPtr & img0_msg, const sensor_msgs::msg::Image::ConstSharedPtr & img1_msg)
        {   
            // process timestamps
            double time_0 = img0_msg->header.stamp.sec + img0_msg->header.stamp.nanosec * (1e-9);
            double time_1 = img1_msg->header.stamp.sec + img1_msg->header.stamp.nanosec * (1e-9);
            std::vector<double> time_vec = {time_0, time_1};

            // check for big delays
            // Get minimum and maximum value
            double max_val = *std::max_element(time_vec.begin(), time_vec.end());
            double min_val = *std::min_element(time_vec.begin(), time_vec.end());
            if((max_val - min_val) > 0.01)
            {
                double dt = max_val - min_val;
                ROS_WARN("big delay between stereocamera images: %f sec", dt);
            }

            // evaluate mean time
            double sum_t = std::accumulate(time_vec.begin(), time_vec.end(), 0.0);
            double avg_t = sum_t / time_vec.size();

            // process stereo images
            // convert the messages in images
            cv::Mat image0, image1;
            image0 = getImageFromMsg(img0_msg, true).clone();
            image1 = getImageFromMsg(img1_msg, true).clone();

            // input data in the buffers
            // stereo images
            {
                std::lock_guard<std::mutex> lock(m_image_buf);
                images_input_buf.push(make_pair(avg_t, make_pair(image0, image1)));
            }


            return;
        }

        void img0_only_callback(const sensor_msgs::msg::Image::SharedPtr img0_msg)
        {
            // process timestamps
            double time = img0_msg->header.stamp.sec + img0_msg->header.stamp.nanosec * (1e-9);

            // process stereo images
            // convert the messages in images
            cv::Mat image0;
            image0 = getImageFromMsg(img0_msg, true).clone();

            // input data in the buffers
            // stereo images
            {
                std::lock_guard<std::mutex> lock(m_image_buf);
                images_input_buf.push(make_pair(time, make_pair(image0, cv::Mat())));
            }

            return;
        }

        void img0_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr img0_info_msg)
        {   
            if(cam0_info_ft)
            {                  
                // set extrinsic parameters
                std::string from_frame = IMU_FRAME;
                std::string to_frame = img0_info_msg->header.frame_id; 
                try 
                {   
                    geometry_msgs::msg::TransformStamped t;
                    t = tf_buffer_->lookupTransform(from_frame, to_frame, tf2::TimePointZero);

                    // Convert TransformStamped to Eigen::Isometry3d
                    Eigen::Isometry3d eigen_transform = tf2::transformToEigen(t.transform);
                    
                    // Convert to 4x4 matrix
                    Eigen::Matrix4d T = eigen_transform.matrix();
                    
                    RIC.push_back(T.block<3, 3>(0, 0));
                    TIC.push_back(T.block<3, 1>(0, 3));

                }
                catch (const tf2::TransformException & ex) 
                {
                    RCLCPP_INFO(
                      this->get_logger(), "Could not transform %s to %s: %s",
                      to_frame.c_str(), from_frame.c_str(), ex.what());
                    return;
                }
                
                // save reference frame of the left camera to obtain all the points (among vins and the tree plug i in the same reference frame)
                
                ref_frame["left_camera"] = *img0_info_msg;
                
                // set variable for next iterations
                cam0_info_ft = false; // flag used to get inside this callback only at the beginning
                cam_info_set_flag += 1; // counter used to free the image topics once all the camera informations has been acquired

                if(NUM_OF_CAM == 1)
                {
                    // set intrinsic parameters
                    estimator.setParameter_topic();
                }

            }
        }

        void img1_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr img1_info_msg)
        {   
            if(cam1_info_ft && !cam0_info_ft) // to ensure that we process first the informations of cam0 
            {   
                // set extrinsic parameters
                std::string from_frame = IMU_FRAME;
                std::string to_frame = img1_info_msg->header.frame_id; 
                try 
                {
                    geometry_msgs::msg::TransformStamped t;
                    
                    t = tf_buffer_->lookupTransform(from_frame, to_frame, tf2::TimePointZero);

                    // Convert TransformStamped to Eigen::Isometry3d
                    Eigen::Isometry3d eigen_transform = tf2::transformToEigen(t.transform);

                    // Convert to 4x4 matrix
                    Eigen::Matrix4d T = eigen_transform.matrix();
                    
                    RIC.push_back(T.block<3, 3>(0, 0));
                    TIC.push_back(T.block<3, 1>(0, 3));

                }
                catch (const tf2::TransformException & ex) 
                {
                    RCLCPP_INFO(
                      this->get_logger(), "Could not transform %s to %s: %s",
                      to_frame.c_str(), from_frame.c_str(), ex.what());
                    return;
                }

                // save reference frame of the left camera to obtain all the points (among vins and the tree plug i in the same reference frame)
                ref_frame["right_camera"] = *img1_info_msg;

                // set variable for next iterations
                cam1_info_ft = false; // flag used to get inside this callback only at the beginning
                cam_info_set_flag += 1; // counter used to free the image topics once all the camera informations has been acquired

                // set intrinsic parameters
                estimator.setParameter_topic();

            }
        }
        
        void tree_imgd_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr tree_imgd_info_msg)
        {   
            if(cam_tree_info_ft)
            {   
                // save the camera info relative to the depth image
                ref_frame["tree_camera"] = *tree_imgd_info_msg;
                ref_frame["tree_camera"].header.frame_id = "oakd_rgb_camera_optical_frame1"; // DEBUG only

                // set variable for next iterations
                cam_tree_info_ft = false; // flag to set the tree camera info only once
                cam_info_set_flag += 1; // counter used to free the image/tree topics once all the camera informations has been acquired

                // check if the tree images are already in the left_camera reference frame
                if (ref_frame["tree_camera"].header.frame_id != ref_frame["left_camera"].header.frame_id && !tree_lcam_flag)
                {   
                    // if not get the transformation matrix between the two
                    try 
                    {   
                        // reset the flag to execute it only once
                        tree_lcam_flag = true;

                        // init the transformation matrix to the identity quaternion (no rotation or translation)
                        geometry_msgs::msg::TransformStamped t_;

                        t_ = tf_buffer_->lookupTransform(ref_frame["left_camera"].header.frame_id, ref_frame["tree_camera"].header.frame_id, tf2::TimePointZero);

                        // Convert TransformStamped to Eigen::Isometry3d
                        Eigen::Isometry3d eigen_transform = tf2::transformToEigen(t_.transform);

                        // Convert to 4x4 matrix
                        T_lcam_tree = eigen_transform.matrix();
                    }
                    catch (const tf2::TransformException & ex) 
                    {
                        RCLCPP_INFO(
                        this->get_logger(), "Could not transform %s to %s: %s",
                        ref_frame["left_camera"].header.frame_id, ref_frame["tree_camera"].header.frame_id, ex.what());
                        return;
                    }
                }

                // in any case get the vertical unit vector needed for the root normal in the ICP point to plane case
                if(!tree_baselink_flag)
                {
                    try 
                    {   
                        // reset the flag to execute it only once
                        tree_baselink_flag = true;

                        // init the transformation matrix to the identity quaternion (no rotation or translation)
                        geometry_msgs::msg::TransformStamped t_;

                        t_ = tf_buffer_->lookupTransform(ref_frame["left_camera"].header.frame_id, "base_link", tf2::TimePointZero);

                        // Convert TransformStamped to Eigen::Isometry3d
                        Eigen::Isometry3d eigen_transform = tf2::transformToEigen(t_.transform);

                        // Convert to 4x4 matrix
                        Eigen::Matrix4d T_tree_bodylink = eigen_transform.matrix();
                        Eigen::Matrix3d R_tree_bodylink = T_tree_bodylink.block<3,3>(0,0);
                        Eigen::Vector3d z(0, 0, 1);
                        base_link_z = R_tree_bodylink * z;
                        R_bodylink_tree = R_tree_bodylink.transpose();
                    }
                    catch (const tf2::TransformException & ex) 
                    {
                        RCLCPP_INFO(
                        this->get_logger(), "Could not transform %s to %s: %s",
                        ref_frame["left_camera"].header.frame_id, "base_link", ex.what());
                        return;
                    }
                }
            }
        }

        void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
        {
            if (restart_msg->data == true)
            {
                ROS_WARN("restart the estimator!");
                estimator.clearState();
                estimator.setParameter();
            }
            return;
        }

        void imu_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
        {
            if (switch_msg->data == true)
            {
                //ROS_WARN("use IMU!");
                estimator.changeSensorType(1, STEREO);
            }
            else
            {
                //ROS_WARN("disable IMU!");
                estimator.changeSensorType(0, STEREO);
            }
            return;
        }

        void cam_switch_callback(const std_msgs::msg::Bool::SharedPtr switch_msg)
        {
            if (switch_msg->data == true)
            {
                //ROS_WARN("use stereo!");
                estimator.changeSensorType(USE_IMU, 1);
            }
            else
            {
                //ROS_WARN("use mono camera (left)!");
                estimator.changeSensorType(USE_IMU, 0);
            }
            return;
        }

        cv::Mat convertDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr& imgd_msg) 
        {
            // Use cv_bridge directly for depth images
            cv_bridge::CvImageConstPtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvShare(imgd_msg, imgd_msg->encoding);
            } catch (cv_bridge::Exception& e) {
                throw std::runtime_error("cv_bridge exception: " + std::string(e.what()));
            }
            
            cv::Mat depth_float;
            
            // Determine scale factor
            double depth_scale = 1.0;
            if (imgd_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                // 16UC1 is typically millimeters, convert to meters
                depth_scale = 1000.0;
            }
            
            // Override with custom scale if provided
            if (TREE_DPP_DEPTH_SCALE > 0) {
                depth_scale = TREE_DPP_DEPTH_SCALE;
            }
            
            // Convert to float32 in meters
            if (cv_ptr->image.type() == CV_16UC1) {
                cv_ptr->image.convertTo(depth_float, CV_32F, 1.0 / depth_scale);
            } 
            else if (cv_ptr->image.type() == CV_32FC1) {
                // Assume already in correct units
                depth_float = cv_ptr->image.clone();
            } 
            else {
                throw std::runtime_error("Unsupported depth image type");
            }
            
            return depth_float;
        }

        cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstPtr &img_msg, bool gray)
        {   
            cv::Mat gray_image;
            cv_bridge::CvImagePtr ptr;
            if (img_msg->encoding == "bgr8")
            {   
                ptr = cv_bridge::toCvCopy(img_msg, img_msg->encoding);
                if(gray)
                {
                    cv::cvtColor(ptr->image, gray_image, cv::COLOR_BGR2GRAY);
                    return gray_image;
                }
                else
                {
                    return ptr->image.clone();
                }
            }
            else
            {
                ptr = cv_bridge::toCvCopy(img_msg, img_msg->encoding);
                gray_image = ptr->image.clone();
            }
            
            return gray_image;
        }
    
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto V_node = std::make_shared<VinsNode>();
  rclcpp::spin(V_node);
  rclcpp::shutdown();

  return 0;
}