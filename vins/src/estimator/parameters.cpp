/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int IMU_FILTER;
double IMU_FILTER_ALPHA;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string IMAGE0_INFO_TOPIC, IMAGE1_INFO_TOPIC, IMU_FRAME;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int USE_TOPIC;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

int USE_TREE;
std::string TREE_COLOR_TOPIC;
std::string TREE_DEPTH_TOPIC;
std::string TREE_DEPTH_INFO_TOPIC;
double DOWNSAMPLE_P;
Eigen::Matrix4d T_lcam_tree;
bool tree_lcam_flag;    
bool tree_baselink_flag;
double TREE_DPP_DEPTH_SCALE;
double TREE_DPP_MIN_D;
double TREE_DPP_MAX_D;
int TREE_DPP_B_FILTER_F;
int TREE_DPP_T_FILTER_F;
double TREE_DPP_T_FILTER_A;
int TREE_DPP_HF_FILTER_F;
std::string YOLO_MODEL_PATH;
float TS_CONFIDENCE_T;
float TS_IOU_T;
double TS_MASK_T;
double H_CLUSTERING_T_SQ;
double DOWN_SKEL;
int MAX_T_CNT;
int TP_FD_LENGHT;
double TREE_METRIC_MATCH_THRESH;
double STAT_OUT_REJ_K;
double MIN_TREE_PARALLAX;
int ICP_P2L;
double TREE_OUTLIERS_TRESH;

std::map<std::string, sensor_msgs::msg::CameraInfo> ref_frame;
Eigen::Vector3d base_link_z;
std::string mahalanobis_filePath;
Eigen::MatrixXd M_dist;


template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

void read_mahalanobis_matrix(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<double>> rows;
    std::string line;

    // Read file line by line
    while (std::getline(infile, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            rows.push_back(row);
        }
    }

    infile.close();

    // Check that all rows have the same length
    if (rows.empty()) {
        throw std::runtime_error("Matrix file is empty.");
    }
    size_t n_cols = rows[0].size();
    for (size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].size() != n_cols) {
            throw std::runtime_error("Inconsistent row sizes in matrix file.");
        }
    }

    // Copy data into Eigen::MatrixXd
    Eigen::MatrixXd M(rows.size(), n_cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            M(i, j) = rows[i][j];
        }
    }

    std::cout << "read mahalnobis distance with shape " << rows.size() << " " << n_cols << std::endl;
    M_dist = M;
    return;
}

sensor_msgs::msg::CameraInfo parseCameraConfig(const std::string& cam_filepath) 
{
    sensor_msgs::msg::CameraInfo cam_info;
            
    try {
        // Load YAML file
        YAML::Node config = YAML::LoadFile(cam_filepath);
                
        // Get model type
        std::string model_type = config["model_type"].as<std::string>();
                
        // Get basic parameters
        cam_info.width = config["image_width"].as<uint32_t>();
        cam_info.height = config["image_height"].as<uint32_t>();
                
        // Initialize matrices
        cam_info.k.fill(0.0);
        cam_info.r.fill(0.0);
        cam_info.p.fill(0.0);
                
        // Set R to identity (no rotation)
        cam_info.r[0] = 1.0;
        cam_info.r[4] = 1.0;
        cam_info.r[8] = 1.0;
                
        if (model_type == "PINHOLE") {
            YAML::Node dist = config["distortion_parameters"];
            if(dist["k6"])
            {
                // Set distortion model
                cam_info.distortion_model = "rational_polynomial";
                        
                // Get distortion parameters - use assignment instead of resize
                cam_info.d = {
                    dist["k1"].as<double>(),
                    dist["k2"].as<double>(),
                    dist["p1"].as<double>(),
                    dist["p2"].as<double>(),
                    dist["k3"].as<double>(),
                    dist["k4"].as<double>(),
                    dist["k5"].as<double>(),
                    dist["k6"].as<double>()
                };
                        
                // Get projection parameters
                YAML::Node proj = config["projection_parameters"];
                double fx = proj["fx"].as<double>();
                double fy = proj["fy"].as<double>();
                double cx = proj["cx"].as<double>();
                double cy = proj["cy"].as<double>();
                        
                // Fill K matrix (intrinsic camera matrix)
                cam_info.k[0] = fx;
                cam_info.k[2] = cx;
                cam_info.k[4] = fy;
                cam_info.k[5] = cy;
                cam_info.k[8] = 1.0;
                        
                // Fill P matrix (projection matrix)
                cam_info.p[0] = fx;
                cam_info.p[2] = cx;
                cam_info.p[5] = fy;
                cam_info.p[6] = cy;
                cam_info.p[10] = 1.0;
            }
            else
            {
                // Set distortion model
                cam_info.distortion_model = "plumb_bob";
                        
                // Get distortion parameters - use assignment instead of resize
                cam_info.d = {
                    dist["k1"].as<double>(),
                    dist["k2"].as<double>(),
                    dist["p1"].as<double>(),
                    dist["p2"].as<double>(),
                    dist["k3"] ? dist["k3"].as<double>() : 0.0
                };
                        
                // Get projection parameters
                YAML::Node proj = config["projection_parameters"];
                double fx = proj["fx"].as<double>();
                double fy = proj["fy"].as<double>();
                double cx = proj["cx"].as<double>();
                double cy = proj["cy"].as<double>();
                        
                // Fill K matrix (intrinsic camera matrix)
                cam_info.k[0] = fx;
                cam_info.k[2] = cx;
                cam_info.k[4] = fy;
                cam_info.k[5] = cy;
                cam_info.k[8] = 1.0;
                        
                // Fill P matrix (projection matrix)
                cam_info.p[0] = fx;
                cam_info.p[2] = cx;
                cam_info.p[5] = fy;
                cam_info.p[6] = cy;
                cam_info.p[10] = 1.0;
            }
                    
        } else if (model_type == "MEI") {
            // MEI (omnidirectional) model
            cam_info.distortion_model = "equidistant"; 
                    
            // Get mirror parameter
            double xi = config["mirror_parameters"]["xi"].as<double>();
                    
            // Get distortion parameters - use assignment instead of resize
            YAML::Node dist = config["distortion_parameters"];
            cam_info.d = {
                dist["k1"].as<double>(),
                dist["k2"].as<double>(),
                dist["p1"].as<double>(),
                dist["p2"].as<double>(),
                xi  // Store xi in d[4]
            };
                    
            // Get projection parameters (MEI uses gamma notation)
            YAML::Node proj = config["projection_parameters"];
            double gamma1 = proj["gamma1"].as<double>();
            double gamma2 = proj["gamma2"].as<double>();
            double u0 = proj["u0"].as<double>();
            double v0 = proj["v0"].as<double>();
                    
            // Fill K matrix
            cam_info.k[0] = gamma1;
            cam_info.k[2] = u0;
            cam_info.k[4] = gamma2;
            cam_info.k[5] = v0;
            cam_info.k[8] = 1.0;
                    
            // Fill P matrix
            cam_info.p[0] = gamma1;
            cam_info.p[2] = u0;
            cam_info.p[5] = gamma2;
            cam_info.p[6] = v0;
            cam_info.p[10] = 1.0;
                    
        } else {
            throw std::runtime_error("Unknown camera model type: " + model_type);
        }
                
        // Set binning (no binning by default)
        cam_info.binning_x = 0;
        cam_info.binning_y = 0;
                
        // Set ROI (full image by default)
        cam_info.roi.x_offset = 0;
        cam_info.roi.y_offset = 0;
        cam_info.roi.height = 0;
        cam_info.roi.width = 0;
        cam_info.roi.do_rectify = false;
                
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to parse camera config file: " + 
                                std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error reading camera config: " + 
                                std::string(e.what()));
    }
            
    return cam_info;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_GPU = fsSettings["use_gpu"];
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];

        fsSettings["imu_filter"] >> IMU_FILTER;
        if(IMU_FILTER)
        {
            fsSettings["imu_filter_alpha"] >> IMU_FILTER_ALPHA;
        }
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    USE_TOPIC = fsSettings["use_topic"];
    if(USE_TOPIC)
    {
        fsSettings["image0_info_topic"] >> IMAGE0_INFO_TOPIC;
        if(NUM_OF_CAM == 2)
        {
            STEREO = 1;
            fsSettings["image1_info_topic"] >> IMAGE1_INFO_TOPIC;
        }
        fsSettings["imu_frame"] >> IMU_FRAME;
    }
    else
    {
        int pn = config_file.find_last_of('/');
        std::string configPath = config_file.substr(0, pn);
        
        std::string cam0Calib;
        fsSettings["cam0_calib"] >> cam0Calib;
        std::string cam0Path = configPath + "/" + cam0Calib;
        CAM_NAMES.push_back(cam0Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));

        std::string cam0_frame;
        fsSettings["image0_frame"] >> cam0_frame;
        ref_frame["left_camera"].header.frame_id = cam0_frame;

        if(NUM_OF_CAM == 2)
        {
            STEREO = 1;
            std::string cam1Calib;
            fsSettings["cam1_calib"] >> cam1Calib;
            std::string cam1Path = configPath + "/" + cam1Calib; 
            //printf("%s cam1 path\n", cam1Path.c_str() );
            CAM_NAMES.push_back(cam1Path);
            
            cv::Mat cv_T;
            fsSettings["body_T_cam1"] >> cv_T;
            Eigen::Matrix4d T;
            cv::cv2eigen(cv_T, T);
            RIC.push_back(T.block<3, 3>(0, 0));
            TIC.push_back(T.block<3, 1>(0, 3));

            std::string cam1_frame;
            fsSettings["image1_frame"] >> cam1_frame;
            ref_frame["right_camera"].header.frame_id = cam1_frame;
        }

        ROW = fsSettings["image_height"];
        COL = fsSettings["image_width"];
        ROS_INFO("ROW: %d COL: %d ", ROW, COL);
    }
    

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;
    
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    // tree informations
    USE_TREE = fsSettings["use_tree"];
    if(USE_TREE)
    {   
        fsSettings["color_topic"] >> TREE_COLOR_TOPIC;
        fsSettings["depth_topic"] >> TREE_DEPTH_TOPIC;
        fsSettings["tree_depth_info_topic"] >> TREE_DEPTH_INFO_TOPIC;
        fsSettings["downsample_period"] >> DOWNSAMPLE_P;

        if(!USE_TOPIC)
        {   
            // set the tree camera intrinsic
            std::string cam_tree_Calib;
            fsSettings["cam_tree_calib"] >> cam_tree_Calib;
            int pn = config_file.find_last_of('/');
            std::string configPath = config_file.substr(0, pn);
            std::string cam_tree_path = configPath + "/" + cam_tree_Calib;
            ref_frame["tree_camera"] = parseCameraConfig(cam_tree_path);
            
            // set the camera extrinsic
            std::string cam_tree_frame = fsSettings["image_tree_frame"];
            ref_frame["tree_camera"].header.frame_id = cam_tree_frame;
            cv::Mat cv_T_lcam_tree;
            fsSettings["cam0_T_camtree"] >> cv_T_lcam_tree;
            cv::cv2eigen(cv_T_lcam_tree, T_lcam_tree);
            tree_lcam_flag = true;

            // get vertical vector (for icp point line)
            cv::Mat cv_T_bodylink_tree;
            fsSettings["cam0_T_baselink"] >> cv_T_bodylink_tree;
            Eigen::Matrix4d T_bodylink_tree;
            cv::cv2eigen(cv_T_bodylink_tree, T_bodylink_tree);
            Eigen::Matrix3d R_bodylink_tree = T_bodylink_tree.block<3,3>(0,0);
            Eigen::Vector3d z(0, 0, 1);
            base_link_z = R_bodylink_tree * z;
            tree_baselink_flag = true;
        }

        fsSettings["depth_scale"] >> TREE_DPP_DEPTH_SCALE;
        fsSettings["min_depth"] >> TREE_DPP_MIN_D;
        fsSettings["max_depth"] >> TREE_DPP_MAX_D;
        fsSettings["bilateral_filter_flag"] >> TREE_DPP_B_FILTER_F;
        fsSettings["temporal_filter_flag"] >> TREE_DPP_T_FILTER_F;
        fsSettings["temporal_alpha"] >> TREE_DPP_T_FILTER_A;
        fsSettings["hole_filling_flag"] >> TREE_DPP_HF_FILTER_F;
        
        // load NN weights
        int pn = config_file.find_last_of('/');
        std::string configPath = config_file.substr(0, pn);
        std::string weights_file = fsSettings["tree_seg_weights_file"];
        std::string rootPath = configPath.substr(0, configPath.find_last_of('/'));
        std::string weightPath = rootPath + "/weights/";
        YOLO_MODEL_PATH = weightPath + weights_file;
        std::filesystem::path modelPath(YOLO_MODEL_PATH);
        
        if(USE_GPU)
        {
            assert(modelPath.extension() == ".engine" && "Expected model file with .engine extension");
        }
        else
        {
            assert(modelPath.extension() == ".onnx" && "Expected model file with .onnx extension");
        }

        
        TS_CONFIDENCE_T = fsSettings["tree_seg_confidence_threshold"];
        TS_IOU_T = fsSettings["tree_seg_iou_threshold"];
        TS_MASK_T = fsSettings["tree_seg_mask_threshold"];
        double h_clustering_t = fsSettings["tree_horizontal_mask_clustering_threshold"]; // Pre-square the threshold to avoid sqrt in distance calculations
        H_CLUSTERING_T_SQ = h_clustering_t * h_clustering_t;
        fsSettings["downsample_skel_metric"] >> DOWN_SKEL;
        MAX_T_CNT = fsSettings["max_t_cnt"];
        TP_FD_LENGHT = fsSettings["topological_fd_lenght"];
        TREE_METRIC_MATCH_THRESH = fsSettings["metric_match_threshold"];
        STAT_OUT_REJ_K = fsSettings["stat_out_rej_k"];
        MIN_TREE_PARALLAX = fsSettings["keyframe_tree_parallax"];
        fsSettings["icp_point_to_line"] >> ICP_P2L;
        fsSettings["outliers_rejection_threshold"] >> TREE_OUTLIERS_TRESH;
        fsSettings["mahalanobis_filepath"] >> mahalanobis_filePath;
        std::string M_filePath = configPath + "/" + mahalanobis_filePath;
        read_mahalanobis_matrix(M_filePath);
    }
    fsSettings.release();
}
