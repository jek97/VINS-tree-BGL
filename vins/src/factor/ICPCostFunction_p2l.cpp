#include "ICPCostFunction_p2l.h"
// ==============================================================================
// ICPCostFunction_p2l Implementation - MODIFIED VERSION
// Reprojects anchor point to current frame for error computation
// WITH DEPTH OPTIMIZATION SUPPORT
// ==============================================================================

ICPCostFunction_p2l::ICPCostFunction_p2l(
    const Eigen::Vector3d& _pts_i,
    const Eigen::Vector3d& _velocity_i,
    double _td_i,
    const Eigen::Vector3d& _normal_i,
    const Eigen::Vector3d& _pts_j,
    const Eigen::Vector3d& _velocity_j,
    double _td_j)
    : _pts_i_(_pts_i.normalized()),  // Store normalized direction vector
      _velocity_i_(_velocity_i), 
      _td_i_(_td_i),            
      _normal_i_(_normal_i.normalized()), 
      _pts_j_(_pts_j),          
      _velocity_j_(_velocity_j), 
      _td_j_(_td_j) {           
    
    // Debug output for velocity magnitudes (can be removed in production)
    double anchor_vel_magnitude = _velocity_i_.norm();
    double current_vel_magnitude = _velocity_j_.norm();
    
    // if (anchor_vel_magnitude > 50.0 || current_vel_magnitude > 50.0) {
    //     std::cerr << "Warning: High 3D velocity detected - "
    //               << "Anchor: " << anchor_vel_magnitude << " m/s, "
    //               << "Current: " << current_vel_magnitude << " m/s" << std::endl;
    // }
}

template <typename T>
Eigen::Matrix<T, 4, 4> ICPCostFunction_p2l::poseToMatrix(const T* pose) const {
    // VINS pose parameterization: [px, py, pz, qx, qy, qz, qw]
    Eigen::Matrix<T, 3, 1> translation(pose[0], pose[1], pose[2]);
    Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]); // w, x, y, z
    q.normalize();
    
    Eigen::Matrix<T, 4, 4> T_matrix = Eigen::Matrix<T, 4, 4>::Identity();
    T_matrix.block(0, 0, 3, 3) = q.toRotationMatrix();
    T_matrix.block(0, 3, 3, 1) = translation;
    
    return T_matrix;
}

template <typename T>
bool ICPCostFunction_p2l::operator()(
    const T* const para_Pose_i,
    const T* const para_Pose_j,
    const T* const para_Ex_Pose_0,
    const T* const para_depth,
    const T* const para_Td,
    T* residuals) const {
    
    const double weight = 1.0;
    
    // Extract optimized depth
    T depth = para_depth[0];
    
    // Reconstruct 3D point from normalized direction and depth
    Eigen::Matrix<T, 3, 1> pts_i_3d;
    pts_i_3d(0) = T(_pts_i_(0)) * depth;
    pts_i_3d(1) = T(_pts_i_(1)) * depth;
    pts_i_3d(2) = T(_pts_i_(2)) * depth;
    
    // Extract optimized time offset
    T td = para_Td[0];
    
    // Apply time delay correction to anchor point using 3D velocity
    T anchor_td_total = T(_td_i_) + td;
    T current_td_total = T(_td_j_) + td;
    
    // Time-corrected anchor point in camera frame i
    Eigen::Matrix<T, 3, 1> anchor_point_corrected;
    anchor_point_corrected(0) = pts_i_3d(0) + T(_velocity_i_(0)) * anchor_td_total;
    anchor_point_corrected(1) = pts_i_3d(1) + T(_velocity_i_(1)) * anchor_td_total;
    anchor_point_corrected(2) = pts_i_3d(2) + T(_velocity_i_(2)) * anchor_td_total;
    
    // Time-corrected current point in camera frame j
    Eigen::Matrix<T, 3, 1> current_point_corrected;
    current_point_corrected(0) = T(_pts_j_(0)) + T(_velocity_j_(0)) * current_td_total;
    current_point_corrected(1) = T(_pts_j_(1)) + T(_velocity_j_(1)) * current_td_total;
    current_point_corrected(2) = T(_pts_j_(2)) + T(_velocity_j_(2)) * current_td_total;
    
    // Get transformation matrices
    Eigen::Matrix<T, 4, 4> T_wb_i = poseToMatrix(para_Pose_i);  // world to body i
    Eigen::Matrix<T, 4, 4> T_wb_j = poseToMatrix(para_Pose_j);  // world to body j
    Eigen::Matrix<T, 4, 4> T_bc = poseToMatrix(para_Ex_Pose_0); // body to left camera
    
    // Transform corrected anchor point from camera frame i to world frame
    // Path: cam_i -> body_i -> world
    Eigen::Matrix<T, 4, 1> anchor_point_cam_h;
    anchor_point_cam_h << anchor_point_corrected, T(1);
    Eigen::Matrix<T, 4, 1> anchor_point_body_i = T_bc * anchor_point_cam_h;
    Eigen::Matrix<T, 4, 1> anchor_point_world = T_wb_i * anchor_point_body_i;
    
    // Reproject anchor point from world frame to camera frame j
    // Path: world -> body_j -> cam_j
    // Extract rotation and translation from T_wb_j
    Eigen::Matrix<T, 3, 3> R_wb_j = T_wb_j.template block<3,3>(0,0);  // rotation
    Eigen::Matrix<T, 3, 1> t_wb_j = T_wb_j.template block<3,1>(0,3);  // translation

    // Compute inverse
    Eigen::Matrix<T, 3, 3> R_bw_j = R_wb_j.transpose();         // inverse rotation
    Eigen::Matrix<T, 3, 1> t_bw_j = -R_bw_j * t_wb_j;             // inverse translation

    // Extract rotation and translation from T_bc
    Eigen::Matrix<T, 3, 3> R_bc = T_bc.template block<3,3>(0,0);  // rotation
    Eigen::Matrix<T, 3, 1> t_bc = T_bc.template block<3,1>(0,3);  // translation

    // Compute inverse
    Eigen::Matrix<T, 3, 3> R_cb = R_bc.transpose();         // inverse rotation
    Eigen::Matrix<T, 3, 1> t_cb = -R_cb * t_bc;             // inverse translation

    // Rebuild the inverse transformation matrix
    Eigen::Matrix<T, 4, 4> T_cb = Eigen::Matrix<T, 4, 4>::Identity();
    T_cb.template block<3,3>(0,0) = R_cb;
    T_cb.template block<3,1>(0,3) = t_cb;

    // Rebuild the inverse transformation matrix
    Eigen::Matrix<T, 4, 4> T_bw_j = Eigen::Matrix<T, 4, 4>::Identity();
    T_bw_j.template block<3,3>(0,0) = R_bw_j;
    T_bw_j.template block<3,1>(0,3) = t_bw_j;
    
    Eigen::Matrix<T, 4, 1> anchor_point_body_j = T_bw_j * anchor_point_world;
    Eigen::Matrix<T, 4, 1> anchor_point_cam_j = T_cb * anchor_point_body_j;
    
    // Extract 3D coordinates of reprojected anchor point in camera frame j
    Eigen::Matrix<T, 3, 1> anchor_point_reprojected = anchor_point_cam_j.head(3);
    
    // Transform anchor normal from camera frame i to camera frame j
    // Path: cam_i -> body_i -> world -> body_j -> cam_j
    Eigen::Matrix<T, 3, 3> R_wb_i = T_wb_i.block(0, 0, 3, 3);
    
    Eigen::Matrix<T, 3, 1> normal_cam_i(T(_normal_i_(0)), T(_normal_i_(1)), T(_normal_i_(2)));
    Eigen::Matrix<T, 3, 1> normal_body_i = R_bc * normal_cam_i;
    Eigen::Matrix<T, 3, 1> normal_world = R_wb_i * normal_body_i;
    Eigen::Matrix<T, 3, 1> normal_body_j = R_bw_j * normal_world;
    Eigen::Matrix<T, 3, 1> normal_cam_j = R_cb * normal_body_j;
    normal_cam_j = normal_cam_j.normalized();
    
    // Compute vector between points in camera frame j
    Eigen::Matrix<T, 3, 1> point_diff = current_point_corrected - anchor_point_reprojected;
    
    // Project the difference onto the plane normal to the transformed normal
    // projected_diff = point_diff - (point_diff · normal) * normal
    T dot_product = point_diff.dot(normal_cam_j);
    Eigen::Matrix<T, 3, 1> projection_on_normal = dot_product * normal_cam_j;
    Eigen::Matrix<T, 3, 1> projected_diff = point_diff - projection_on_normal;
    
    // Vector residual (3 components) in camera frame j coordinates
    residuals[0] = weight * projected_diff(0);  // x-component in camera frame j
    residuals[1] = weight * projected_diff(1);  // y-component in camera frame j
    residuals[2] = weight * projected_diff(2);  // z-component in camera frame j
    
    return true;
}

ceres::CostFunction* ICPCostFunction_p2l::Create(
    const Eigen::Vector3d& _pts_i,
    const Eigen::Vector3d& _velocity_i,
    double _td_i,
    const Eigen::Vector3d& _normal_i,
    const Eigen::Vector3d& _pts_j,
    const Eigen::Vector3d& _velocity_j,
    double _td_j) {
    
    // Updated: Added depth parameter (size 1) between extrinsics and time delay
    return new ceres::AutoDiffCostFunction<ICPCostFunction_p2l, 3, 7, 7, 7, 1, 1>(
        new ICPCostFunction_p2l(
            _pts_i, _velocity_i, _td_i, _normal_i,  // anchor frame parameters
            _pts_j, _velocity_j, _td_j              // current frame parameters
        )
    );
}