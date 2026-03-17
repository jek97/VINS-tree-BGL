#pragma once

#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cassert>

// ==============================================================================
// ICP POINT-TO-PLANE PROJECTED COST FUNCTION FOR VINS - MODIFIED VERSION
// Compatible with VINS-Fusion/VINS-Mono architecture
// Reprojects anchor point to current frame for error computation
// WITH DEPTH OPTIMIZATION SUPPORT
// ==============================================================================

/**
 * @brief ICP point-to-plane cost function with time offset compensation, 3D velocity, and depth optimization
 * 
 * This cost function implements a point-to-plane ICP constraint that:
 * 1. Reconstructs 3D anchor point from normalized direction vector and optimized depth
 * 2. Compensates anchor point for temporal misalignment using 3D velocity
 * 3. Transforms anchor point from frame i to world coordinates
 * 4. Reprojects anchor point from world to current frame j coordinates
 * 5. Transforms normal vector from frame i to frame j
 * 6. Computes point-to-plane residual in frame j coordinates
 * 7. Optimizes robot poses, depth, and time offset simultaneously
 * 8. Uses VINS-style pose parameterization [px, py, pz, qx, qy, qz, qw]
 * 
 * The residual is the 3D vector of projected differences in camera frame j coordinates.
 * 
 * Transformation pipeline:
 * - Anchor point: normalized_dir * depth ? cam_i (compensated) ? body_i ? world ? body_j ? cam_j
 * - Current point: cam_j (compensated) - already in target frame
 * - Normal vector: cam_i ? body_i ? world ? body_j ? cam_j
 * - Error computation: All performed in cam_j coordinates
 */
struct ICPCostFunction_p2l {
public:
    /**
     * @brief Constructor for ICP cost function with reprojection and depth optimization
     * 
     * @param _pts_i 3D anchor point direction (will be normalized) in left camera frame i
     * @param _velocity_i 3D velocity of anchor point in camera frame i (m/s)
     * @param _td_i Current time offset estimate at anchor frame i (seconds)
     * @param _normal_i Normal vector associated with anchor point in frame i (will be normalized)
     * @param _pts_j 3D current point in left camera frame j (meters)
     * @param _velocity_j 3D velocity of current point in camera frame j (m/s)
     * @param _td_j Current time offset estimate at current frame j (seconds)
     */
    ICPCostFunction_p2l(const Eigen::Vector3d& _pts_i,
                                   const Eigen::Vector3d& _velocity_i,
                                   double _td_i,
                                   const Eigen::Vector3d& _normal_i,
                                   const Eigen::Vector3d& _pts_j,
                                   const Eigen::Vector3d& _velocity_j,
                                   double _td_j);

    /**
     * @brief Cost function evaluation operator with depth optimization and anchor point reprojection
     * 
     * This operator performs the following steps:
     * 1. Reconstruct 3D anchor point from normalized direction and optimized depth
     * 2. Apply velocity compensation to both anchor and current points
     * 3. Transform compensated anchor point: cam_i ? body_i ? world ? body_j ? cam_j
     * 4. Transform normal vector: cam_i ? body_i ? world ? body_j ? cam_j
     * 5. Compute point difference in cam_j coordinates
     * 6. Project difference onto plane perpendicular to transformed normal
     * 7. Return 3D projected residual vector
     * 
     * @param para_Pose_i Anchor frame pose [px, py, pz, qx, qy, qz, qw]
     * @param para_Pose_j Current frame pose [px, py, pz, qx, qy, qz, qw]
     * @param para_Ex_Pose_0 Left camera extrinsics [px, py, pz, qx, qy, qz, qw]
     * @param para_depth Depth parameter [depth] (meters)
     * @param para_Td Time offset parameter [td] (seconds)
     * @param residuals Output residuals (3-component vector in cam_j coordinates)
     * @return true if evaluation successful
     */
    template <typename T>
    bool operator()(const T* const para_Pose_i,
                    const T* const para_Pose_j,
                    const T* const para_Ex_Pose_0,
                    const T* const para_depth,
                    const T* const para_Td,
                    T* residuals) const;

    /**
     * @brief Factory method to create Ceres cost function with reprojection and depth optimization
     * 
     * @param _pts_i 3D anchor point direction in camera frame i (will be normalized)
     * @param _velocity_i 3D velocity of anchor point in frame i
     * @param _td_i Time offset estimate at anchor frame i
     * @param _normal_i Normal vector at anchor point in frame i
     * @param _pts_j 3D current point in camera frame j
     * @param _velocity_j 3D velocity of current point in frame j
     * @param _td_j Time offset estimate at current frame j
     * @return Ceres cost function pointer (3 residuals, 5 parameter blocks: 7+7+7+1+1)
     */
    static ceres::CostFunction* Create(const Eigen::Vector3d& _pts_i,
                                      const Eigen::Vector3d& _velocity_i,
                                      double _td_i,
                                      const Eigen::Vector3d& _normal_i,
                                      const Eigen::Vector3d& _pts_j,
                                      const Eigen::Vector3d& _velocity_j,
                                      double _td_j);

private:
    /**
     * @brief Convert VINS pose parameters to 4x4 transformation matrix
     * 
     * Converts VINS pose parameterization to homogeneous transformation matrix.
     * VINS uses quaternion order [qx, qy, qz, qw] in pose vector.
     * 
     * @param pose VINS pose [px, py, pz, qx, qy, qz, qw]
     * @return 4x4 transformation matrix T = [R t; 0 1]
     */
    template <typename T>
    Eigen::Matrix<T, 4, 4> poseToMatrix(const T* pose) const;

    // Member variables - stored measurements and estimates
    const Eigen::Vector3d _pts_i_;      ///< Normalized 3D direction of anchor point in camera frame i
    const Eigen::Vector3d _pts_j_;      ///< 3D current point in camera frame j
    const Eigen::Vector3d _normal_i_;   ///< Normalized normal vector in camera frame i
    const Eigen::Vector3d _velocity_i_; ///< 3D velocity of anchor point (m/s)
    const Eigen::Vector3d _velocity_j_; ///< 3D velocity of current point (m/s)
    const double _td_i_;                ///< Time offset estimate at frame i (seconds)
    const double _td_j_;                ///< Time offset estimate at frame j (seconds)
};
