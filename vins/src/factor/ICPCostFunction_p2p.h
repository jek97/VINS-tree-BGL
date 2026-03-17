#ifndef ICP_COST_FUNCTION_P2P_VCOMP_TEST_H
#define ICP_COST_FUNCTION_P2P_VCOMP_TEST_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

// ==============================================================================
// ICPCostFunction_p2p - Point-to-Point ICP Cost Function
// Reprojects anchor point to current frame for error computation
// WITH DEPTH OPTIMIZATION SUPPORT
// ==============================================================================

class ICPCostFunction_p2p {
public:
    ICPCostFunction_p2p(
        const Eigen::Vector3d& _pts_i,
        const Eigen::Vector3d& _velocity_i,
        double _td_i,
        const Eigen::Vector3d& _pts_j,
        const Eigen::Vector3d& _velocity_j,
        double _td_j);

    template <typename T>
    bool operator()(
        const T* const para_Pose_i,
        const T* const para_Pose_j,
        const T* const para_Ex_Pose_0,
        const T* const para_depth,
        const T* const para_Td,
        T* residuals) const;

    static ceres::CostFunction* Create(
        const Eigen::Vector3d& _pts_i,
        const Eigen::Vector3d& _velocity_i,
        double _td_i,
        const Eigen::Vector3d& _pts_j,
        const Eigen::Vector3d& _velocity_j,
        double _td_j);

private:
    template <typename T>
    Eigen::Matrix<T, 4, 4> poseToMatrix(const T* pose) const;

    Eigen::Vector3d _pts_i_;       // Normalized direction vector for anchor point
    Eigen::Vector3d _velocity_i_;  // 3D velocity of anchor point
    double _td_i_;                 // Time delay for anchor point
    Eigen::Vector3d _pts_j_;       // Current point in camera frame j
    Eigen::Vector3d _velocity_j_;  // 3D velocity of current point
    double _td_j_;                 // Time delay for current point
};

#endif // ICP_COST_FUNCTION_P2P_VCOMP_TEST_H