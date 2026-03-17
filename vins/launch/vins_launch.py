from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
import launch_ros.actions


def generate_launch_description():
    vins_config_file = LaunchConfiguration('vins_config_file', default = './src/VINS-tree/vins/config/euroc/euroc_stereo_imu_config.yaml')

    declare_vins_config_file = DeclareLaunchArgument(
        'vins_config_file',
        default_value=vins_config_file
    )
    vins_node = launch_ros.actions.Node(
        package='vins',
        executable='vins_node',
        output='screen',
        parameters=[{'vins_config_file': vins_config_file}]
    )
    
    ld = LaunchDescription()

    ld.add_action(declare_vins_config_file)
    ld.add_action(vins_node)
    return ld