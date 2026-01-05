from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'debug',
            default_value='1',
            description='Debug level'
        ),
        DeclareLaunchArgument(
            'debug_dir',
            default_value='debug_ros2_service',
            description='Debug directory'
        ),
        DeclareLaunchArgument(
            'downscale',
            default_value='0.5',
            description='Downscale factor for images'
        ),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_color_optical_frame',
            description='TF2 parent frame (camera frame)'
        ),
        DeclareLaunchArgument(
            'object_frame',
            default_value='foundationpose_object',
            description='TF2 child frame (object frame)'
        ),
        DeclareLaunchArgument(
            'score_threshold',
            default_value='100.0',
            description='Maximum score threshold - reset tracking if score exceeds this value'
        ),
        DeclareLaunchArgument(
            'color_topic',
            default_value='/sim_camera_rgb',
            description='Color image topic'
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/sim_camera_depth',
            description='Depth image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/sim_camera_info',
            description='Camera info topic'
        ),
        DeclareLaunchArgument(
            'enable_visualization',
            default_value='true',
            description='Enable/disable OpenCV visualization window'
        ),

        Node(
            package='foundationpose_bridge',
            executable='foundationpose_service_node',
            name='foundationpose_service_node',
            output='screen',
            parameters=[{
                'debug': LaunchConfiguration('debug'),
                'debug_dir': LaunchConfiguration('debug_dir'),
                'downscale': LaunchConfiguration('downscale'),
                'camera_frame': LaunchConfiguration('camera_frame'),
                'object_frame': LaunchConfiguration('object_frame'),
                'score_threshold': LaunchConfiguration('score_threshold'),
                'color_topic': LaunchConfiguration('color_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'enable_visualization': LaunchConfiguration('enable_visualization'),
            }]
        ),
    ])
