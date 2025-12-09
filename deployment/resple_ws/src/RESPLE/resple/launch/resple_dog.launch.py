import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():    
    config_yaml_fusion = os.path.join(
        get_package_share_directory('resple'),
        'config',
        'config_dog.yaml')    
    return launch.LaunchDescription([             	                
        launch_ros.actions.Node(
            package='resple',
            executable='RESPLE',
            name='RESPLE',
            emulate_tty=True,
            parameters=[config_yaml_fusion],
            arguments=['--ros-args', '--log-level', 'warn'])          
  ])

