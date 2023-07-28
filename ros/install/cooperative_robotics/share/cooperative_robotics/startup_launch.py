#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    models = os.getcwd() + "/src/cooperative_robotics/models/"

    world = models+"gridworld3d.world"
    print("PRITING", world)

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )


    urdf_path_1 = models + "turtlebot3_burger/model_r1.sdf"
    with open(urdf_path_1, 'r') as infp:
        robot_desc1 = infp.read()

    robot_state_publisher_cmd1 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher_1',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc1
            }],
        )
    
    urdf_path_2 = models + "turtlebot3_burger/model_r2.sdf"
    with open(urdf_path_2, 'r') as infp:
        robot_desc2 = infp.read()

    robot_state_publisher_cmd2 = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher_2',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc2
            }],
        )
    
    start_gazebo_ros_spawner_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        namespace="Turtle1",
        name="ROBOT1",
        arguments=[
            '-entity', "robot1",
            '-file', urdf_path_1,
            '-x', LaunchConfiguration('x_pose', default=f"{(6*0.3):.2f}"),
            '-y', LaunchConfiguration('y_pose', default=f"{(12*0.3):.2f}"),
            '-z', '0.01'
        ],
        output='screen',
    )
    
    start_gazebo_ros_spawner_cmd2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        namespace="Turtle2",
        name="ROBOT2",
        arguments=[
            '-entity', "robot2",
            '-file', urdf_path_2,
            '-x', LaunchConfiguration('x_pose', default=f"{(6*0.3):.2f}"),
            '-y', LaunchConfiguration('y_pose', default=f"{(9*0.3):.2f}"),
            '-z', '0.01'
        ],
        output='screen',
    )


    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd1)
    ld.add_action(robot_state_publisher_cmd2)

    # Add any conditioned actions
    ld.add_action(start_gazebo_ros_spawner_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd2)

    return ld
