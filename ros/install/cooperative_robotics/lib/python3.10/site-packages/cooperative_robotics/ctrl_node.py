import numpy as np
import sys
import termios

from geometry_msgs.msg import Twist
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Pose, TransformStamped
import tf2_geometry_msgs.tf2_geometry_msgs
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException

from src.environment.gridworld import MRS
from src.environment.trajectory import compute_center_path, compute_robots_path, compute_reference_input, nonlin_controller, make_plots
from src.models.MultiAgentA2C import MAA2C    
from PIL import Image, ImageOps
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

terminal_msg = """
Turtlebot3 Position Control
------------------------------------------------------
From the current pose,
x: goal position x (unit: m)
y: goal position y (unit: m)
theta: goal orientation (range: -180 ~ 180, unit: deg)
------------------------------------------------------
"""

class TrajectoryTracker(Node):
    def __init__(self, robot_name) -> None:
        
        self.robot_name =robot_name
        super().__init__(self.robot_name+"_myposctrl")


        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        self.odom = Odometry()


        self.follow_path_state = False
        self.init_odom_state = False  # To get the initial pose at the beginning

        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.robot_name+'/cmd_vel', qos)
        # Initialise subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            self.robot_name+'/odom',
            self.odom_callback,
            qos)
        self.nois_odom_sub = self.create_subscription(
            Odometry,
            self.robot_name+'/noisy_odom',
            self.noisy_odom_callback,
            qos)
        
        self.last_synch_msg = ""
        self.synch_pub = self.create_publisher(String, '/'+self.robot_name+'_synch', qos)
        self.synch_sub = self.create_subscription(
            String,
            {"robot1": "/robot2_synch", "robot2": "/robot1_synch"}[self.robot_name],
            self.synch_callback,
            qos)
        self._do_prediction = True
        self._msg_data = ""
        self._self_action = -1
        self._other_action = -1
        self.checkpoint_num = 0

        self.dt = 0.010 # unit: s
        self.T = 5 # seconds
        self.t=0
        self.samples = int(self.T/self.dt)

        self.update_timer = self.create_timer(self.dt, self.update_callback)  

        self.block_size = 0.3
        self.k=0.2
        self.old_mrs = MRS(np.array([[1.5, 180], [1.5, 0]]), np.array([6, 10.5]), angle=-90, mass=2.0, inertia=0.08)

        self.vpn = MAA2C.load("../models/ROS/VPN-VProp_ROS32_Coop", device="cpu")    
        self.vpn.policy.mlp_extractor.vi_k = 69

        self.obs = np.zeros((4, 34, 34))
        self.obs[0, :, :] = 1-(np.array(ImageOps.grayscale(Image.open("./src/cooperative_robotics/models/map32/map1.png"))) // 255)
        # temp_mat = self.obs[0, :, :].astype(np.int8)
        # print(temp_mat)
        # np.savetxt("map_2d.csv", temp_mat, fmt="%d", delimiter=',')
        # print("saved")
        # quit()
        _robots_pos = np.rint(self.old_mrs.get_agents_positions()).astype(int)
        x1, y1 = _robots_pos[{"robot1": 0, "robot2": 1}[self.robot_name]]
        self.obs[1, x1, y1] = 1
        x2, y2 = _robots_pos[{"robot1": 1, "robot2": 0}[self.robot_name]]
        self.obs[2, x2, y2] = 1

        self.last_pose_x = x1 * self.block_size
        self.last_pose_y = y1 * self.block_size
        self.last_pose_theta = 0.0

        self.last_pose_x_noisy = x1 * self.block_size
        self.last_pose_y_noisy = y1 * self.block_size
        self.last_pose_theta_noisy = 0.0

        print(self.old_mrs.get_agents_positions())
        self.get_logger().info(f"self in ({x1}, {y1}), other in: ({x2}, {y2})")
        self.target_pos = (18, 16)
        self.target_shape = (5, 3)
        self.obs[3, self.target_pos[0]-self.target_shape[0]//2:self.target_pos[0]+self.target_shape[0]//2+1, self.target_pos[1]-self.target_shape[1]//2:self.target_pos[1]+self.target_shape[1]//2+1] = 1

        self._action_to_direction = {
            0: np.array([-1, -1]),
            1: np.array([-1, 0]), #up
            2: np.array([-1, 1]),            
            3: np.array([0, -1]), #left
            4: np.array([0, 1]), #right
            5: np.array([1, -1]),
            6: np.array([1, 0]), #down
            7: np.array([1, 1]),           
        }

        self.tb_writer = SummaryWriter("../log/ROS/Simulations")


        self.data_xref = []
        self.data_yref = []
        self.data_x = []
        self.data_y = []
        self.logger = {"x": [], "y": [], "theta": [], "x_ref": [], "y_ref": [], "theta_ref": [], "x_n": [], "y_n": [], "theta_n": [], "u1": [], "u2": [], "e1": [], "e2": [], "e3": []}

        self.get_logger().info("Turtlebot3 position control node has been initialised.")
        
    def synch_callback(self, msg):
        self.last_synch_msg = msg.data

    def compute_trajectory(self, rl, ra):
        robot_heading = np.rad2deg(np.arctan2(self.new_mrs.pos[1]-self.old_mrs.pos[1], self.new_mrs.pos[0]-self.old_mrs.pos[0]))
        self.theta_i = np.deg2rad(self.old_mrs.angle + 90 + robot_heading)
        self.theta_f = np.deg2rad(self.new_mrs.angle + 90 + robot_heading)
        xa, ya = compute_robots_path(self.old_mrs, self.new_mrs, self.theta_i, self.theta_f, self.block_size, k=self.k)
        t = np.linspace(0, self.T, self.samples+1)
        s = t/self.T
        return xa(s, rl, ra), ya(s, rl, ra)

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.init_odom_state = True

    def noisy_odom_callback(self, msg):
        self.last_pose_x_noisy = msg.pose.pose.position.x
        self.last_pose_y_noisy = msg.pose.pose.position.y
        _, _, self.last_pose_theta_noisy = self.euler_from_quaternion(msg.pose.pose.orientation)
        self._apply_loc_corr()
        self.init_odom_state = True

    def _apply_loc_corr(self):
        pose_err = Pose()
        pose_err.position.x = self.last_pose_x_noisy
        pose_err.position.y = self.last_pose_y_noisy
        qx, qy, qz, qw = get_quaternion_from_euler(0, 0, self.last_pose_theta_noisy)
        pose_err.orientation.x = qx
        pose_err.orientation.y = qy
        pose_err.orientation.z = qz
        pose_err.orientation.w = qw
               
        trans = TransformStamped()
        try:
            trans = self._tf_buffer.lookup_transform("map", 'noisy_odom_r'+{"robot1":'1', "robot2":'2'}[self.robot_name], rclpy.time.Time())
        except LookupException as e:
            self.get_logger().error('failed to get transform {} \n'.format(repr(e)))

        pose_corr = tf2_geometry_msgs.do_transform_pose(pose_err, trans)
        self.last_pose_x_noisy = pose_corr.position.x
        self.last_pose_y_noisy = pose_corr.position.y
        _, _, self.last_pose_theta_noisy = self.euler_from_quaternion(pose_corr.orientation)


    def update_callback(self):
        if self.init_odom_state is True:
            self.generate_path()

    def generate_path(self):
        twist = Twist()
        
        if self.follow_path_state is False:

            terminated = False if -1 in self.obs[3] - np.sum(self.obs[[1, 2], :, :], axis=0) else True
            if terminated:
                print("GOAL REACHED!")
                xi, yi, thetai = np.array(self.logger["x"]), np.array(self.logger["y"]), np.array(self.logger["theta"])
                xr, yr, thetar = np.array(self.logger["x_ref"]), np.array(self.logger["y_ref"]), np.array(self.logger["theta_ref"])
                xo, yo, thetao = np.array(self.logger["x_n"]), np.array(self.logger["y"]), np.array(self.logger["theta"])
                np.savetxt(f"./data/run_data_{self.robot_name}.txt", np.stack((xi, yi, thetai, xr, yr, thetar, xo, yo, thetao)))
                print("data saved")
                exit()

            if self._do_prediction:
                action, __ = self.vpn.predict(self.obs, deterministic=True)
                self._self_action = int(action)
                self._msg_data = 'ready_'+str(self.robot_name)+"###"+str(self.checkpoint_num)+"###"+str(action)
                self._do_prediction = False

            msg = String()
            msg.data = self._msg_data
            self.synch_pub.publish(msg)
            if 'ready_'+{"robot1": "robot2", "robot2": "robot1"}[self.robot_name]+"###"+str(self.checkpoint_num) not in self.last_synch_msg:
                return
            
            self._do_prediction = True
            print("sent message:", msg.data)
            print("recived message:", self.last_synch_msg)
            __, __, action_msg = self.last_synch_msg.split("###")

            self.get_logger().info(f"Navigating to checkpoint {self.checkpoint_num+1}")
            self._other_action = int(action_msg)

            dir_self = self._action_to_direction[self._self_action]
            dir_other = self._action_to_direction[self._other_action]
            if self.robot_name == "robot1":
                directions = np.array([dir_self, dir_other])
            else:
                directions = np.array([dir_other, dir_self])
                
            if self.checkpoint_num > 0:
                """
                s = np.linspace(0, 1, self.samples)
                t = np.linspace(0, self.T, self.samples)
                st = t/self.T   

                samples = 800
                s = np.linspace(0, 1, samples)
                T = 8
                t = np.linspace(0, T, samples)
                st = t/T
                print(self.t, len(self.logger["e1"]))
                self.logger["xref"], self.logger["yref"], self.logger["thetaref"] = self.x_r[1:], self.y_r[1:], self.theta_r[1:]
                self.logger["vref"], self.logger["wref"] = self.v_r[1:], self.w_r[1:]
                make_plots(self.old_mrs, self.new_mrs, self.logger, self.theta_i, self.theta_f, self.k, self.block_size, s, self.dt, t, st)
                """
                self.old_mrs = self.new_mrs

            #compute desired path  
            self.new_mrs = self.old_mrs.move(directions)
            #self.new_mrs = self.old_mrs.move(np.array([[1, 0], [0, 1]]))
            #update obs
            _robots_pos = np.rint(self.old_mrs.get_agents_positions()).astype(int)
            x1, y1 = _robots_pos[{"robot1": 0, "robot2": 1}[self.robot_name]]
            self.obs[1, x1, y1] = 0
            x2, y2 = _robots_pos[{"robot1": 1, "robot2": 0}[self.robot_name]]
            self.obs[2, x2, y2] = 0
            _robots_pos = np.rint(self.new_mrs.get_agents_positions()).astype(int)
            x1, y1 = _robots_pos[{"robot1": 0, "robot2": 1}[self.robot_name]]
            self.obs[1, x1, y1] = 1
            x2, y2 = _robots_pos[{"robot1": 1, "robot2": 0}[self.robot_name]]
            self.obs[2, x2, y2] = 1
            self.get_logger().info(f"directions: agent 0: {directions[0]} (action: {self._self_action}), agent 1: {directions[1]} (action: {self._other_action})")
            self.get_logger().info(f"updating obs: self in ({x1}, {y1}), other in: ({x2}, {y2})")

            image = np.bitwise_or(np.sum(self.obs[:-1, :, :], axis=0).astype(np.int8), self.obs[-1, :, :].astype(np.int8))
            self.tb_writer.add_image("Nav/"+self.robot_name+'/obs', np.expand_dims(image, axis=0), self.checkpoint_num)

            rl, ra = self.old_mrs.agents[{"robot1": 0, "robot2": 1}[self.robot_name]]
            self.x_r, self.y_r = self.compute_trajectory(rl, ra)
            self.theta_r, self.v_r, self.w_r = compute_reference_input(self.x_r, self.y_r, self.dt)
            
            current_heading = np.rad2deg(self.last_pose_theta_noisy)
            self.sample_rot = 300
            theta_rot = []
            for i in range(self.sample_rot):
                theta_rot.append(current_heading*(1-(i/self.sample_rot)) + np.rad2deg(self.theta_r[0])*i/self.sample_rot)
            self.x_r, self.y_r = np.concatenate([np.ones((self.sample_rot))*self.x_r[0], self.x_r]), np.concatenate([np.ones((self.sample_rot))*self.y_r[0], self.y_r]) 
            self.theta_r, self.v_r, self.w_r = np.concatenate([np.deg2rad(np.array(theta_rot)), self.theta_r]), np.concatenate([np.zeros((self.sample_rot)), self.v_r]), np.concatenate([np.ones((self.sample_rot))*np.deg2rad(np.rad2deg(self.theta_r[0])-current_heading)/(self.sample_rot*self.dt), self.w_r]) 

            
            figure = plt.figure()
            plt.plot(np.array(self.data_xref)/self.block_size, np.array(self.data_yref)/self.block_size, label="reference")
            plt.plot(np.array(self.data_x)/self.block_size, np.array(self.data_y)/self.block_size, label="actual path")
            plt.legend()
            plt.grid()
            self.tb_writer.add_figure("Nav/"+self.robot_name+'/path', figure, global_step=self.checkpoint_num)

            self.t = 0
            self.follow_path_state = True
            print("\n")

        else:
            # Step 1: Turn
            if self.t <= self.samples+self.sample_rot:
                self.data_xref.append(self.x_r[self.t])
                self.data_yref.append(self.y_r[self.t])
                self.data_x.append(self.last_pose_x)
                self.data_y.append(self.last_pose_y)

                v, w, u1, u2, e1, e2, e3 = nonlin_controller(self.last_pose_x_noisy, self.last_pose_y_noisy, self.last_pose_theta_noisy, self.x_r[self.t], self.y_r[self.t], self.theta_r[self.t], self.v_r[self.t], self.w_r[self.t])
                twist.linear.x = v
                twist.angular.z = w
                if self.t < self.samples+self.sample_rot:
                    for name, val in zip(["x", "y", "theta", "x_ref", "y_ref", "theta_ref", "x_n", "y_n", "theta_n", "u1", "u2", "e1", "e2", "e3"], [self.last_pose_x, self.last_pose_y, self.last_pose_theta, self.x_r[self.t], self.y_r[self.t], self.theta_r[self.t], self.last_pose_x_noisy, self.last_pose_y_noisy, self.last_pose_theta_noisy, u2, e1, e2, e3]):
                        self.logger[name].append(val)

                self.t +=1

            # Reset
            else:
                self.checkpoint_num += 1
                self.follow_path_state = False
                
            self.cmd_vel_pub.publish(twist)

    def euler_from_quaternion(self, quat):
        """
        Convert quaternion (w in last place) to euler roll, pitch, yaw.

        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


def main(args=None):
    rclpy.init(args=args)

    name = input("what robot do you want to control? (robot1 or robot2)")
    tt = TrajectoryTracker(name)
    rclpy.spin(tt)
    tt.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
