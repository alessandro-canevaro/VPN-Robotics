from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import threading
import copy

class NoiseOdom(Node):
    def __init__(self, robot_name) -> None:
        
        self.robot_name =robot_name
        super().__init__(self.robot_name+"_noise_odom")
        
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.noisy_odom_pub = self.create_publisher(Odometry, 
                                                    self.robot_name+'/noisy_odom', 
                                                    qos)
        # Initialise subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            self.robot_name+'/odom',
            self.odom_callback,
            qos)
        
        if self.robot_name == "robot1":
            x, y = 6, 12
        else:
            x, y = 6, 9

        self.block_size = 0.3
        self.last_pose_x = x * self.block_size
        self.last_pose_y = y * self.block_size
        self.last_pose_theta = 0.0

        self.last_pose_x_noisy = self.last_pose_x
        self.last_pose_y_noisy = self.last_pose_y
        self.last_pose_theta_noisy = self.last_pose_theta

        # Initialize the transform broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        """
        self.map_tf = StaticTransformBroadcaster(self)
        t = TransformStamped()
        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'noisy_odom_r'+{"robot1":'1', "robot2":'2'}[self.robot_name]
        # Send the transformation
        self.map_tf.sendTransform(t)
        """

        self.get_logger().info(f"node Initialized")

    def odom_callback(self, msg):
        self.old_x, self.old_y, self.old_theta = self.last_pose_x, self.last_pose_y, self.last_pose_theta
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.last_pose_x_noisy, self.last_pose_y_noisy, self.last_pose_theta_noisy = self.noisy_odom(self.last_pose_x, self.last_pose_y, self.last_pose_theta,
                                                                                                     self.old_x, self.old_y, self.old_theta,
                                                                                                     self.last_pose_x_noisy, self.last_pose_y_noisy, self.last_pose_theta_noisy, 
                                                                                                     np.sign(msg.twist.twist.linear.x))
        pub_msg = copy.copy(msg)
        pub_msg.pose.pose.position.x = self.last_pose_x_noisy
        pub_msg.pose.pose.position.y = self.last_pose_y_noisy

        qx, qy, qz, qw = get_quaternion_from_euler(0, 0, self.last_pose_theta_noisy)
        pub_msg.pose.pose.orientation.x = qx
        pub_msg.pose.pose.orientation.y = qy
        pub_msg.pose.pose.orientation.z = qz
        pub_msg.pose.pose.orientation.w = qw
        
        self.noisy_odom_pub.publish(pub_msg)

        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'noisy_odom_r'+{"robot1":'1', "robot2":'2'}[self.robot_name]
        t.child_frame_id = 'base_footprint_r'+{"robot1":'1', "robot2":'2'}[self.robot_name]

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = self.last_pose_x_noisy
        t.transform.translation.y = self.last_pose_y_noisy
        t.transform.translation.z = 0.0
        #t.transform.rotation = msg.pose.pose.orientation
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


    def _angle_diff(self, a, b):
        a = np.arctan2(np.sin(a), np.cos(a))
        b = np.arctan2(np.sin(b), np.cos(b))
        d1 = a - b
        d2 = 2 * np.pi - np.abs(d1)
        if d1 > 0:
            d2 *= -1.0
        if np.abs(d1) < np.abs(d2):
            return d1
        else:
            return d2
  
    def noisy_odom(self, x, y, theta, xo, yo, thetao, xon, yon, thetaon, direction=1.0):
        alpha1, alpha2, alpha3, alpha4 = 0.01, 0.01, 0.01, 0.01

        if np.sqrt(np.square(x-xo)+np.square(y-yo)) < 0.01:
            delta_rot_1 = 0.0
        else:
            delta_rot_1 = np.arctan2(y-yo, x-xo) - thetao
        delta_trans = np.sqrt((xo-x)**2 + (yo-y)**2)
        delta_rot_2 = theta - thetao - delta_rot_1

        min_delta_rot_1 = min([np.abs(self._angle_diff(delta_rot_1, 0.0)), np.abs(self._angle_diff(delta_rot_1, np.pi))])
        min_delta_rot_2 = min([np.abs(self._angle_diff(delta_rot_2, 0.0)), np.abs(self._angle_diff(delta_rot_2, np.pi))])

        delta_rot_1_n = delta_rot_1 + np.random.normal(0.0, np.sqrt(alpha1*np.square(min_delta_rot_1) + alpha2*np.square(delta_trans)))
        delta_trans_n = delta_trans + np.random.normal(0.0, np.sqrt(alpha3*np.square(delta_trans) + alpha4*np.square(min_delta_rot_1) + alpha4*np.square(min_delta_rot_2)))
        delta_rot_2_n = delta_rot_2 + np.random.normal(0.0, np.sqrt(alpha1*np.square(min_delta_rot_2) + alpha2*np.square(delta_trans)))

        xn = xon + direction * delta_trans_n * np.cos(thetaon + delta_rot_1_n)
        yn = yon + direction * delta_trans_n * np.sin(thetaon + delta_rot_1_n)
        thetan = thetaon + delta_rot_1_n + delta_rot_2_n
        #print("{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x, xn, xo, xon, delta_trans_n * np.cos(thetaon + delta_rot_1_n)))

        return xn, yn, thetan

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

    #name = input("what robot do you want to control? (robot1 or robot2)")
    

    rclpy.shutdown()


    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()

    no1 = NoiseOdom("robot1")
    executor.add_node(no1)

    # Spin in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    no2 = NoiseOdom("robot2")
    rclpy.spin(no2)
    no2.destroy_node()

    rclpy.shutdown()
    executor_thread.join()


if __name__ == '__main__':
    main()
