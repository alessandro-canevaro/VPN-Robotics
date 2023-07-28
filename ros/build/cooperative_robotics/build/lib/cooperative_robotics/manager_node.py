import sys
from threading import Thread

from example_interfaces.srv import AddTwoInts
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Twist, Pose
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import threading

class GetterClientSync(Node):

    def __init__(self):
        super().__init__('getter_client_sync')
        self.cli = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = GetEntityState.Request()

    def send_request(self, agent, frame="world"):
        self.req.name = agent
        self.req.reference_frame = frame
        return self.cli.call(self.req)

class SetterClientSync(Node):

    def __init__(self):
        super().__init__('setter_client_sync')
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetEntityState.Request()

    def send_request(self, state):
        self.req.state = state
        return self.cli.call(self.req)
    
class GetterOdometry(Node):
    def __init__(self, robot_name):
        super().__init__('odometry_getter')

        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(
            Odometry,
            robot_name+'/odom',
            self.odom_callback,
            qos)

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

    def get_odom(self):
        return self.last_pose_x, self.last_pose_y, self.last_pose_theta
    
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

def main():
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()

    getter_client = GetterClientSync()
    executor.add_node(getter_client)
    setter_client = SetterClientSync()
    executor.add_node(setter_client)

    odom_r1 = GetterOdometry("robot1")
    executor.add_node(odom_r1)
    odom_r2 = GetterOdometry("robot2")
    executor.add_node(odom_r2)

    # Spin in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    logger = {"x1_r": [], "y1_r": [], "x1_o": [], "y1_o": [], "x2_r": [], "y2_r": [], "x2_o": [], "y2_o": []}

    i=0
    while i<15000:
        time.sleep(0.01)
        res_r1 = getter_client.send_request("robot1")
        #getter_client.get_logger().info(f'robot1 pos {res_r1.state.pose.position.x} {res_r1.state.pose.position.y}')
        x1_r, y1_r = res_r1.state.pose.position.x, res_r1.state.pose.position.y
        x1_o, y1_o, __ = odom_r1.get_odom()

        res_r2 = getter_client.send_request("robot2")
        #getter_client.get_logger().info(f'robot2 pos {res_r2.state}')
        x2_r, y2_r = res_r2.state.pose.position.x, res_r2.state.pose.position.y
        x2_o, y2_o, __ = odom_r2.get_odom()
        logger["x1_r"].append(x1_r)
        logger["y1_r"].append(y1_r)
        logger["x1_o"].append(x1_o)
        logger["y1_o"].append(y1_o)
        logger["x2_r"].append(x2_r)
        logger["y2_r"].append(y2_r)
        logger["x2_o"].append(x2_o)
        logger["y2_o"].append(y2_o)

        state = EntityState()
        state.name = "object"
        state.pose = Pose()
        state.pose.position.x = (res_r1.state.pose.position.x + res_r2.state.pose.position.x)/2
        state.pose.position.y = (res_r1.state.pose.position.y + res_r2.state.pose.position.y)/2 
        state.pose.position.z = 0.21
        angle = math.atan2(res_r1.state.pose.position.y - res_r2.state.pose.position.y, res_r1.state.pose.position.x - res_r2.state.pose.position.x)
        qx, qy, qz, qw = get_quaternion_from_euler(0, 0, angle-np.pi/2)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.twist = Twist()
        state.reference_frame = "world"
        
        #, pose: {position:{x: 1.77, y: 4.15, z: 0.21}}, twist: {linear:{x: 0.0, y: 0.0, z: 0.0}, angular:{x: 0.0, y: 0.0, z: 0.0}}, reference_frame: world}"
        response = setter_client.send_request(state)
        #setter_client.get_logger().info(f'object pos {response.success}')

        response = getter_client.send_request("object")
        #getter_client.get_logger().info(f'object pos {response.state}')

        i+=1

    #plt.plot(np.array(logger["x1_r"][10:])-np.array(logger["x1_o"][10:]), label="x1")
    #plt.plot(np.array(logger["y1_r"][10:])-np.array(logger["y1_o"][10:]), label="y1")
    #plt.plot(np.array(logger["x2_r"][10:])-np.array(logger["x2_o"][10:]), label="x2")
    #plt.plot(np.array(logger["y2_r"][10:])-np.array(logger["y2_o"][10:]), label="y2")

    """
    diff_x = np.array(logger["x1_r"][10:])-np.array(logger["x2_r"][10:])
    diff_y = np.array(logger["y1_r"][10:])-np.array(logger["y2_r"][10:])                                      
    diff = np.stack((diff_x, diff_y))
    data = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)*1000
    plt.plot(data, label="abs. dist. err.")

    data_lidar = np.loadtxt("run_lidar.txt")
    plt.plot(data_lidar, label="lidar")
    #data_radar = np.loadtxt("run_radar.txt")
    #plt.plot(data_radar, label="radar")

    #data_odom = np.loadtxt("run_odom.txt")
    #plt.plot(data_odom, label="odom")

    plt.xlabel("time")
    plt.ylabel("millimeters")
    plt.grid()
    plt.legend()
    plt.savefig("odometry_error.pdf")
    print("plotted")
    np.savetxt("run", data)
    """

    rclpy.shutdown()
    executor_thread.join()


if __name__ == '__main__':
    main()