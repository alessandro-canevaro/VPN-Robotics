import numpy as np
import matplotlib.pyplot as plt 


def computer_errors(data_r1, data_r2):
    diff_x = data_r1[0, :] - data_r2[0, :]
    diff_y = data_r1[1, :] - data_r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    local_error = np.abs(np.linalg.norm(diff, ord=2, axis=0) - 0.9)*1000

    diff_x = data_r1[3, :] - data_r1[0, :]
    diff_y = data_r1[4, :] - data_r1[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    global_error_r1 = np.abs(np.linalg.norm(diff, ord=2, axis=0))*1000

    diff_x = data_r2[3, :] - data_r2[0, :]
    diff_y = data_r2[4, :] - data_r2[1, :]                                 
    diff = np.stack((diff_x, diff_y))
    global_error_r2 = np.abs(np.linalg.norm(diff, ord=2, axis=0))*1000

    diff_x = data_r1[0, :] - data_r1[6, :]
    diff_y = data_r1[1, :] - data_r1[7, :]                                 
    diff = np.stack((diff_x, diff_y))
    odometry_error_r1 = np.abs(np.linalg.norm(diff, ord=2, axis=0))*1000

    diff_x = data_r2[0, :] - data_r2[6, :]
    diff_y = data_r2[1, :] - data_r2[7, :]                                 
    diff = np.stack((diff_x, diff_y))
    odometry_error_r2 = np.abs(np.linalg.norm(diff, ord=2, axis=0))*1000
    
    return local_error, global_error_r1, global_error_r2, odometry_error_r1, odometry_error_r2

def plot_path():
    data_r1_ideal = np.loadtxt("./data/run_data_robot1_ideal.txt")
    data_r2_ideal = np.loadtxt("./data/run_data_robot2_ideal.txt")
    data_r1_odom = np.loadtxt("./data/run_data_robot1_odom.txt")
    data_r2_odom = np.loadtxt("./data/run_data_robot2_odom.txt")
    data_r1_lidar = np.loadtxt("./data/run_data_robot1_lidar.txt")
    data_r2_lidar = np.loadtxt("./data/run_data_robot2_lidar.txt")
    data_r1_6Gsensor = np.loadtxt("./data/run_data_robot1_6Gsensor.txt")
    data_r2_6Gsensor = np.loadtxt("./data/run_data_robot2_6Gsensor.txt")
    local_error_ideal, global_error_r1_ideal, global_error_r2_ideal, odometry_error_r1_ideal, odometry_error_r2_ideal = computer_errors(data_r1_ideal, data_r2_ideal)
    local_error_odom, global_error_r1_odom, global_error_r2_odom, odometry_error_r1_odom, odometry_error_r2_odom = computer_errors(data_r1_odom, data_r2_odom)
    local_error_lidar, global_error_r1_lidar, global_error_r2_lidar, odometry_error_r1_lidar, odometry_error_r2_lidar = computer_errors(data_r1_lidar, data_r2_lidar)
    local_error_6Gsensor, global_error_r1_6Gsensor, global_error_r2_6Gsensor, odometry_error_r1_6Gsensor, odometry_error_r2_6Gsensor = computer_errors(data_r1_6Gsensor, data_r2_6Gsensor)

    
    plt.plot(data_r1_ideal[0, :], data_r1_ideal[1, :], color="blue", label="ideal")
    plt.plot(data_r2_ideal[0, :], data_r2_ideal[1, :], color="blue")
    plt.plot(data_r1_odom[0, :], data_r1_odom[1, :], color="red", label="odom")
    plt.plot(data_r2_odom[0, :], data_r2_odom[1, :], color="red")
    plt.plot(data_r1_lidar[0, :], data_r1_lidar[1, :], color="green", label="lidar")
    plt.plot(data_r2_lidar[0, :], data_r2_lidar[1, :], color="green")
    plt.plot(data_r1_6Gsensor[0, :], data_r1_6Gsensor[1, :], color="orange", label="6Gsensor")
    plt.plot(data_r2_6Gsensor[0, :], data_r2_6Gsensor[1, :], color="orange")
    """
    plt.plot(data_r1_ideal[3, :], data_r1_ideal[4, :], color="blue", label="r1-ideal")
    plt.plot(data_r2_ideal[3, :], data_r2_ideal[4, :], color="blue", label="r2-ideal")
    plt.plot(data_r1_odom[3, :], data_r1_odom[4, :], color="red", label="r1-odom")
    plt.plot(data_r2_odom[3, :], data_r2_odom[4, :], color="red", label="r2-odom")
    plt.plot(data_r1_lidar[3, :], data_r1_lidar[4, :], color="green", label="r1-lidar")
    plt.plot(data_r2_lidar[3, :], data_r2_lidar[4, :], color="green", label="r2-lidar")
    plt.plot(data_r1_6Gsensor[3, :], data_r1_6Gsensor[4, :], color="orange", label="r1-6Gsensor")
    plt.plot(data_r2_6Gsensor[3, :], data_r2_6Gsensor[4, :], color="orange", label="r2-6Gsensor") """
    """
    plt.plot(data_r1_ideal[6, :], data_r1_ideal[7, :], color="blue", label="r1-ideal")
    plt.plot(data_r2_ideal[6, :], data_r2_ideal[7, :], color="blue", label="r2-ideal")
    plt.plot(data_r1_odom[6, :], data_r1_odom[7, :], color="red", label="r1-odom")
    plt.plot(data_r2_odom[6, :], data_r2_odom[7, :], color="red", label="r2-odom")
    plt.plot(data_r1_lidar[6, :], data_r1_lidar[7, :], color="green", label="r1-lidar")
    plt.plot(data_r2_lidar[6, :], data_r2_lidar[7, :], color="green", label="r2-lidar")
    plt.plot(data_r1_6Gsensor[6, :], data_r1_6Gsensor[7, :], color="orange", label="r1-6Gsensor")
    plt.plot(data_r2_6Gsensor[6, :], data_r2_6Gsensor[7, :], color="orange", label="r2-6Gsensor")"""

    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.grid()
    plt.legend()
    plt.savefig("path_error.pdf")
    print("plotted")

def plot_err():
    data_r1_ideal = np.loadtxt("./data/run_data_robot1_ideal.txt")
    data_r2_ideal = np.loadtxt("./data/run_data_robot2_ideal.txt")
    data_r1_odom = np.loadtxt("./data/run_data_robot1_odom.txt")
    data_r2_odom = np.loadtxt("./data/run_data_robot2_odom.txt")
    data_r1_lidar = np.loadtxt("./data/run_data_robot1_lidar.txt")
    data_r2_lidar = np.loadtxt("./data/run_data_robot2_lidar.txt")
    data_r1_6Gsensor = np.loadtxt("./data/run_data_robot1_6Gsensor.txt")
    data_r2_6Gsensor = np.loadtxt("./data/run_data_robot2_6Gsensor.txt")
    local_error_ideal, global_error_r1_ideal, global_error_r2_ideal, odometry_error_r1_ideal, odometry_error_r2_ideal = computer_errors(data_r1_ideal, data_r2_ideal)
    local_error_odom, global_error_r1_odom, global_error_r2_odom, odometry_error_r1_odom, odometry_error_r2_odom = computer_errors(data_r1_odom, data_r2_odom)
    local_error_lidar, global_error_r1_lidar, global_error_r2_lidar, odometry_error_r1_lidar, odometry_error_r2_lidar = computer_errors(data_r1_lidar, data_r2_lidar)
    local_error_6Gsensor, global_error_r1_6Gsensor, global_error_r2_6Gsensor, odometry_error_r1_6Gsensor, odometry_error_r2_6Gsensor = computer_errors(data_r1_6Gsensor, data_r2_6Gsensor)

    """
    plt.plot(local_error_ideal, label="local_error_ideal")
    plt.plot(local_error_odom, label="local_error_odom")
    plt.plot(local_error_lidar, label="local_error_lidar")
    plt.plot(local_error_6Gsensor, label="local_error_6Gsensor")"""

    plt.plot(global_error_r1_ideal, color="blue", label="global_error_ideal")
    plt.plot(global_error_r1_odom, color="red", label="global_error_odom")
    plt.plot(global_error_r1_lidar, color="green", label="global_error_lidar")
    plt.plot(global_error_r1_6Gsensor, color="orange", label="global_error_6Gsensor")
    #plt.plot(global_error_r2_ideal, color="blue")
    #plt.plot(global_error_r2_odom, color="red")
    #plt.plot(global_error_r2_lidar, color="green")
    #plt.plot(global_error_r2_6Gsensor, color="orange")

    """
    plt.plot(odometry_error_r1_ideal, color="blue", label="local_error_ideal")
    plt.plot(odometry_error_r1_odom, color="red", label="local_error_odom")
    plt.plot(odometry_error_r1_lidar, color="green", label="local_error_lidar")
    plt.plot(odometry_error_r1_6Gsensor, color="orange", label="local_error_6Gsensor")
    plt.plot(odometry_error_r2_ideal, color="blue")
    plt.plot(odometry_error_r2_odom, color="red")
    plt.plot(odometry_error_r2_lidar, color="green")
    plt.plot(odometry_error_r2_6Gsensor, color="orange")"""

    plt.xlabel("samples")
    plt.ylabel("millimeters")
    plt.grid()
    plt.legend()
    plt.savefig("odometry_error.pdf")
    print("plotted")

if __name__=="__main__":
    #plot_path()
    plot_err()