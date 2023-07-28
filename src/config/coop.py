"""configuration file"""
import numpy as np

class directory:
    tb_log = "./log/"
    saved_models = "./models/"
    exp_name = "ROS/"
    model_name = "VPN-VProp_ROS32_Coop"
    tb_log_name = model_name+"_v"

class env: 
    id = "GridWorld-v0"
    n_envs = 16
    agents = np.array([[1.5, 180], [1.5, 0]])
    #agents = np.array([[0., 0.]])
    size = 32+2
    obs_prob = 0.05
    move_obs_prob = 0.0
    goal_shape = (5, 3)
    max_dist = 64
    min_dist= 4
    spawn_starting_dist = 4
    extra_obs_agents = 0
    map_mode="arena"
    step_reward = -0.01
    render_mode = "human"
    render_fps = 4

class train:
    vi_k=10
    lr = 0.01
    max_grad_norm = 10
    n_steps = 8
    cv_thr = 2.0
    log_int = 8
    total_timesteps = n_steps*env.n_envs*log_int*10000 #n_steps*env.n_envs*log_int=5120
    values_callback = n_steps*env.n_envs*log_int*2
    pbar = True
    verbose = 0


"""
class directory:
    tb_log = "./log/"
    saved_models = "./models/"
    exp_name = "vpn_coop/"
    model_name = "VPN-VProp_Arena_C32"
    tb_log_name = model_name+"_v"

class env:
    id = "GridWorld_Multi-v0"
    n_envs = 16
    size = 32+2
    obs_prob = 0.05
    max_dist = 64
    step_reward = -0.01
    map_mode="arena"
    min_dist=4
    render_mode = "human"
    render_fps = 1

class train:
    vi_k=5
    lr = 0.01
    max_grad_norm = 10
    n_steps = 8
    cv_thr = 1.0
    log_int = 8
    total_timesteps = n_steps*env.n_envs*log_int*5000 #n_steps*env.n_envs*log_int=5120
    values_callback = n_steps*env.n_envs*log_int*2
    pbar = True
    verbose = 0
"""