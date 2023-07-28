"""configuration file"""


class directory:
    tb_log = "./log/"
    saved_models = "./models/"
    exp_name = "vpn_coop/" #"vpn_32/"
    model_name = "VPN-VProp_Arena_S32"
    tb_log_name = model_name+"_v"

class env:
    id = "GridWorld_Single-v1"
    n_envs = 16
    size = 32+2
    obs_prob = 0.2
    max_dist = 64
    map_mode="arena"
    step_reward = -0.01
    min_dist=1
    render_mode = "human"
    render_fps = 8


class train:
    vi_k=2
    lr = 0.01
    max_grad_norm = 10
    n_steps = 8
    cv_thr = 0.2
    log_int = 8
    total_timesteps = n_steps*env.n_envs*log_int*2000 #n_steps*env.n_envs*log_int=5120
    values_callback = n_steps*env.n_envs*log_int*2
    pbar = True
    verbose = 0