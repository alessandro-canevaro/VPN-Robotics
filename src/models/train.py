from src.environment import env_register
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from src.models.VPN_basic import VPN
from stable_baselines3.common.policies import ActorCriticPolicy
from src.config import single as s
from src.config.coop import directory, train, env
from stable_baselines3.common.callbacks import EveryNTimesteps, CallbackList
from src.models.callbacks import LogValues, HParamCallback, IncreaseCurriculum, MyCheckpointCallback
from typing import Callable
from src.models.MultiAgentA2C import MAA2C
from src.models.VPN_coop import VPN_coop

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
        #return initial_value

    return func


def training():
    environment = make_vec_env(
        env.id,
        env_kwargs={
            "agents": env.agents,
            "size": env.size,
            "obstacles_prob": env.obs_prob,
            "move_obstacles_prob": env.move_obs_prob,
            "goal_shape": env.goal_shape,
            "spawn_max_dist": env.max_dist,
            "spawn_min_dist": env.min_dist,
            "spawn_starting_dist": env.spawn_starting_dist,
            "extra_obs_agents": env.extra_obs_agents,
            "map_mode": env.map_mode,
            "step_reward": env.step_reward,
            "render_mode": None,
            "render_fps": env.render_fps,
        },
        n_envs=env.n_envs,
        monitor_kwargs={"info_keywords": ("o")}
    )

    logval = LogValues()
    log_event_callback = EveryNTimesteps(n_steps=train.values_callback, callback=logval)
    hparams_callback = HParamCallback()
    curriculum_callback = IncreaseCurriculum(threshold=train.cv_thr, max_dist=env.min_dist, verbose=train.verbose)
    eval_callback = EveryNTimesteps(n_steps=train.log_int*train.n_steps*env.n_envs, callback=curriculum_callback)  #
    save_callback = MyCheckpointCallback(save_freq=train.log_int*train.n_steps*env.n_envs, max_vi_k=env.max_dist+1, save_path=directory.saved_models + directory.exp_name, name_prefix=directory.model_name, verbose=train.verbose)
    # Create the callback list
    callback = CallbackList([log_event_callback, hparams_callback, save_callback, eval_callback])

    model = A2C(
        VPN_coop,
        environment,
        policy_kwargs={"vi_k": train.vi_k},
        learning_rate=linear_schedule(train.lr),
        n_steps=train.n_steps,
        max_grad_norm=train.max_grad_norm,
        tensorboard_log=directory.tb_log + directory.exp_name,
        verbose=train.verbose,
        device="cpu",
    )
    model = MAA2C.load("./models/ROS/VPN-VProp_ROS32_Single", env=environment, device="cpu", reset_num_timesteps=True)    
    #model.policy.mlp_extractor.vi_k = 5

    model.learn(
        total_timesteps=train.total_timesteps,
        log_interval=train.log_int,
        progress_bar=train.pbar,
        tb_log_name=directory.tb_log_name,
        callback=callback,
    )

    model.save(directory.saved_models + directory.exp_name + directory.model_name)


def train_single():
    # Parallel environments
    environment = make_vec_env(
        s.env.id,
        env_kwargs={
            "size": s.env.size,
            "obstacles_prob": s.env.obs_prob,
            "max_dist": s.env.max_dist,            
            "is_train": True,
            "step_reward": s.env.step_reward,
            "min_dist": s.env.min_dist,
            "map_mode": s.env.map_mode,
        },
        n_envs=s.env.n_envs,
        monitor_kwargs={"info_keywords": ("o")}
    )

    logval = LogValues()
    log_event_callback = EveryNTimesteps(n_steps=s.train.values_callback, callback=logval)
    hparams_callback = HParamCallback()

    curriculum_callback = IncreaseCurriculum(threshold=s.train.cv_thr, verbose=s.train.verbose)
    eval_callback = EveryNTimesteps(n_steps=s.train.log_int*s.train.n_steps*s.env.n_envs, callback=curriculum_callback)  #

    #save_callback = MyCheckpointCallback(save_freq=s.train.log_int*s.train.n_steps*s.env.n_envs, max_vi_k=s.env.max_dist+1, save_path=s.directory.saved_models + s.directory.exp_name, name_prefix=s.directory.model_name, verbose=s.train.verbose)


    # Create the callback list
    callback = CallbackList([log_event_callback, hparams_callback, eval_callback])

    model = A2C(
        VPN_coop,
        environment,
        policy_kwargs={"vi_k": s.train.vi_k},
        learning_rate=linear_schedule(s.train.lr),
        n_steps=s.train.n_steps,
        max_grad_norm=s.train.max_grad_norm,
        tensorboard_log=s.directory.tb_log + s.directory.exp_name,
        verbose=s.train.verbose,
        device="cpu",
    )

    #model = A2C.load(s.directory.saved_models + s.directory.exp_name + s.directory.model_name, env=environment, device="cpu", reset_num_timesteps=True)
    #model.policy.mlp_extractor.vi_k = 50

    model.learn(
        total_timesteps=s.train.total_timesteps,
        log_interval=s.train.log_int,
        progress_bar=s.train.pbar,
        tb_log_name=s.directory.tb_log_name,
        callback=callback,
    )

    model.save(s.directory.saved_models + s.directory.exp_name + s.directory.model_name)

def train_coop():
    # Parallel environments
    environment = make_vec_env(
        c.env.id,
        env_kwargs={
            "size": c.env.size,
            "obstacles_prob": c.env.obs_prob,
            "max_dist": c.env.max_dist,
            "is_train": True,
            "step_reward": c.env.step_reward,
            "min_dist": c.env.min_dist,
            "map_mode": c.env.map_mode,
        },
        n_envs=c.env.n_envs,
        monitor_kwargs={"info_keywords": ("o")}
    )

    logval = LogValues()
    log_event_callback = EveryNTimesteps(n_steps=c.train.values_callback, callback=logval)
    hparams_callback = HParamCallback()

    curriculum_callback = IncreaseCurriculum(threshold=c.train.cv_thr, max_dist=c.env.min_dist, verbose=c.train.verbose)
    eval_callback = EveryNTimesteps(n_steps=c.train.log_int*c.train.n_steps*c.env.n_envs, callback=curriculum_callback)  #

    save_callback = MyCheckpointCallback(save_freq=c.train.log_int*c.train.n_steps*c.env.n_envs, max_vi_k=c.env.max_dist+1, save_path=c.directory.saved_models + c.directory.exp_name, name_prefix=c.directory.model_name, verbose=c.train.verbose)


    # Create the callback list
    callback = CallbackList([log_event_callback, hparams_callback, save_callback, eval_callback])

    model = MAA2C(
        VPN_coop,
        environment,
        policy_kwargs={"vi_k": c.train.vi_k},
        learning_rate=linear_schedule(c.train.lr),
        n_steps=c.train.n_steps,
        max_grad_norm=c.train.max_grad_norm,
        tensorboard_log=c.directory.tb_log + c.directory.exp_name,
        verbose=c.train.verbose,
        device="cpu",
    )

    model = MAA2C.load("./models/vpn_coop/VPN-VProp_Arena_S32", env=environment, device="cpu", reset_num_timesteps=True)    
    model.policy.mlp_extractor.vi_k = 5

    model.learn(
        total_timesteps=c.train.total_timesteps,
        log_interval=c.train.log_int,
        progress_bar=c.train.pbar,
        tb_log_name=c.directory.tb_log_name,
        callback=callback,
    )

    model.save(c.directory.saved_models + c.directory.exp_name + c.directory.model_name)


if __name__=="__main__":
    #train_single()
    #train_coop()
    training()
    print("All done")
