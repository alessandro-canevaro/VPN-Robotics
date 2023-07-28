import gym
import numpy as np
from src.environment import env_register
from src.environment.gridworld import GridWorldEnv
import keyboard
import time
from stable_baselines3 import A2C

from src.models.MultiAgentA2C import MAA2C
from src.config import single as s
from src.config import coop as c
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.policies import obs_as_tensor

def single():
    environment = gym.make(
        s.env.id,
        render_mode=s.env.render_mode,
        render_fps=s.env.render_fps,
        size=s.env.size,
        wall_prob=s.env.wall_prob,
        max_dist=s.env.max_dist,
        min_dist=s.env.min_dist,
        is_train=False,
        agent_loc=s.env.agent_location,
        target_loc=s.env.target_location,
        step_reward=s.env.step_reward,
    )

    model = A2C.load(s.directory.saved_models + s.directory.exp_name + s.directory.model_name)

    model.policy.mlp_extractor.vi_k = 65
    model.policy.observation_space = environment.observation_space

    for i in range(10):
        obs = environment.reset()
        for j in range(200):
            action, _state = model.predict(obs)
            
            """
            obs = obs_as_tensor(obs, model.policy.device).reshape((1, 3, 16, 16))
            dis = model.policy.get_distribution(obs)
            probs = dis.distribution.probs
            probs_np = probs.detach().numpy()
            print(probs_np)
            """
            #action_name = {0: "right", 1: "down", 2: "left", 3: "up"}[action.tolist()]

            values = model.policy.mlp_extractor.last_values[0, :, :]
            # print(f"j: {j}, action: {action_name}")
            # print(obs)
            # print(values)
            
            """
            plt.clf()
            plt.imshow(values)

            for (j, i), label in np.ndenumerate(values):
                if obs[1, j, i] == 1:
                    plt.text(
                        i,
                        j,
                        "A",
                        ha="center",
                        va="center",
                        color="r",
                        fontweight="bold",
                        size="x-large",
                    )
                elif obs[2, j, i] == 1:
                    plt.text(
                        i,
                        j,
                        "G",
                        ha="center",
                        va="center",
                        color="r",
                        fontweight="bold",
                        size="x-large",
                    )
                else:
                    plt.text(i, j, "{:.2f}".format(label), ha="center", va="center")

            plt.colorbar()
            plt.pause(0.01)
            """
            obs, reward, terminated, info = environment.step(action.tolist())

            # print(f"j: {j}, R: {reward}, terminated: {terminated}")

            #environment.render()

            if terminated:
                break
        time.sleep(1)

    plt.show(block=False)
    plt.close()
    environment.close()

def coop():
    environment = gym.make(
        c.env.id,
        agents = c.env.agents,
        render_mode=c.env.render_mode,
        render_fps=c.env.render_fps,
        size=c.env.size,
        obstacles_prob=c.env.obs_prob,
        move_obstacles_prob=0.3,#c.env.move_obs_prob,
        goal_shape = c.env.goal_shape,
        spawn_max_dist=c.env.max_dist,
        spawn_min_dist=15,#c.env.min_dist,
        spawn_starting_dist=15,#c.env.spawn_starting_dist,
        extra_obs_agents=c.env.extra_obs_agents,
        step_reward=c.env.step_reward,
        map_mode = c.env.map_mode,
    )

    #record video for the second episode    
    #environment = gym.wrappers.RecordVideo(environment, 'video', video_length=1500, episode_trigger = lambda x: x < 10)

    model = MAA2C.load(c.directory.saved_models + c.directory.exp_name + c.directory.model_name)

    model.policy.mlp_extractor.vi_k = 33
    model.policy.observation_space = environment.observation_space

    for i in range(10):
        obs = environment.reset()
        for j in range(200):
            obs1, obs2 = model._split_obs(np.expand_dims(obs, axis=0))
            a1, _state = model.predict(obs1)
            a2, _state = model.predict(obs2)
            
            """
            obs = obs_as_tensor(obs, model.policy.device).reshape((1, 3, 16, 16))
            dis = model.policy.get_distribution(obs)
            probs = dis.distribution.probs
            probs_np = probs.detach().numpy()
            print(probs_np)
            """
            #action_name = {0: "right", 1: "down", 2: "left", 3: "up"}[action.tolist()]

            values = model.policy.mlp_extractor.last_values[0, :, :]
            # print(f"j: {j}, action: {action_name}")
            # print(obs)
            # print(values)
            
            
            plt.clf()
            plt.imshow(values)
            """
            for (j, i), label in np.ndenumerate(values):
                if obs[1, j, i] == 1:
                    plt.text(
                        i,
                        j,
                        "A",
                        ha="center",
                        va="center",
                        color="r",
                        fontweight="bold",
                        size="x-large",
                    )
                elif obs[2, j, i] == 1:
                    plt.text(
                        i,
                        j,
                        "G",
                        ha="center",
                        va="center",
                        color="r",
                        fontweight="bold",
                        size="x-large",
                    )
                else:
                    plt.text(i, j, "{:.2f}".format(label), ha="center", va="center")
            """
            plt.colorbar()
            plt.pause(0.01)
            
            
            obs, reward, terminated, info = environment.step(np.concatenate((a1, a2)).tolist())

            # print(f"j: {j}, R: {reward}, terminated: {terminated}")
            #environment.move_obstacles()

            #environment.render()

            if terminated:
                break
        time.sleep(1)

    plt.show(block=False)
    plt.close()
    environment.close()

if __name__=="__main__":
    #single()
    coop()
    print("All done")
