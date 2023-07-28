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
from tqdm import tqdm
import multiprocess as mp
import matplotlib.pyplot as plt
import copy

if __name__=="__main__":

    def run_episode_batch(args):
        i, model, eps = args
        #print(f"process {i}, eps: {eps}")
        environment = gym.make(
            c.env.id,
            render_mode=None,#"human",#c.env.render_mode,
            render_fps=c.env.render_fps,
            size=c.env.size,
            obstacles_prob=c.env.obs_prob,
            max_dist=c.env.max_dist,
            min_dist=c.env.min_dist,
            is_train=False,
            map_mode=c.env.map_mode,
            step_reward=c.env.step_reward,
        )
        success = []
        length = []
        opt_length = []
        for i in range(eps):
            obs = environment.reset()
            for j in range(20):
                obs1, obs2 = model._split_obs(np.expand_dims(obs, axis=0))
                a1, _state = model.predict(obs1)
                a2, _state = model.predict(obs2)
                obs, reward, terminated, info = environment.step(np.concatenate((a1, a2)).tolist())
                environment.move_obstacles()

                if terminated:
                    break

            success.append(j+1 < 20)
            if j+1 < 20:
                opt_length.append(environment.optimal_episode_length)
                length.append(j+1)
        
        environment.close()
        return opt_length, length, success

    model = MAA2C.load(c.directory.saved_models + c.directory.exp_name + c.directory.model_name)

    model.policy.mlp_extractor.vi_k = 33

    episodes = 1024
    cpus = mp.cpu_count()
    success = []
    length = []
    opt_length = []

    args = [(i, model, episodes//cpus) for i in range(cpus)]

    with tqdm(total=episodes) as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for opt, actual, succ in pool.imap_unordered(run_episode_batch, args):
                opt_length += opt
                length += actual
                success += succ
                pbar.update(episodes//cpus)

    length = np.array(length)
    opt_length = np.array(opt_length)
    success = np.array(success)
    #print(length, opt_length, success)
        
    length_mean, length_std = np.mean(length), np.std(length)
    opt_length_mean, opt_length_std = np.mean(opt_length), np.std(opt_length)
    succ_rate = sum(success)/len(success)
    #print(length_mean, opt_length_mean, succ_rate)

    rel_diff = (length-opt_length)/opt_length * 100
    rel_diff_mean, rel_diff_std = np.mean(rel_diff), np.std(rel_diff)


    print(f"Test Results on {episodes} episodes:")
    print(f"Success rate: {succ_rate*100:.2f}%")
    print(f"Episode mean length: {length_mean:.2f} +- {length_std:.2f}")
    print(f"Episode optimal mean length: {opt_length_mean:.2f} +- {opt_length_std:.2f}")
    print(f"Path length rel. diff.: {rel_diff_mean:.2f}% +- {rel_diff_std:.2f}%")


    import matplotlib.pyplot as plt
    import seaborn as sns
    bins = np.arange(0, np.rint(np.max(length))+1)
    data = np.stack([length, opt_length]).T

    sns.set(style="whitegrid")

    g2 = sns.displot(data, bins=bins, element="step", kde=True, legend=False, discrete=True, stat="probability")
    plt.axvline(length_mean, color='blue', ls='--', lw=2.5)
    plt.axvline(opt_length_mean, color='orange', ls='--', lw=2.5)
    plt.legend(loc='upper right', labels=["Optimal", '_nolegend_', "Actual", '_nolegend_'])
    plt.xlabel("Episode Length")
    plt.xlim([2.5, 20.5])
    g2.fig.set_figwidth(16)
    g2.fig.set_figheight(4)
    plt.tight_layout()
    #plt.show()

    g1 = sns.displot(rel_diff, bins=np.arange(-40, 200, 10), element="step", kde=True, rug=True, legend=False, stat="probability")
    plt.axvline(rel_diff_mean, color='blue', ls='--', lw=2.5)
    plt.xlim([-40, 200])
    plt.legend(loc='upper right', labels=["Rel. Diff.", '_nolegend_'])
    plt.xlabel("Relative difference %")
    g1.fig.set_figwidth(16)
    g1.fig.set_figheight(4)
    plt.tight_layout()
    plt.show()

    print("All done")
