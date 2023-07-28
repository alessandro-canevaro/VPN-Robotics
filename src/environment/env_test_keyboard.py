import gym
import numpy as np
from src.environment import env_register
import keyboard
from src.config import single as s
from src.config import coop as c
import time
import seaborn_image as isns
import matplotlib.pyplot as plt
import seaborn as sns

def single():
    env = gym.make("GridWorld_Single\-v1", render_mode="human", size=18, obstacles_prob=0.3, max_dist=32, min_dist=1, is_train=False, map_mode="random")

    for ep in range(5):
        obs = env.reset()
        #print(obs)
        for i in range(200):
            time.sleep(0.15)
            a1 = -1
            while a1 < 0:
                key = keyboard.read_key()
                try:
                    a1 = {"right": 4, "down": 6, "left": 3, "up": 1}[key]
                except KeyError:
                    pass
        
            observation, reward, terminated, info = env.step(a1)
            print(f"i: {i}, ", reward, terminated, info)
            env.render()

            if terminated:
                env.reset()
                break

    env.close()

def cooperative():
    env = gym.make(
        c.env.id,
        render_mode=c.env.render_mode,
        render_fps=c.env.render_fps,
        size=c.env.size,
        obstacles_prob=c.env.obs_prob,
        max_dist=c.env.max_dist,
        min_dist=c.env.min_dist,
        is_train=False,
        step_reward=c.env.step_reward,
        map_mode = c.env.map_mode,
    )

    obs = env.reset()
    print(obs[0], obs[1], obs[2], obs[3], obs[4])
    for i in range(200):
        time.sleep(0.15)
        a1 = -1
        while a1 < 0:
            key = keyboard.read_key()
            try:
                a1 = {"right": 4, "down": 6, "left": 3, "up": 1}[key]
            except KeyError:
                pass
        a2 = -1
        while a2 < 0:
            key = keyboard.read_key()
            try:
                a2 = {"d": 4, "s": 6, "a": 3, "w": 1}[key]
            except KeyError:
                pass

        observation, reward, terminated, info = env.step([a1, a2])
        print(f"i: {i}, ", reward, terminated, info)
        env.render()

        if terminated:
            env.reset()
            break

    env.close()


def plot_env(data):
    isns.set_context("notebook")

    # change image related settings
    isns.set_image(cmap="mako", despine=True)  # set the colormap and despine the axes
    #isns.set_scalebar(color="red")  # change scalebar color
    #sns.color_palette("Blues", as_cmap=True)

    key, val = list(data.keys()), list(data.values())
    rows, cols = 2, 4
    f, axarr = plt.subplots(rows, cols, sharex=True, sharey=True)
    for row in range(rows):
        for col in range(cols):
            isns.imshow(val[row*4+col], ax=axarr[row, col], origin="upper", cbar=False)
            #axarr[i%2, i%3].matshow(item, cmap="magma")
            axarr[row, col].title.set_text(key[row*4+col])
    plt.tight_layout()
    plt.show()

def new():
    env = gym.make(
        "GridWorld-v0",
        agents = np.array([[1.5, 180], [1.5, 0]]),
        render_mode="human",
        render_fps=4,
        size=8+2,
        obstacles_prob=0.0,
        move_obstacles_prob=0.0,
        spawn_max_dist=16,
        spawn_min_dist=1,
        spawn_starting_dist=1,
        step_reward=c.env.step_reward,
        map_mode = "random",
    )

    obs = env.reset()
    plot_env(env.get_plot_data())
    for i in range(200):
        actions = []
        while len(actions) < len(env.agents):
            print(f"action for agent {len(actions)+1}: ", end="")
            time.sleep(0.15)
            key = keyboard.read_key()
            try:
                a = {"q":0, "w":1, "e":2, "a":3, "d":4, "z":5, "x":6, "c":7}[key]
                print(a)
            except KeyError:
                print("selected action is not valid")
                continue
            actions.append(a)

        observation, reward, terminated, info = env.step(actions)
        #print(f"i: {i}, ", reward, terminated, info)
        #env.render()

        if terminated:
            env.reset()
            break

    env.close()


if __name__=="__main__":
    #single()
    #cooperative()
    new()
    print("All done")
