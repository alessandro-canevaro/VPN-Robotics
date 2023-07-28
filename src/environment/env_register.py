"""Script used to register my custom environments into gym
"""

import gym

# from src.environment.gridworld import GridWorldEnv

gym.envs.register(
    id="GridWorld_Single-v1",
    entry_point="src.environment.gridworld:GridWorldEnv",
    max_episode_steps=200,
    kwargs=dict(),
)

gym.envs.register(
    id="GridWorld_Multi-v0",
    entry_point="src.environment.gridworld:CooperativeGridWorldEnv",
    max_episode_steps=200,
    kwargs=dict(),
)

gym.envs.register(
    id="GridWorld-v0",
    entry_point="src.environment.gridworld:GridWorldEnvV2",
    max_episode_steps=150,
    kwargs=dict(),
)