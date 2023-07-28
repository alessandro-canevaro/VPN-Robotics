from stable_baselines3 import A2C
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import random

class MAA2C(A2C):
    def __init__(self, policy: Union[str, Type[ActorCriticPolicy]], env: Union[GymEnv, str], learning_rate: Union[float, Schedule] = 0.0007, n_steps: int = 5, gamma: float = 0.99, gae_lambda: float = 1, ent_coef: float = 0, vf_coef: float = 0.5, max_grad_norm: float = 0.5, rms_prop_eps: float = 0.00001, use_rms_prop: bool = True, use_sde: bool = False, sde_sample_freq: int = -1, normalize_advantage: bool = False, tensorboard_log: Optional[str] = None, policy_kwargs: Optional[Dict[str, Any]] = None, verbose: int = 0, seed: Optional[int] = None, device: Union[th.device, str] = "auto", _init_setup_model: bool = True):
        super().__init__(policy, env, learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm, rms_prop_eps, use_rms_prop, use_sde, sde_sample_freq, normalize_advantage, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)
        try:
            self.episode_q_buffer = [[] for i in range(env.num_envs)]
            self.q_buffer = []
            self.first_step = [True for i in range(env.num_envs)]
            self.old_delta_rt = [0 for i in range(env.num_envs)]
        except AttributeError:
            pass

    def _split_obs(self, obs):
        obs1 = obs[:, [0, 1, 2, 3], :, :]
        obs2 = obs[:, [0, 2, 1, 3], :, :]
        return obs1, obs2
    
    def compute_delta_q(self, v1, v2, r1, r2, dones):
        for idx, done in enumerate(dones):
            delta_vt = v1[idx] - v2[idx]
            new_delta_rt = r1[idx] - r2[idx]
            if self.first_step[idx]:
                self.first_step[idx]=False
            else:
                delta_qt = abs(self.old_delta_rt[idx] + self.gamma * delta_vt)
                self.episode_q_buffer[idx].append(delta_qt)
            self.old_delta_rt[idx] = new_delta_rt

            if done:
                self.first_step[idx]=True
                delta_q = safe_mean(self.episode_q_buffer[idx])
                self.episode_q_buffer[idx] = []
                self.q_buffer.append(delta_q)

                if len(self.q_buffer) > 100:
                    self.q_buffer.pop(0)
                
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                
                obs1, obs2 = self._split_obs(obs_tensor)
                a1, v1, lp1 = self.policy(obs1)
                a2, v2, lp2 = self.policy(obs2)

            a1 = a1.cpu().numpy()
            a2 = a2.cpu().numpy()
            
            # Rescale and perform action
            ca1, ca2 = a1, a2
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                ca1 = np.clip(a1, self.action_space.low, self.action_space.high)
                ca2 = np.clip(a2, self.action_space.low, self.action_space.high)

            #new_obs, rewards, dones, infos = env.step(clipped_actions)
            actions = np.stack((ca1, ca2), axis=-1)
            new_obs, rewards, dones, infos = env.step(actions)

            self.compute_delta_q(v1.ravel().tolist(), v2.ravel().tolist(), rewards.ravel().tolist(), [infos[idx]["reward2"] for idx in range(self.env.num_envs)], dones.ravel().tolist())

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                a1 = a1.reshape(-1, 1)
                a2 = a2.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            #rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            rollout_buffer.add(self._last_obs, a1, rewards, self._last_episode_starts, v1, lp1)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            #values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            obs1, obs2 = self._split_obs(obs_as_tensor(new_obs, self.device))
            v1 = self.policy.predict_values(obs1)

        #rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_returns_and_advantage(last_values=v1, dones=dones)

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        super().train()
        mean_delta_q = safe_mean(self.q_buffer)
        self.logger.record("rollout/delta_q_mean", mean_delta_q)