from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.utils import safe_mean
from src.config import single as s
from src.config import single as c
import matplotlib
import numpy as np
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MyCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, max_vi_k: int, save_path: str, name_prefix: str = "rl_model", save_replay_buffer: bool = False, save_vecnormalize: bool = False, verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.max_vi_k = max_vi_k
        self.min_rel_diff = 10000

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}{checkpoint_type}_chk.{extension}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0 and self.model.policy.mlp_extractor.vi_k >= self.max_vi_k:         
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                ep_opt_mean = safe_mean([ep_info["o"] for ep_info in self.model.ep_info_buffer])
            rel_diff = (ep_len_mean-ep_opt_mean)/ep_opt_mean

            if rel_diff < self.min_rel_diff:
                self.min_rel_diff = rel_diff

                model_path = self._checkpoint_path(extension="zip")
                self.model.save(model_path)
                if self.verbose >= 2:
                    print(f"Saving model checkpoint to {model_path}")

                if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                    # If model has a replay buffer, save it too
                    replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                    self.model.save_replay_buffer(replay_buffer_path)
                    if self.verbose > 1:
                        print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

                if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                    # Save the VecNormalize statistics
                    vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                    self.model.get_vec_normalize_env().save(vec_normalize_path)
                    if self.verbose >= 2:
                        print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True

class LogValues(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        imgs = self.model.policy.mlp_extractor.getValuesParams()
        if self.model.policy.mlp_extractor.__class__.__name__ != "CustomNetworkMVProp":
            names = ["Values", "Rin", "Rout", "P"]
        else:
            names = ["Values", "R", "P"]

        fig, axs = plt.subplots(1, 4, figsize=(11, 3))
        for ax, img, name in zip(axs, imgs, names):
            ax.imshow(img, cmap="inferno")
            ax.set_title(name)
        plt.tight_layout()

        # Close the figure after logging it
        self.logger.record(
            "figures/values",
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        plt.close()
        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        d1, d2, d3 = dict(vars(s.env)), dict(vars(s.train)), dict(vars(s.directory))
        d1.update(d2)
        d1.update(d3)

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
        }
        hparam_dict.update(d1)

        hparam_dict.pop('__module__', None)
        hparam_dict.pop('__dict__', None)
        hparam_dict.pop('__weakref__', None)
        hparam_dict.pop('__doc__', None)

        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0.0,
            "rollout/ep_opt_len_mean": 0.0,
            "rollout/ep_rel_diff_mean": 0.0,
            "rollout/ep_rew_mean": 0.0,
            "rollout/episodes": 0.0,
            "rollout/max_distance": 0.0,
            "rollout/vi_k": 0.0,
            "time/fps": 0.0,
            "train/entropy_loss": 0.0,
            "train/explained_variance": 0.0,
            "train/learning_rate": 0.0,
            "train/policy_loss": 0.0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
    

class IncreaseCurriculum(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    def __init__(self, threshold=0.2, max_dist=1, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.increase_threshold = threshold
        self.max_dist = max_dist

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            #ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            ep_opt_mean = safe_mean([ep_info["o"] for ep_info in self.model.ep_info_buffer])
        else:
            return True

        rel_diff = (ep_len_mean-ep_opt_mean)/ep_opt_mean
        self.logger.record("rollout/ep_opt_len_mean", ep_opt_mean)
        self.logger.record("rollout/ep_rel_diff_mean", rel_diff)
        self.logger.record("rollout/max_distance", self.max_dist)
        self.logger.record("rollout/vi_k", self.model.policy.mlp_extractor.vi_k)

        #log number of episodes
        episodes = 0
        for idx in range(self.model.env.num_envs):
                episodes += self.model.env.envs[idx].episode_counter
        self.logger.record("rollout/episodes", episodes)

        if rel_diff < self.increase_threshold:
            for idx in range(self.model.env.num_envs):
                self.max_dist = self.model.env.envs[idx].increase_curriculum()
            self.model.policy.mlp_extractor.vi_k = self.max_dist + 5
            if self.verbose > 0:
                print(f"New Max Distance: {self.max_dist} (at timestep: {self.num_timesteps})")

        return True
