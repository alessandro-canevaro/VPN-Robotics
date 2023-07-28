from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym import spaces
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.VPN_basic import Identity, VPN, CustomNetworkVProp
import numpy as np


class CustomNetworkVPropCoop(CustomNetworkVProp):
  
    def __init__(
        self,
        vi_k: int = 16,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1,
        neighbor_size: int = 7,
        in_channels: int = 4,
    ):
        super(CustomNetworkVProp, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.vi_k = vi_k
        self.maze_size = None# maze_size
        self.neighbor_size = neighbor_size
        self.in_channels = in_channels

        self.PhiConv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.PhiConv2 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.Logit = torch.nn.Linear((self.in_channels+1) * self.neighbor_size * self.neighbor_size, self.latent_dim_pi)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    #def _compute_values(self, obs: torch.Tensor) -> torch.Tensor:
        #reduced_obs = obs[:, [0, 1, 4], :, :]
    #    return super()._compute_values(reduced_obs)

    def _v_prop(self, rin: torch.Tensor, rout: torch.Tensor, p: torch.Tensor, actions=[(0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (0, 2), (2, 0), (2, 2)]) -> torch.Tensor:
        return super()._v_prop(rin, rout, p, actions)

    def _compute_logits(self, obs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Forward of the policy net (logits)

        :param obs: Observation
        :param values: Values matrix
        :return: logits, output of the latent layer of the policy net
        """
        B = obs.shape[0]

        # concatenate values to obs
        #obs_val = obs.to(torch.int64)
        #obs_val[:, 1, :, :] = torch.bitwise_or(obs_val[:, 1, :, :], obs_val[:, 2, :, :])
        #obs_val[:, 1, :, :] = torch.bitwise_or(obs_val[:, 1, :, :], obs_val[:, 3, :, :])
        #obs_val = obs_val[:,[0, 1, 4], :, :].to(torch.float32)
        obs_val = torch.cat((obs, values.unsqueeze(dim=1)), dim=1)

        padding = self.neighbor_size//2
        padded_obs_val = torch.nn.functional.pad(obs_val, (padding, padding, padding, padding, 0, 0, 0, 0), "constant", 0)

        pos = torch.nonzero(obs_val[:, 1, :, :])[:, 1:]
        if pos.numel() == 0:  # no agent
            assert False, "NO AGENT POS"

        selected_obs_val = torch.zeros((B, self.in_channels+1, self.neighbor_size, self.neighbor_size))#.to("cuda")
        for b in range(B):
            i, j = pos[b, 0] + padding, pos[b, 1] + padding
            selected_obs_val[b, :, :, :] = padded_obs_val[b, :, i - padding : i + 1 + padding, j - padding : j + 1 + padding]

        logits = self.Logit(torch.flatten(selected_obs_val, start_dim=1))
        logits = self.relu(logits)
        return logits

class VPN_coop(VPN):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Callable[[float], float], vi_k: int, net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, activation_fn: Type[nn.Module] = nn.Tanh, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, vi_k, net_arch, activation_fn, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkVPropCoop(self.vi_k)
