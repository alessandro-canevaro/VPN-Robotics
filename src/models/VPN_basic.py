from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym import spaces
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomNetworkVProp(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        vi_k: int = 16,
        last_layer_dim_pi: int = 16,
        last_layer_dim_vf: int = 1,
    ):
        super(CustomNetworkVProp, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.vi_k = vi_k
        self.maze_size = None# maze_size

        self.PhiConv = torch.nn.Conv2d(
            in_channels=3,
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
        self.Logit = torch.nn.Linear(3 * 3 * 4, self.latent_dim_pi)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def _compute_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward the value net (Phi + VProp)

        :param obs: Observations
        :return: values, values matrix
        """
        obs = obs.to(torch.float32)

        phi_out = self.relu(self.PhiConv(obs))
        phi_out = self.sigmoid(self.PhiConv2(phi_out))
        rin = phi_out[:, 0, :, :]
        rout = phi_out[:, 1, :, :]
        p = phi_out[:, 2, :, :]

        values = self._v_prop(rin, rout, p)
        self.last_rin = rin
        self.last_rout = rout
        self.last_p = p
        self.last_values = values
        return values

    def getValuesParams(self):
        return (
            self.last_values[0, :, :].to("cpu").detach().numpy(),
            self.last_rin[0, :, :].to("cpu").detach().numpy(),
            self.last_rout[0, :, :].to("cpu").detach().numpy(),
            self.last_p[0, :, :].to("cpu").detach().numpy(),
        )

    def _v_prop(self, rin: torch.Tensor, rout: torch.Tensor, p: torch.Tensor, actions=[(0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (0, 2), (2, 0), (2, 2)]) -> torch.Tensor:
        """
        Value propagation algorithm

        :param rin: reward matrix
        :param rout: reward matrix
        :param p: propagation matrix
        :return: values, values matrix
        """
        values = torch.zeros_like(rin)

        padded_rin = torch.nn.functional.pad(rin, (1, 1, 1, 1, 0, 0), "constant", 0)

        for __ in range(self.vi_k):
            padded_v = torch.nn.functional.pad(values, (1, 1, 1, 1, 0, 0), "constant", 0)

            for h_offset, w_offset in actions:
                shifted_v = padded_v[
                    :,
                    h_offset : h_offset + self.maze_size,
                    w_offset : w_offset + self.maze_size,
                ]
                shifted_rin = padded_rin[
                    :,
                    h_offset : h_offset + self.maze_size,
                    w_offset : w_offset + self.maze_size,
                ]

                #mask = torch.eq(shifted_rin, torch.zeros_like(rin))
                #masked_rout = rout * (1 - mask.int().float())

                nv = p * shifted_v + shifted_rin - rout#masked_rout
                values = values.maximum(nv)

        return values

    def _compute_logits(self, obs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Forward of the policy net (logits)

        :param obs: Observation
        :param values: Values matrix
        :return: logits, output of the latent layer of the policy net
        """
        B = obs.shape[0]

        obs_val = torch.cat((obs, values.unsqueeze(dim=1)), dim=1)  # concatenate values to obs
        padded_obs_val = torch.nn.functional.pad(obs_val, (1, 1, 1, 1, 0, 0, 0, 0), "constant", 0)

        pos = torch.nonzero(obs_val[:, 1, :, :])[:, 1:]
        if pos.numel() == 0:  # no agent
            assert False, "NO AGENT POS"

        selected_obs_val = torch.zeros((B, 4, 3, 3))#.to("cuda")
        for b in range(B):
            i, j = pos[b, 0] + 1, pos[b, 1] + 1
            selected_obs_val[b, :, :, :] = padded_obs_val[b, :, i - 1 : i + 2, j - 1 : j + 2]

        #softmax values
        #selected_val = torch.flatten(selected_obs_val[:, 3, :, :], start_dim=1)
        #selected_val_norm = self.softmax(selected_val).reshape((B, 3, 3))
        #selected_obs_val[:, 3, :, :] = selected_val_norm

        logits = self.Logit(torch.flatten(selected_obs_val, start_dim=1))
        logits = self.relu(logits)
        return logits
        
    def _get_value(self, obs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Extract the value at the agent position

        :param obs: Observation
        :param values: Values matrix
        :return: value, value at agent position
        """
        B = obs.shape[0]

        pos = torch.nonzero(obs[:, 1, :, :])[:, 1:]
        if pos.numel() == 0:  # no agent
            assert False, "NO AGENT POS"

        # Value at the agent position
        values = values.reshape((B, self.maze_size * self.maze_size))
        pos = (self.maze_size * pos[:, 0] + pos[:, 1]).reshape((B, 1))
        value = torch.gather(values, 1, pos).reshape((B, 1))
        return value

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #infer maze size the first time the model is called
        if self.maze_size == None:
            self.maze_size = features.shape[-1]

        values = self._compute_values(features)
        value = self._get_value(features, values)
        logits = self._compute_logits(features, values)

        return logits, value

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        values = self._compute_values(features)
        value = self._get_value(features, values)
        return value

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        #infer maze size the first time the model is called
        if self.maze_size == None:
            self.maze_size = features.shape[-1]

        values = self._compute_values(features)
        logits = self._compute_logits(features, values)
        return logits

class CustomNetworkMVProp(CustomNetworkVProp):
    def __init__(self, vi_k: int = 16, last_layer_dim_pi: int = 16, last_layer_dim_vf: int = 1):
        super().__init__(vi_k, last_layer_dim_pi, last_layer_dim_vf)
        self.PhiConv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )
        self.PhiConv2 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

    def _compute_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward the value net (Phi + MVProp)

        :param obs: Observations
        :return: values, values matrix
        """
        obs = obs.to(torch.float32)

        phi_out = self.relu(self.PhiConv(obs))
        phi_out = self.sigmoid(self.PhiConv2(phi_out))
        r = phi_out[:, 0, :, :]
        p = phi_out[:, 1, :, :]

        values = self._mv_prop(r, p)
        self.last_r = r
        self.last_p = p
        self.last_values = values
        return values

    def getValuesParams(self):
        return (
            self.last_values[0, :, :].to("cpu").detach().numpy(),
            self.last_r[0, :, :].to("cpu").detach().numpy(),
            self.last_p[0, :, :].to("cpu").detach().numpy(),
        )

    def _mv_prop(self, r: torch.Tensor, p: torch.Tensor, actions = [(0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (0, 2), (2, 0), (2, 2)]) -> torch.Tensor:
        """
        Maximum Value propagation algorithm

        :param r: reward matrix
        :param p: propagation matrix
        :return: values, values matrix
        """
        values = r

        for __ in range(self.vi_k):
            padded_v = torch.nn.functional.pad(values, (1, 1, 1, 1, 0, 0), "constant", 0)

            for h_offset, w_offset in actions:
                shifted_v = padded_v[
                    :,
                    h_offset : h_offset + self.maze_size,
                    w_offset : w_offset + self.maze_size,
                ]
                
                nv = r + p * (shifted_v -r)
                values = values.maximum(nv)

        return values

class CustomNetworkMVPropA(CustomNetworkMVProp):
    def __init__(self, vi_k: int = 16, last_layer_dim_pi: int = 8, last_layer_dim_vf: int = 1):
        super().__init__(vi_k, last_layer_dim_pi, last_layer_dim_vf)

    def _compute_logits(self, obs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        pos = torch.nonzero(obs[:, 1, :, :])[:, 1:]
        if pos.numel() == 0:  # no agent
            assert False, "NO AGENT POS"

        selected_val = torch.zeros((B, 3, 3))#.to("cuda")
        for b in range(B):
            i, j = pos[b, 0], pos[b, 1]
            selected_val[b, :, :] = values[b, i - 1 : i + 2, j - 1 : j + 2]

        flat_val = torch.flatten(selected_val, start_dim=1)
        flat_val = torch.cat((flat_val[:, :4], flat_val[:, 5:]), dim=1)
        index = torch.argmax(flat_val, dim=1)
        logits = torch.nn.functional.one_hot(index, 8).to(torch.float32)
        return logits

class Identity(BaseFeaturesExtractor):
    """
    Feature extract that does nothing to the imputs.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, spaces.utils.flatdim(observation_space))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class VPN(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        vi_k: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        self.vi_k = vi_k

        super(VPN, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=Identity,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkVProp(self.vi_k)
        
