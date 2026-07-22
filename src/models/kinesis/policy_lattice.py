############################################################################################################
### Functions are from https://github.com/Khrylx/PyTorch-RL
############################################################################################################

import torch.nn as nn
import numpy as np
import os
import torch
import mujoco
from models.kinesis.mlp import MLP
from models.kinesis.running_norm import RunningNorm
from torch.distributions import MultivariateNormal
from models.kinesis.policy import Policy
import logging

logger = logging.getLogger(__name__)


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class PolicyLattice(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        net_out_dim=None,
        units=(2048, 1536, 1024, 1024, 512, 512),
        activation="silu",
        fix_std=False,
        log_std=0,
        clip_actions=True,
    ):
        print(
            "###################### \n Using policy: lattice \n ######################"
        )
        super().__init__()
        self.type = "lattice"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.norm = RunningNorm(state_dim)

        policy_hsize = units
        policy_htype = activation
        fix_std = fix_std
        log_std = log_std
        self.net = net = MLP(state_dim, policy_hsize, policy_htype)

        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + units[-1]) * log_std,
            requires_grad=not fix_std,
        )
        self.actions_low = torch.tensor(action_space.low, device=self.device)
        self.actions_high = torch.tensor(action_space.high, device=self.device)
        self.clip_actions = clip_actions
        self.action_dim = action_dim
        self.latent_dim = units[-1]

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        action_mean = self.action_mean(x)
        std = torch.exp(self.log_std)
        action_var = std[:, : self.action_dim] ** 2
        latent_var = std[:, self.action_dim :] ** 2
        sigma_mat = (self.action_mean.weight * latent_var[..., None, :]).matmul(
            self.action_mean.weight.T
        )
        sigma_mat = (sigma_mat + sigma_mat.mT) / 2
        sigma_mat[
            ..., torch.arange(self.action_dim), torch.arange(self.action_dim)
        ] += action_var
        self.lattice_dist = MultivariateNormal(action_mean, sigma_mat)
        return self.lattice_dist

    def select_action(self, x, data, model, mean_action=False):
        dist = self.forward(x)
        action = dist.loc if mean_action else dist.rsample()
        if self.clip_actions:
            action = rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clip(action, self.actions_low, self.actions_high),
            ).flatten().cpu().numpy()
        activations = target_length_to_activation(action, data, model)

        activations = np.clip(activations, 0.0005527786369235996, 0.9241418199787566)
        myosuite_activations = 0.5 - 0.2 * np.log(1 / activations - 1)
        return myosuite_activations

    def get_log_prob(self, x, value):
        dist = self.forward(x)
        return dist.log_prob(value).unsqueeze(1)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {"std_id": std_id, "std_index": std_index}

    def preprocess_actions(self, actions: np.ndarray) -> np.ndarray:

        if self.clip_actions:
            actions = rescale_actions(
                self.actions_low,
                self.actions_high,
                np.clip(actions, self.actions_low, self.actions_high),
            )
        return actions

    def load(self, path):
        """
        Load a checkpoint based on the specified path.

        Args:
            path (str): Path to the checkpoint to load.
        """
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.set_full_state_weights(state)
            logger.info(f"Loaded checkpoint from {path}")
        else:
            raise ValueError(f"Checkpoint {path} does not exist.")

    def set_full_state_weights(self, state):
        """
        Load the full state, including network weights and optimizer states.

        Args:
            state (dict): Comprehensive state including networks, optimizers, epoch, and frame count.
        """
        self.load_state_dict(state["policy"])
        self.epoch = state["epoch"]
        print(
            f"==============================Loading checkpoint model: Epoch {self.epoch}=============================="
        )
        logger.info(f"Loaded checkpoint model: Epoch {self.epoch}")


def force_to_activation(forces, model, data):
    """
    Converts actuator forces to activation levels for each actuator in the Mujoco model.

    Args:
        forces (np.ndarray): Array of forces applied to the actuators.
        model: The Mujoco model containing actuator properties.
        data: The Mujoco data structure with runtime actuator states.

    Returns:
        list: Activation levels for each actuator, clipped between 0 and 1.
    """
    activations = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator, :9]
        prmg = model.actuator_gainprm[idx_actuator, :9]
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        activations.append(np.clip((forces[idx_actuator] - bias) / gain, 0, 1))

    return activations


def target_length_to_force(lengths: np.ndarray, data, model) -> list:
    """
    Converts target muscle lengths to forces using a PD control law.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        list: Clipped forces for each actuator, constrained by peak force.
    """
    forces = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        peak_force = model.actuator_biasprm[idx_actuator, 2]
        kp = 5 * peak_force
        kd = 0.1 * kp
        force = kp * (lengths[idx_actuator] - length) - kd * velocity
        clipped_force = np.clip(force, -peak_force, 0)
        forces.append(clipped_force)

    return forces


def target_length_to_activation(lengths: np.ndarray, data, model) -> np.ndarray:
    """
    Converts target lengths to activation levels via force computation.

    Args:
        lengths (np.ndarray): Target lengths for the actuators.
        data: Mujoco data structure containing current actuator states.
        model: Mujoco model containing actuator properties.

    Returns:
        np.ndarray: Activation levels for each actuator, clipped between 0 and 1.
    """
    forces = target_length_to_force(lengths, data, model)
    activations = force_to_activation(forces, model, data)
    return np.clip(activations, 0, 1)


def action_to_target_length(action: np.ndarray, model) -> list:
    """
    Maps actions to target lengths for actuators based on their length ranges.

    Args:
        action (np.ndarray): Action values in the range [-1, 1].
        model: Mujoco model containing actuator length range properties.

    Returns:
        list: Target lengths for each actuator.
    """
    target_lengths = []
    for idx_actuator in range(model.nu):
        # Set high to max length and low=0
        hi = model.actuator_lengthrange[idx_actuator, 1]
        lo = 0
        target_lengths.append((action[idx_actuator] + 1) / 2 * (hi - lo) + lo)
    return target_lengths
