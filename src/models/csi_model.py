from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

# def CSI_wrapper(policy: ActorCriticPolicy, projection: torch.Tensor, trainable: bool = True):
#     """
#     CSI policy wrapper
#     Project the action to the CSI subspace
#     """
    
#     # add projection to the policy parameters
#     policy.projection = torch.nn.Parameter(projection, requires_grad=trainable)
        
#     def forward(obs: torch.Tensor, deterministic: bool = False) :
#         action, value, log_prob = policy(obs, deterministic=deterministic)
#         action = projection.transpose(-1, -2) @ projection @ action
#         return action, value, log_prob
    
#     policy.forward = forward
#     return policy

class CSIActionNet(ActorCriticPolicy):
    """
    Wraps an existing action_net and post-projects its output:
        action_proj = mean_action @ P^T @ P
    where mean_action has shape (..., A) and P has shape (r, A).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = nn.Parameter(torch.eye(self.action_space.shape[0], device=self.device), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(self.action_space.shape[0], device=self.device), requires_grad=False)
        self.optimizer.param_groups[0]["params"].append(self.projection)
        self.optimizer.param_groups[0]["params"].append(self.mean)
        self.subspace = self.action_space.shape[0]
    
    def change_projection(self, projection: torch.Tensor = None, mean: torch.Tensor = None, subspace: int = None, trainable: bool = True):
        if projection is not None:
            self.projection.copy_(projection.to(self.device))
        if mean is not None:
            self.mean.copy_(mean.to(self.device))
        if subspace is not None:
            self.subspace = subspace
        # make projection trainable
        self.projection.requires_grad = trainable
        self.mean.requires_grad = trainable
        # self.projection = nn.Parameter(projection, requires_grad=trainable)
        # self.optimizer.param_groups[0]["params"].append(self.projection)
        # self.mean = nn.Parameter(mean, requires_grad=trainable)
        # self.optimizer.param_groups[0]["params"].append(self.mean)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        mean_action = self.action_net(latent_pi)                    # (..., A)
        P = self.projection[:self.subspace]                                         # (r, A)
        mean_action_projected = self.mean + (mean_action-self.mean) @ P.transpose(-1, -2) @ torch.linalg.inv(P @ P.transpose(-1, -2) + torch.eye(P.shape[-2], device=P.device) * 1e-6) @ P   # (..., A)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_action_projected, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_action_projected)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_action_projected)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_action_projected)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_action_projected, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

# def CSI_wrapper(policy: ActorCriticPolicy, projection: torch.Tensor, trainable: bool = True):
#     # register parameter on the module (gets into state_dict)
#     policy.action_net.projection = nn.Parameter(projection, requires_grad=trainable)
#     # adding projection to optimizer
#     policy.optimizer.param_groups[0]["params"].append(policy.action_net.projection)
#     # policy.optimizer.add_param_group({"params": [policy.projection]})
#     # import ipdb
#     # ipdb.set_trace()
#     # save original forward
#     orig_forward = policy.action_net.forward

#     def new_forward(self, feature: torch.Tensor):
#         mean_action = orig_forward(feature)
#         P = self.projection
#         action_projected = mean_action @ P.transpose(-1, -2) @ P
#         return action_projected

#     policy.action_net.forward = MethodType(new_forward, policy.action_net)
#     return policy

        