import torch
import numpy as np
from typing import Optional
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Tuple
from torch.distributions import Normal
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    StateDependentNoiseDistribution,
    SquashedDiagGaussianDistribution,
    Distribution,
)
from gymnasium import spaces
from models.helpers import NoisyAttention, ReplicateInputAttentionWrapper
from typing import Dict, Any
from stable_baselines3.common.preprocessing import get_action_dim


def make_proba_distribution(
    action_space: spaces.Space,
    use_sde: bool = False,
    dist_kwargs: Optional[Dict[str, Any]] = None,
    use_lattice: bool = False,
    use_bilateral: bool = False,
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if isinstance(action_space, spaces.Box):
        if use_lattice:
            if use_sde:
                raise NotImplementedError(
                    "Lattice not yet implemented for transformer policy"
                )
                # assert use_sde
                # self.dist_kwargs = {
                #     "squash_output": squash_output,
                #     "use_expln": use_expln,
                #     "learn_features": False,
                #     "std_clip": std_clip,
                #     "std_reg": std_reg,
                # }
                # self.action_dist = LatticeAttentionNoiseDistribution(
                #     get_action_dim(action_space), **self.dist_kwargs
                # )            else:
                return LatticeNoiseDistribution(
                    get_action_dim(action_space), **dist_kwargs
                )
            else:
                raise NotImplementedError(
                    "Lattice not yet implemented for transformer policy"
                )
        else:
            if use_sde:
                raise NotImplementedError(
                    "SDE not yet implemented for transformer policy"
                )
                # self.action_dist = TransformerStateDependentNoiseDistribution(
                #     get_action_dim(action_space), **self.dist_kwargs
                # )
            else:
                if use_bilateral :
                    return BilateralTransformerGaussianDistribution(
                        action_dim=get_action_dim(action_space)
                    )
                else :
                    return TransformerGaussianDistribution(
                        action_dim=get_action_dim(action_space)
                    )
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class TransformerGaussianDistribution(DiagGaussianDistribution):
    """WARNING: it currently does not support masked actions, therefore it only works when the whole batch
    requires an action element for each feature passed to this distribution.

    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0
    ) -> Tuple[nn.Module, nn.Module]:
        action_element = nn.Linear(latent_dim, 1)
        log_std_element = nn.Linear(latent_dim, 1)
        remove_redundant = nn.Flatten(start_dim=1)
        softmax = nn.Softmax(dim=1)
        actions_net = nn.Sequential(action_element, remove_redundant)
        std_scale_net = nn.Sequential(log_std_element, remove_redundant, softmax)
        log_std = nn.Parameter(torch.tensor(log_std_init), requires_grad=True)
        return actions_net, std_scale_net, log_std

    def proba_distribution(
        self, mean_actions, std_actions
    ) -> "TransformerGaussianDistribution":
        self.distribution = Normal(mean_actions, std_actions)
        return self

    # def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    #     # Rescale the log prob based on the action size
    #     return super().log_prob(actions) / self.action_dim

    # def entropy(self) -> torch.Tensor:
    #     # Rescale the entropy based on the action size
    #     return super().entropy() / self.action_dim

class BilateralTransformerGaussianDistribution(DiagGaussianDistribution):
    """WARNING: it currently does not support masked actions, therefore it only works when the whole batch
    requires an action element for each feature passed to this distribution.

    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0
    ) -> Tuple[nn.Module, nn.Module]:
        action_element = nn.Linear(latent_dim, 1)
        log_std_element = nn.Linear(latent_dim, 1)
        softmax = nn.Softmax(dim=1)
        actions_net = nn.Sequential(action_element, nn.Flatten(start_dim=-2))
        std_scale_net = nn.Sequential(log_std_element, softmax, nn.Flatten(start_dim=-2))
        log_std = nn.Parameter(torch.tensor(log_std_init), requires_grad=True)
        return actions_net, std_scale_net, log_std
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return torch.sum(log_prob, dim=-1)

    def proba_distribution(
        self, mean_actions, std_actions
    ) -> "BilateralTransformerGaussianDistribution":
        self.distribution = Normal(mean_actions, std_actions)
        return self

    # def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    #     # Rescale the log prob based on the action size
    #     return super().log_prob(actions) / self.action_dim

    # def entropy(self) -> torch.Tensor:
    #     # Rescale the entropy based on the action size
    #     return super().entropy() / self.action_dim


class PerMuscleDiagGaussianDistribution(DiagGaussianDistribution):
    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        """Differently from the method of the superclass, it expects the latent variables
        to be a tensor of features specific to each action element. Therefore, it just outputs
        one value per feature, corresponding to one action element. It then flattens the output,
        so that the final shape is (batch_size, action_size).

        Args:
            latent_dim (int): dimension of the latent encoding
            log_std_init (float, optional): initial value for the log std. Defaults to 0.0.
        """
        # the same extractor is used for all action elements
        action_element = nn.Linear(latent_dim, 1)
        remove_redundant = nn.Flatten(start_dim=1)
        mean_actions = nn.Sequential(action_element, remove_redundant)
        # log_std = nn.Parameter(
        #     torch.ones(self.action_dim) * log_std_init, requires_grad=True
        # )
        log_std = nn.Parameter(torch.tensor(log_std_init), requires_grad=True)
        return mean_actions, log_std


class TransformerStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    def __init__(
        self,
        action_dim: int,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_reg: float = 0.0,
        **unused_kwargs,
    ):
        print("WARNING: unused kwargs", unused_kwargs)
        super().__init__(
            action_dim=action_dim,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )
        self.std_reg = std_reg

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        std = self.get_std(log_std)
        self.perturb_dist = Normal(
            torch.zeros(self.action_dim).to(std.device),
            std * torch.ones(self.action_dim).to(std.device),
        )
        self.action_perturb = self.perturb_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        self.action_element = nn.Linear(latent_dim, 1)
        self.remove_redundant = nn.Flatten(start_dim=1)
        self.mean_actions_net = nn.Sequential(
            self.action_element,
            self.remove_redundant,
        )
        self.log_std_scale_element = nn.Linear(latent_dim, 1)
        self.log_std_scale_net = nn.Sequential(
            self.log_std_scale_element, self.remove_redundant
        )
        log_std = nn.Parameter(torch.tensor(log_std_init), requires_grad=True)
        self.sample_weights(log_std)
        return self.mean_actions_net, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> StateDependentNoiseDistribution:
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        std = self.get_std(log_std)
        std_scale = (
            self.get_std(self.log_std_scale_net(self._latent_sde)) + self.std_reg
        )
        std = std * std_scale
        self.distribution = Normal(mean_actions, std)
        return self

    def sample(self) -> torch.Tensor:
        std_scale = self.get_std(self.log_std_scale_net(self._latent_sde))
        noise = self.action_perturb * (std_scale + self.std_reg)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions


class LatticeAttentionNoiseDistribution(StateDependentNoiseDistribution):
    """Introduces an attention block and uses the attention matrix to build the covariance
    of the action distribution. Replaces the diagonal gaussian with a full gaussian, whose
    covariance matrix is H^T*diag(sigma^2)*H, where H is the attention matrix and sigma is
    a learnt variance vector

    Args:
        Distribution (_type_): _description_
    """

    def __init__(
        self,
        action_dim: int,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_clip: Tuple[float, float] = (1e-3, 1.0),
        std_reg: float = 0.0,
        **unused_kwargs,
    ):
        print("WARNING: unused kwargs", unused_kwargs)
        super().__init__(
            action_dim=action_dim,
            full_std=False,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )
        self.min_std, self.max_std = std_clip
        self.std_reg = std_reg

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        log_std = log_std.clip(min=np.log(self.min_std), max=np.log(self.max_std))
        log_std[: self.latent_sde_dim] = log_std[: self.latent_sde_dim] - 0.5 * np.log(
            self.latent_sde_dim
        )

        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)
        # Make the std the same size of the V matrix of the self-attention
        std = (
            torch.ones(self.action_dim, self.latent_sde_dim + 1).to(log_std.device)
            * std
        )
        return std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        std = self.get_std(log_std)
        self.perturb_dist = Normal(torch.zeros_like(std), std)
        perturb_matrices = self.perturb_dist.rsample((batch_size,))
        self.value_perturb_matrices = perturb_matrices[:, :, : self.latent_sde_dim]
        self.action_perturb_vec = perturb_matrices[:, :, -1]

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        self.dist_attention = NoisyAttention(dim=latent_dim)
        self.action_element = nn.Linear(latent_dim, 1)
        self.remove_redundant = nn.Flatten(start_dim=1)
        self.mean_actions_net = nn.Sequential(
            ReplicateInputAttentionWrapper(self.dist_attention, 1),
            self.action_element,
            self.remove_redundant,
        )
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = torch.ones(self.latent_sde_dim + 1)  # +1 for the action perturbation
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.sample_weights(log_std)
        return self.mean_actions_net, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "LatticeAttentionNoiseDistribution":
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        std = self.get_std(log_std)
        value_std = std[:, : self.latent_sde_dim]
        action_std = std[:, -1]
        m_cov_diag = torch.mm(self.action_element.weight**2, value_std.T**2)
        dist_latent, attention_matrix = self.dist_attention(self._latent_sde)
        sigma_mat = (attention_matrix * m_cov_diag[:, None, :]).matmul(
            attention_matrix.transpose(1, 2)
        )
        sigma_mat[:, range(self.action_dim), range(self.action_dim)] += (
            action_std + self.std_reg
        ) ** 2

        self.distribution = MultivariateNormal(
            loc=mean_actions, covariance_matrix=sigma_mat, validate_args=True
        )

        # DEBUG: remove after checking that it works
        assert torch.allclose(
            mean_actions, self.remove_redundant(self.action_element(dist_latent))
        )
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        log_prob = self.distribution.log_prob(gaussian_actions)

        if self.bijector is not None:
            # Squash correction
            log_prob -= torch.sum(
                self.bijector.log_prob_correction(gaussian_actions), dim=1
            )
        return log_prob

    def entropy(self) -> torch.Tensor:
        if self.bijector is not None:
            return None
        return self.distribution.entropy()

    def get_noise(
        self,
        latent_sde: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def sample(self) -> torch.Tensor:
        actions = self.dist_attention(self._latent_sde, self.value_perturb_matrices)[0]
        actions = self.action_element(actions)
        actions = self.remove_redundant(actions)
        actions += self.action_perturb_vec
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions


class PerMuscleSquashedDiagGaussianDistribution(SquashedDiagGaussianDistribution):
    def proba_distribution_net(self, latent_dim, log_std_init=0):
        action_element = nn.Linear(latent_dim, 1)
        remove_redundant = nn.Flatten(start_dim=1)
        mean_actions = nn.Sequential(action_element, remove_redundant)
        std_element = nn.Linear(latent_dim, 1)
        log_std = nn.Sequential(std_element, remove_redundant)
        return mean_actions, log_std


class LatticeNoiseDistribution(DiagGaussianDistribution):
    """Like Lattice noise distribution, but not state-dependent. Does not allow time correlation, but
    it is more efficient.
    """

    def __init__(self, action_dim: int, std_reg: float = 0.0):
        super().__init__(action_dim=action_dim)
        self.std_reg = std_reg

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        self.mean_actions = nn.Linear(latent_dim, self.action_dim)
        self.std_init = torch.tensor(log_std_init).exp()
        log_std = nn.Parameter(
            torch.zeros(self.action_dim + latent_dim), requires_grad=True
        )
        return self.mean_actions, log_std

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        std = log_std.exp() * self.std_init
        action_variance = std[: self.action_dim] ** 2
        latent_variance = std[self.action_dim :] ** 2
        sigma_mat = (self.mean_actions.weight * latent_variance).matmul(
            self.mean_actions.weight.T
        )
        sigma_mat = (sigma_mat + sigma_mat.mT) / 2
        sigma_mat[
            range(self.action_dim), range(self.action_dim)
        ] += action_variance  # + self.std_reg ** 2
        self.distribution = MultivariateNormal(mean_actions, sigma_mat)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class StateDependentLatticeNoiseDistribution(StateDependentNoiseDistribution):
    def __init__(
        self,
        action_dim: int,
        full_std: bool = False,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_clip: Tuple[float, float] = (1e-3, 1.0),
        std_reg: float = 0.0,
        alpha: float = 1,
        **unused_kwargs,
    ):
        print("WARNING: unused kwargs", unused_kwargs)
        super().__init__(
            action_dim=action_dim,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )
        self.min_std, self.max_std = std_clip
        self.std_reg = std_reg
        self.alpha = alpha

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        # Apply correction to remove scaling of action std as a function of the latent dimension (see paper for details)
        log_std = log_std.clip(min=np.log(self.min_std), max=np.log(self.max_std))
        log_std = log_std - 0.5 * np.log(self.latent_sde_dim)

        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        if self.full_std:
            assert std.shape == (
                self.latent_sde_dim,
                self.latent_sde_dim + self.action_dim,
            )
            corr_std = std[:, : self.latent_sde_dim]
            ind_std = std[:, -self.action_dim :]
        else:
            # Reduce the number of parameters:
            assert std.shape == (self.latent_sde_dim, 2), std.shape
            corr_std = (
                torch.ones(self.latent_sde_dim, self.latent_sde_dim).to(log_std.device)
                * std[:, 0:1]
            )
            ind_std = (
                torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device)
                * std[:, 1:]
            )
        return corr_std, ind_std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        corr_std, ind_std = self.get_std(log_std)
        self.corr_weights_dist = Normal(torch.zeros_like(corr_std), corr_std)
        self.ind_weights_dist = Normal(torch.zeros_like(ind_std), ind_std)

        # Reparametrization trick to pass gradients
        self.corr_exploration_mat = self.corr_weights_dist.rsample()
        self.ind_exploration_mat = self.ind_weights_dist.rsample()

        # Pre-compute matrices in case of parallel exploration
        self.corr_exploration_matrices = self.corr_weights_dist.rsample((batch_size,))
        self.ind_exploration_matrices = self.ind_weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        # Note: we always consider that the noise is based on the features of the last layer, so latent_sde_dim is the same as latent_dim
        self.mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = (
            torch.ones(self.latent_sde_dim, self.latent_sde_dim + self.action_dim)
            if self.full_std
            else torch.ones(self.latent_sde_dim, 2)
        )

        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return self.mean_actions_net, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "LatticeNoiseDistribution":
        # Detach the last layer features because we do not want to update the noise generation
        # to influence the features of the policy
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # TODO: check that it might need to be transposed
        corr_std, ind_std = self.get_std(log_std)
        latent_corr_variance = torch.mm(
            self._latent_sde**2, corr_std**2
        )  # Variance of the hidden state
        # TODO: Is it necessary to regularize?
        latent_ind_variance = (
            torch.mm(self._latent_sde**2, ind_std**2) + self.std_reg**2
        )  # Variance of the action

        # First consider the correlated variance
        sigma_mat = self.alpha**2 * (
            self.mean_actions_net.weight * latent_corr_variance[:, None, :]
        ).matmul(self.mean_actions_net.weight.T)
        # Then the independent one, to be added to the diagonal
        sigma_mat[
            :, range(self.action_dim), range(self.action_dim)
        ] += latent_ind_variance
        self.distribution = MultivariateNormal(
            loc=mean_actions, covariance_matrix=sigma_mat, validate_args=False
        )
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        log_prob = self.distribution.log_prob(gaussian_actions)

        if self.bijector is not None:
            # Squash correction
            log_prob -= torch.sum(
                self.bijector.log_prob_correction(gaussian_actions), dim=1
            )
        return log_prob

    def entropy(self) -> torch.Tensor:
        if self.bijector is not None:
            return None
        return self.distribution.entropy()

    def get_noise(
        self,
        latent_sde: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
    ) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(exploration_matrices):
            return torch.mm(latent_sde, exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, exploration_matrices)
        return noise.squeeze(dim=1)

    def sample(self) -> torch.Tensor:
        latent_noise = self.alpha * self.get_noise(
            self._latent_sde, self.corr_exploration_mat, self.corr_exploration_matrices
        )
        action_noise = self.get_noise(
            self._latent_sde, self.ind_exploration_mat, self.ind_exploration_matrices
        )
        actions = self.mean_actions_net(self._latent_sde + latent_noise) + action_noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions
