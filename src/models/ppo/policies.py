import gym
import torch
import warnings
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.utils import get_device
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import (
    Distribution,
)
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    BasePolicy,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from models.distributions import (
    LatticeNoiseDistribution,
    StateDependentLatticeNoiseDistribution,
    TransformerGaussianDistribution,
    BilateralTransformerGaussianDistribution
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN
)
from models.extractors import (
    TransformerExtractor,
    PredictiveTransformerExtractor,
    BilateralTransformerExtractor
)
from models.feature_extractors import TransformerFeaturesExtractor, PredictiveTransformerFeaturesExtractor
from models.helpers import Mean
from models.distributions import make_proba_distribution
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from torch_utils import select_by_mask

# These 2 functions cannot be put into utilities.py otherwise there will be circular imports
def unsqueeze_dict_tensor(dct: dict, dim: int) -> dict :

    for key, value in dct.items() :
        if isinstance(value, torch.Tensor) :
            dct[key] = value.unsqueeze(dim)
        else :
            raise NotImplementedError
    return dct

def squeeze_dict_tensor(dct: dict, dim: int) -> dict :

    for key, value in dct.items() :
        if isinstance(value, torch.Tensor) :
            dct[key] = value.squeeze(dim)
        else :
            raise NotImplementedError
    return dct

def move_dict_to_device(dct: dict, device: torch.device) -> dict :
    
        for key, value in dct.items() :
            if isinstance(value, torch.Tensor) :
                dct[key] = value.to(device)
            else :
                raise NotImplementedError
        return dct

def to_tensor_dict(array, **kwargs) -> torch.Tensor:
    """Converts a dict of NumPy array to a dict of PyTorch tensor.

    The data is copied in the case where the array is non-writable. Unfortunately if
    you just use `torch.as_tensor` for this, an ugly warning is logged and there's
    undefined behavior if you try to write to the tensor.

    Args:
        array: The array to convert to a PyTorch tensor.
        kwargs: Additional keyword arguments to pass to `torch.as_tensor`.

    Returns:
        A PyTorch tensor with the same content as `array`.
    """

    if isinstance(array, dict) :
        ret = {}
        for key, val in array.items() :
            ret[key] = to_tensor_dict(val, **kwargs)
        return ret

    elif isinstance(array, torch.Tensor):
        if "device" in kwargs:
            return array.to(kwargs["device"])
        else:
            return array

    elif isinstance(array, np.ndarray) :

        if not array.flags.writeable:
            array = array.copy()

        return torch.as_tensor(array, **kwargs).contiguous()

    else :
        return array

class MuscleTransformerPolicy(ActorCriticPolicy):

    feature_extractor_class = TransformerFeaturesExtractor
    extractor_class = TransformerExtractor

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
        use_sde=False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_lattice: bool = False,
        lattice_kwargs: Optional[Dict[str, Any]] = None,
        policy_outputs_variance: bool = False,
        critic_only_training: bool = False,
        device: Union[torch.device, str] = "cuda",
        **unused_kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        features_extractor_kwargs["activation_fn"] = activation_fn

        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            self.feature_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        if net_arch is None:
            net_arch = {}

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.share_features_extractor = share_features_extractor
        self.device_name = device
        self.features_extractor = self.feature_extractor_class(
            self.observation_space, device=self.device, **self.features_extractor_kwargs
        )
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.feature_extractor_class(
                self.observation_space,
                device=self.device,
                **self.features_extractor_kwargs,
            )
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = True
        self.log_std_init = log_std_init
        dist_kwargs = {}

        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }
        if use_lattice:
            lattice_kwargs = {} if lattice_kwargs is None else lattice_kwargs
            dist_kwargs.update(lattice_kwargs)

        self.dist_kwargs = dist_kwargs

        self.action_dist = make_proba_distribution(
            action_space,
            use_sde=use_sde,
            dist_kwargs=dist_kwargs,
            use_lattice=use_lattice
        )

        self.use_sde = use_sde
        self.use_lattice = use_lattice
        self.policy_outputs_variance = policy_outputs_variance
        self.critic_only_training = critic_only_training

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = self.extractor_class(
            embedding_size=self.features_dim,
            activation_fn=self.activation_fn,
            device=self.device,
            **self.net_arch,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, TransformerGaussianDistribution) :
            (
                self.action_net,
                self.log_std_net,
                self.log_std,
            ) = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Sequential(
            Mean(dim=1), nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            raise NotImplementedError(
                "Ortho init not implemented for MuscleTransformerPolicy"
            )

        # Setup optimizer with initial learning rate
        if self.critic_only_training:
            assert not self.mlp_extractor.share_decoder
            trainable_params = self.mlp_extractor._value_decoder.parameters()
        else:
            trainable_params = self.parameters()
        self.optimizer = self.optimizer_class(
            trainable_params, lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        if isinstance(self.action_dist, TransformerGaussianDistribution):
            mean_actions = self.action_net(latent_pi)
            if self.policy_outputs_variance:
                log_std_actions = self.log_std_net(
                    latent_pi.detach()
                )  # Do not backpropagate the std branch of the network
                std_actions = log_std_actions.exp()
                std_actions = std_actions / std_actions.mean() * self.log_std.exp()
            else:
                std_actions = self.log_std.exp()
            return self.action_dist.proba_distribution(mean_actions, std_actions)
        else:
            return super()._get_action_dist_from_latent(latent_pi)

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            if features_extractor is not None:
                return features_extractor(obs)
            else:
                return self.features_extractor(obs)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.pi_features_extractor(obs)
            vf_features = self.vf_features_extractor(obs)
            return pi_features, vf_features

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        predict_value: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        self.set_training_mode(False)

        with torch.no_grad() :

            observation = to_tensor_dict(observation, device=self.device)

            features = self.extract_features(observation)
            latent_pi, latent_vf = self.mlp_extractor(features)
            # if predict_value :
            #     values = self.value_net(latent_vf)
            # Evaluate the values for the given observations
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)

        # if predict_value :
        #     return actions.cpu().numpy(), values.cpu().numpy(), state
        # else :
        return actions.cpu().numpy(), state


    @property
    def device(self) -> torch.device:
        return get_device(self.device_name)

    def set_vocabulary(self, vocabulary):
        self.pi_features_extractor.set_vocabulary(vocabulary)
        self.vf_features_extractor.set_vocabulary(vocabulary)

    def to(self, device: Union[str, torch.device]) -> "MuscleTransformerPolicy":
        self.device_name = device
        return super().to(device)

class PredictiveMuscleTransformerPolicy(MuscleTransformerPolicy):

    feature_extractor_class = PredictiveTransformerFeaturesExtractor
    extractor_class = PredictiveTransformerExtractor

    def _build(self, lr_schedule: Schedule) -> None:

        super()._build(lr_schedule)
        self.projector = nn.Linear(self.mlp_extractor.latent_dim_pi, self.observation_space["obs"].shape[-1])

    def predict_next_obs(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action: np.ndarray
    ) :
        features = self.extract_features(observation)
        action_features = self.pi_features_extractor.encode_action(observation, action)
        next_obs_features = self.mlp_extractor.forward_prediction(features, action_features)
        return self.projector(next_obs_features)

    def evaluate_actions_and_predict_next_obs(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action: np.ndarray
    ) :

        features = self.extract_features(observation)
        
        # evaluate actions
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(action)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        # predict next obs
        action_features = self.pi_features_extractor.encode_action(observation, action)
        next_obs_features = self.mlp_extractor.forward_prediction(features, action_features)

        return values, log_prob, entropy, self.projector(next_obs_features)

class BilateralMuscleTransformerPolicy(MuscleTransformerPolicy):

    extractor_class = BilateralTransformerExtractor

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde=False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_lattice: bool = False,
        lattice_kwargs: Optional[Dict[str, Any]] = None,
        policy_outputs_variance: bool = False,
        device: Union[torch.device, str] = "cuda",
        **unused_kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        features_extractor_kwargs["activation_fn"] = activation_fn

        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            self.feature_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        if net_arch is None:
            net_arch = {}

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.share_features_extractor = share_features_extractor
        self.device_name = device
        self.features_extractor = self.feature_extractor_class(
            self.observation_space, device=self.device, **self.features_extractor_kwargs
        )
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.feature_extractor_class(
                self.observation_space,
                device=self.device,
                **self.features_extractor_kwargs,
            )
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = True
        self.log_std_init = log_std_init
        dist_kwargs = {}

        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }
        if use_lattice:
            lattice_kwargs = {} if lattice_kwargs is None else lattice_kwargs
            dist_kwargs.update(lattice_kwargs)

        self.dist_kwargs = dist_kwargs

        self.action_dist = make_proba_distribution(
            action_space,
            use_sde=use_sde,
            dist_kwargs=dist_kwargs,
            use_lattice=use_lattice,
            use_bilateral=True
        )

        self.use_sde = use_sde
        self.use_lattice = use_lattice
        self.policy_outputs_variance = policy_outputs_variance

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.k_bilateral = self.net_arch["k_bilateral"]
        self.mlp_extractor = self.extractor_class(
            embedding_size=self.features_dim,
            activation_fn=self.activation_fn,
            device=self.device,
            share_head=True,
            **self.net_arch,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        assert(isinstance(self.action_dist, BilateralTransformerGaussianDistribution))
        (
            self.action_net,
            self.log_std_net,
            self.log_std,
        ) = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )

        self.value_net = nn.Sequential(
            Mean(dim=1), nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            raise NotImplementedError(
                "Ortho init not implemented for MuscleTransformerPolicy"
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:

        if isinstance(self.action_dist, BilateralTransformerGaussianDistribution):
            mean_actions = self.action_net(latent_pi)
            if self.policy_outputs_variance:
                log_std_actions = self.log_std_net(
                    latent_pi.detach()
                )
                std_actions = log_std_actions.exp()
                std_actions = std_actions / std_actions.mean() * self.log_std.exp()
            else:
                std_actions = self.log_std.exp()
            return self.action_dist.proba_distribution(mean_actions, std_actions)

    def _get_src_mask(self, dones) :

        assert(len(dones.shape) == 2)
        batch_size = dones.shape[0]
        seq_len = dones.shape[1]
        causal_mask = torch.full((batch_size * seq_len, batch_size * seq_len), True, dtype=bool, device=self.device)

        # Determine sentence boundaries based on done tensor
        sentence_start = 0
        # causal_mask = []
        for b in range(batch_size) :
            if dones[b].any() :
                full_dones = torch.cat([dones[b], torch.ones((1,), dtype=dones.dtype, device=self.device)])
                full_dones[0] = True
                non_zero = torch.nonzero(full_dones, as_tuple=True)[0]
                causal_mask_size = torch.diff(non_zero)
                assert(sum(causal_mask_size) == seq_len)
            else :
                causal_mask_size = [seq_len]

            for sz in causal_mask_size :
                # causal_mask.append(
                local_causal_mask = torch.triu(
                        torch.full((sz, sz), True, dtype=bool, device=self.device),
                        diagonal=1,
                    )
                causal_mask[sentence_start:sentence_start+sz, sentence_start:sentence_start+sz] = local_causal_mask
                sentence_start += sz
                # )
        # causal_mask = torch.block_diag(*causal_mask)
            
        assert(causal_mask.shape == (batch_size * seq_len, batch_size * seq_len))

        # return causal_mask


        # for b in range(batch_size) :
        #     non_zero = torch.nonzero(dones[b], as_tuple=False)[0]
        #     non_zero = torch.cat([non_zero, torch.Tensor([seq_len], dtype=torch.float32, device=self.device)])
        #     causal_mask_size = torch.diff(non_zero)
        # for b in range(batch_size) :
        #     for i in range(seq_len):
        #         if dones[b, i] == 1 or i == seq_len-1:  # End of a sentence
        #             causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=i-sentence_start+1, device=self.device)
        #             mask[b, sentence_start:i+1, sentence_start:i+1] = causal_mask
        #             sentence_start = i + 1
        
        k_bilateral_mask = torch.tril(  # Only keep k-nearest neighbors
            torch.full(
                (batch_size * seq_len, batch_size * seq_len),
                False,
                dtype=bool,
                device=causal_mask.device
            ),
            diagonal=1,
        )

        mask = torch.logical_or(causal_mask, k_bilateral_mask)

        return mask
    
    def _do_maskings(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        obs_timestep: torch.Tensor,
        act_timestep: torch.Tensor,
        src_mask: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None
    ) :

        ret_obs_dict = {}

        if obs_mask is not None :
            assert(obs_mask.dtype == torch.bool)
            ret_obs_dict["obs"] = select_by_mask(obs["obs"], obs_mask)
            ret_obs_dict["obs_ids"] = select_by_mask(obs["obs_ids"], obs_mask)
            ret_src_mask = src_mask[obs_mask.view(-1), :][:, obs_mask.view(-1)]
            ret_obs_timestep = select_by_mask(obs_timestep, obs_mask)
            
        else :
            ret_obs_dict["obs"] = obs["obs"]
            ret_obs_dict["obs_ids"] = obs["obs_ids"]
            ret_src_mask = src_mask
            ret_obs_timestep = obs_timestep
        
        if act_mask is not None :
            ret_obs_dict["action_ids"] = select_by_mask(obs["action_ids"], act_mask)
        else :
            ret_obs_dict["action_ids"] = obs["action_ids"]
        
        if act_mask is not None and actions is not None :
            assert(act_mask.dtype == torch.bool)
            ret_actions = select_by_mask(actions, act_mask)
        else :
            ret_actions = actions
        
        if act_mask is not None and act_timestep is not None :
            ret_act_timestep = select_by_mask(act_timestep, act_mask)
        else :
            ret_act_timestep = act_timestep

        return ret_obs_dict, ret_src_mask, ret_obs_timestep, ret_actions, ret_act_timestep
        

    def evaluate_actions_bilateral(
        self,
        obs: PyTorchObs,
        dones: torch.Tensor,
        actions: torch.Tensor,
        obs_mask: torch.Tensor = None,
        act_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation, Time x Token x obs_dim
        :param actions: Actions, T x action_dim
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed

        assert(len(dones.shape) == 1)
        assert(len(actions.shape) == 2)
        assert(isinstance(actions, torch.Tensor))
        assert(isinstance(dones, torch.Tensor))

        obs = unsqueeze_dict_tensor(obs, 0)
        actions.unsqueeze_(0)
        dones.unsqueeze_(0)
        obs_mask.unsqueeze_(0)
        act_mask.unsqueeze_(0)

        src_mask = self._get_src_mask(dones)
        obs_timestep: torch.Tensor = obs["timestep"]
        act_timestep: torch.Tensor = obs["timestep"]

        (
            ret_obs_dict,
            ret_src_mask,
            ret_obs_timestep,
            ret_actions,
            ret_act_timestep
        ) = self._do_maskings(
            obs,
            actions,
            obs_timestep,
            act_timestep,
            src_mask,
            obs_mask,
            act_mask
        )

        features = self.extract_features(ret_obs_dict)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(
                features,
                ret_src_mask,
                ret_obs_timestep,
                ret_act_timestep
            )
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, src_mask)
            latent_vf = self.mlp_extractor.forward_critic(vf_features, src_mask)

        latent_pi.squeeze_(0)
        latent_vf.squeeze_(0)

        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(ret_actions)
        log_prob.squeeze_(0)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        obs = squeeze_dict_tensor(obs, 0)
        actions.squeeze_(0)
        dones.squeeze_(0)

        return values, log_prob, entropy

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        predict_value: bool = False,
        done: Optional[np.ndarray] = None,
        obs_mask: torch.Tensor = None,
        act_mask: torch.Tensor = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        assert(len(done.shape) == 2)

        self.set_training_mode(False)

        with torch.no_grad() :
            
            done = torch.from_numpy(done).to(self.device).contiguous()
            obs_mask = None if obs_mask is None else torch.tensor(obs_mask).to(self.device).contiguous()
            act_mask = None if act_mask is None else torch.tensor(act_mask).to(self.device).contiguous()
            src_mask = self._get_src_mask(done)
            observation = to_tensor_dict(observation, device=self.device)
            obs_timestep: torch.Tensor = observation["timestep"]
            act_timestep: torch.Tensor = observation["timestep"]

            (
                ret_obs_dict,
                ret_src_mask,
                ret_obs_timestep,
                ret_actions,
                ret_act_timestep
            ) = self._do_maskings(
                observation,
                None,
                obs_timestep,
                act_timestep,
                src_mask,
                obs_mask,
                act_mask
            )

            fatures = self.extract_features(ret_obs_dict)
            latent_pi, latent_vf = self.mlp_extractor(
                features = fatures,
                src_mask = ret_src_mask,
                obs_timestep = ret_obs_timestep,
                act_timestep = ret_act_timestep
            )
            # if predict_value :
            #     values = self.value_net(latent_vf)
            # Evaluate the values for the given observations
            distribution = self._get_action_dist_from_latent(latent_pi[:, -1, :, :])
            actions = distribution.get_actions(deterministic=deterministic)

            return actions.cpu().numpy(), state
    
    def get_distribution_bilateral(
        self,
        obs: PyTorchObs,
        dones,
        obs_mask: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None
    ) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        assert(len(dones.shape) == 1)
        assert(isinstance(dones, torch.Tensor))

        obs = unsqueeze_dict_tensor(obs, 0)
        dones.unsqueeze_(0)
        obs_mask.unsqueeze_(0)
        act_mask.unsqueeze_(0)

        src_mask = self._get_src_mask(dones)
        obs_timestep: torch.Tensor = obs["timestep"]
        act_timestep: torch.Tensor = obs["timestep"]

        (
            ret_obs_dict,
            ret_src_mask,
            ret_obs_timestep,
            _,
            ret_act_timestep
        ) = self._do_maskings(
            obs,
            None,
            obs_timestep,
            act_timestep,
            src_mask,
            obs_mask,
            act_mask
        )
        
        features = super().extract_features(ret_obs_dict, self.pi_features_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(
                features,
                ret_src_mask,
                ret_obs_timestep,
                ret_act_timestep
            )
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, ret_src_mask)

        latent_pi.squeeze_(0)

        return self._get_action_dist_from_latent(latent_pi)

class MuscleMlpPolicy(ActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if isinstance(observation_space, spaces.Dict) :
            observation_space = observation_space["obs"]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
    
    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if isinstance(obs, dict) :
            obs = obs["obs"]
        
        return super().extract_features(obs, features_extractor)
    
    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[PyTorchObs, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        if isinstance(observation, dict) :
            observation = observation["obs"]
        return super().obs_to_tensor(observation)
    
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        if isinstance(obs, dict) :
            obs = obs["obs"]
        return super().predict_values(obs)

    def set_vocabulary(self, vocabulary):
        pass

class LatticeRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice=True,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if use_lattice:
            if self.use_sde:
                self.dist_kwargs.update(
                    {
                        "epsilon": expln_eps,
                        "std_clip": std_clip,
                        "std_reg": std_reg,
                        "alpha": alpha,
                    }
                )
                self.action_dist = StateDependentLatticeNoiseDistribution(
                    get_action_dim(self.action_space), **self.dist_kwargs
                )
            else:
                self.action_dist = LatticeNoiseDistribution(
                    get_action_dim(self.action_space), std_reg
                )
            self._build(lr_schedule)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        return super().predict(
            observation,
            state,
            episode_start,
            deterministic
        )
