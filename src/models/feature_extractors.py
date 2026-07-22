from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from models.helpers import SinCosPositionalEncoding, LearnedPositionalEncoding
from definitions import (
    MASK_SUFFIX,
    OBS_KEY,
    OBS_ID_KEY,
    ACTION_ID_KEY,
    VALUE_KEY,
    PADDING_KEY,
)
from vocabulary import VOCABULARY
from models.helpers import Identity

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Implements the observation masking, the positional encoding and the early layers of the
    transformer decoder. The observation must be a dictionary with the following keys:
    - <obs_name>: can accept an arbitrary number of observations (e.g., one observation can be
    the state of a joint). Observations under the same <obs_name> share the same linear encoder,
    so they should be homogeneous in the number of features. Each observation is represented by
    a tensor of shape (num_timesteps, num_features, num_obs). num_features is, e.g., 3 for a joint
    (angular position, angular velocity, torque), while num_obs can be arbitrary (e.g., number
    of joints).
    - <obs_name>_encoding: each observation needs to have an associated positional encoding. It
    is passed in the form of an array of indices for each observation element. The positional
    encodings corresponding to the indices in the array will be summed to the observation embedding.
    Negative indices are ignored.
    - <obs_name>_mask: one mask per obs_name, states which of the observations in <obs_name>
    are valid observations. It is used to keep the same observation space for environments with
    a variable number of observations
    - actuators_encoding: index of the positional encoding of the acutators
    - actuators_mask: int indicating how many actuators are valid
    """

    def __init__(
        self,
        observation_space,
        embedding_size=16,
        num_heads=4,
        num_layers=4,
        layer_norm_eps=1e-5,
        dim_feedforward=32,
        dropout=0.1,
        position_embedding="learned",  # "sin_cos" or "learned"
        norm_first=False,
        vocabulary=None,
        activation_fn=torch.nn.ReLU,
        device=None,
    ):
        super().__init__(observation_space=observation_space, features_dim=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._features_dim = embedding_size
        self.dropout = dropout
        self.position_embedding = position_embedding
        self.vocabulary = vocabulary if vocabulary is not None else VOCABULARY
        random_obs = observation_space.sample()
        _, num_timesteps = random_obs["obs"].shape

        # Define the linear embeddings for the observation
        self.tokenizer = nn.Linear(
            in_features=num_timesteps,
            out_features=embedding_size,
        ).to(device)
        self.num_tokens = len(self.vocabulary)
        self.positional_encoder = self.make_positional_encoder()

        if num_layers == 0:
            self.encoder = Identity().to(device)
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                batch_first=True,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                activation=activation_fn(),
            )
            layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
            self.encoder = nn.TransformerEncoder(
                transformer_layer, num_layers, layer_norm
            ).to(device)

        self._reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, observation_dict):
        obs = observation_dict[OBS_KEY]
        obs_token = self.tokenizer(obs)

        obs_ids = observation_dict[OBS_ID_KEY]
        if len(obs_ids.shape) != len(obs.shape) :
            obs_ids = obs_ids.unsqueeze(0)
        positional_encoded_obs = self.positional_encoder(obs_token, obs_ids)

        # Run the transformer encoder (works also without a mask)
        obs_mask = observation_dict.get(OBS_KEY + MASK_SUFFIX)
        encodings = self.encoder(
            positional_encoded_obs, src_key_padding_mask=obs_mask
        )  # (batch, num_obs, embedding_size)

        # Build the target for the decoder
        action_ids = observation_dict[ACTION_ID_KEY]
        if len(action_ids.shape) != len(obs.shape) :
            action_ids = action_ids.unsqueeze(0)
        action_target = torch.zeros(*action_ids.shape[:-1], self._features_dim).to(
            self.device
        )
        positional_encoded_action_target = self.positional_encoder(
            action_target, action_ids
        )
        action_mask = observation_dict.get(ACTION_ID_KEY + MASK_SUFFIX)

        value_target = torch.zeros(*action_ids.shape[:-2], 1, self._features_dim).to(
            self.device
        )
        value_idx = torch.tensor(self.vocabulary[VALUE_KEY]).to(self.device)
        value_target = self.positional_encoder(value_target, value_idx)

        return (
            encodings,
            obs_mask,
            positional_encoded_action_target,
            action_mask,
            value_target,
        )

    def make_positional_encoder(self):
        if self.position_embedding == "sin_cos":
            return SinCosPositionalEncoding(
                num_tokens=self.num_tokens,
                d_model=self._features_dim,
                dropout=self.dropout,
                padding_idx=self.vocabulary.get(PADDING_KEY),
            ).to(self.device)
        elif self.position_embedding == "learned":
            return LearnedPositionalEncoding(
                num_tokens=self.num_tokens,
                d_model=self._features_dim,
                dropout=self.dropout,
                padding_idx=self.vocabulary.get(PADDING_KEY),
            ).to(self.device)
        else:
            raise ValueError(
                "Unknown position embedding type:", self.position_embedding
            )

    def set_vocabulary(self, new_vocabulary):
        if new_vocabulary != self.vocabulary:
            assert all([key in new_vocabulary for key in self.vocabulary])
            self.num_tokens = len(new_vocabulary)
            old_embedding = self.positional_encoder.embedding
            self.positional_encoder = self.make_positional_encoder()
            new_idx_map = torch.zeros(self.num_tokens, dtype=torch.int32)
            for key, idx in self.vocabulary.items():
                new_idx = new_vocabulary[key]
                new_idx_map[new_idx] = idx
            new_idx_map = new_idx_map.to(self.device)
            new_embedding = old_embedding(new_idx_map)
            if isinstance(self.positional_encoder, SinCosPositionalEncoding):
                self.positional_encoder.embedding = new_embedding
            elif isinstance(self.positional_encoder, LearnedPositionalEncoding):
                self.positional_encoder.set_pretrained(
                    new_embedding, padding_idx=new_vocabulary[PADDING_KEY]
                )
            self.vocabulary = new_vocabulary

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        padding_idx = self.vocabulary.get(PADDING_KEY)
        if isinstance(self.positional_encoder, LearnedPositionalEncoding):
            self.positional_encoder.embedding.weight.data[padding_idx].fill_(0)


class PredictiveTransformerFeaturesExtractor(TransformerFeaturesExtractor):
    """Implements the observation masking, the positional encoding and the early layers of the
    transformer decoder. The observation must be a dictionary with the following keys:
    - <obs_name>: can accept an arbitrary number of observations (e.g., one observation can be
    the state of a joint). Observations under the same <obs_name> share the same linear encoder,
    so they should be homogeneous in the number of features. Each observation is represented by
    a tensor of shape (num_timesteps, num_features, num_obs). num_features is, e.g., 3 for a joint
    (angular position, angular velocity, torque), while num_obs can be arbitrary (e.g., number
    of joints).
    - <obs_name>_encoding: each observation needs to have an associated positional encoding. It
    is passed in the form of an array of indices for each observation element. The positional
    encodings corresponding to the indices in the array will be summed to the observation embedding.
    Negative indices are ignored.
    - <obs_name>_mask: one mask per obs_name, states which of the observations in <obs_name>
    are valid observations. It is used to keep the same observation space for environments with
    a variable number of observations
    - actuators_encoding: index of the positional encoding of the acutators
    - actuators_mask: int indicating how many actuators are valid
    """

    def __init__(
        self,
        observation_space,
        embedding_size=16,
        num_heads=4,
        num_layers=4,
        layer_norm_eps=1e-5,
        dim_feedforward=32,
        dropout=0.1,
        position_embedding="learned",  # "sin_cos" or "learned"
        norm_first=False,
        vocabulary=None,
        activation_fn=torch.nn.ReLU,
        device=None,
    ):
        super().__init__(
            observation_space=observation_space,
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_norm_eps=layer_norm_eps,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            position_embedding=position_embedding,
            norm_first=norm_first,
            vocabulary=vocabulary,
            activation_fn=activation_fn,
            device=device
        )

        self.action_tokenizer = nn.Linear(
            in_features=1,
            out_features=embedding_size,
        ).to(device)

    def encode_action(self, observation_dict, action):

        obs = observation_dict[OBS_KEY]
        action_ids = observation_dict[ACTION_ID_KEY]
        if len(action_ids.shape) != len(obs.shape) :
            action_ids = action_ids.unsqueeze(0)
        action_tokens = self.action_tokenizer(action.unsqueeze(-1))
        positional_encoded_action_target = self.positional_encoder(
            action_tokens, action_ids
        )
        action_mask = observation_dict.get(ACTION_ID_KEY + MASK_SUFFIX)
        encodings = self.encoder(
            positional_encoded_action_target, src_key_padding_mask=action_mask
        )  # (batch, num_obs, embedding_size)
        return encodings, action_mask


class MultiTaskFlatFeaturesExtractor(BaseFeaturesExtractor):
    """Features extractor for multi-task MLP policies.

    Flattens obs[OBS_KEY] to a 1-D feature vector. The multi-env wrappers
    already zero-pad each environment's observation to the maximum token/
    timestep count across all tasks, so no additional padding is needed here.
    """

    def __init__(self, observation_space, **kwargs):
        if isinstance(observation_space, spaces.Dict):
            obs_space = observation_space[OBS_KEY]
        else:
            obs_space = observation_space
        flat_size = int(np.prod(obs_space.shape))
        super().__init__(observation_space, features_dim=flat_size)

    def forward(self, obs):
        x = obs[OBS_KEY] if isinstance(obs, dict) else obs
        flat = x.reshape(x.shape[0], -1)
        if flat.shape[1] < self._features_dim:
            padding = torch.zeros(
                flat.shape[0], self._features_dim - flat.shape[1],
                device=flat.device, dtype=flat.dtype,
            )
            flat = torch.cat([flat, padding], dim=1)
        return flat
