import torch
from torch import nn
import os
import math
from definitions import PADDING_KEY


class TransformerExtractor(nn.Module):
    def __init__(
        self,
        embedding_size=16,
        num_heads=1,
        num_encoder_layers=0,
        num_decoder_layers=1,
        layer_norm_eps=1e-5,
        dim_feedforward=32,
        dropout=0,
        norm_first=False,
        activation_fn=torch.nn.ReLU,
        share_decoder=True,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        assert (
            num_decoder_layers > 0
        ), "There must be at least one decoder layer to produce actions and values"

        self.latent_dim_pi = embedding_size
        self.latent_dim_vf = embedding_size

        if num_encoder_layers == 0:
            self.encoder = nn.Identity()
        else:
            encoder_layer = nn.TransformerEncoderLayer(
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
                encoder_layer, num_encoder_layers, layer_norm
            ).to(self.device)

        decoder_layer = nn.TransformerDecoderLayer(
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

        # Store share_decoder setting
        self.share_decoder = share_decoder

        # Initialize as None first
        self._action_decoder = None
        self._value_decoder = None
        self.decoder = None  # Keep original name for backward compatibility

        if share_decoder:
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_decoder_layers, layer_norm
            ).to(self.device)
        else:
            self._action_decoder = nn.TransformerDecoder(
                decoder_layer, num_decoder_layers, layer_norm
            ).to(self.device)
            self._value_decoder = nn.TransformerDecoder(
                decoder_layer, num_decoder_layers, layer_norm
            ).to(self.device)

        self._reset_parameters()

    @property
    def action_decoder(self):
        if self.share_decoder:
            return self.decoder
        return self._action_decoder

    @property
    def value_decoder(self):
        if self.share_decoder:
            return self.decoder
        return self._value_decoder

    def forward(self, features):
        encodings, encodings_mask, action_target, action_mask, value_target = features
        encodings = self.encoder(encodings, src_key_padding_mask=encodings_mask)
        action = self.action_decoder(
            action_target,
            encodings,
            tgt_key_padding_mask=action_mask,
            memory_key_padding_mask=encodings_mask,
        )
        previous = os.environ["ALLOW_WEIGHT_RECORDING"]
        os.environ["ALLOW_WEIGHT_RECORDING"] = "False"
        # print("setted to False")
        value = self.value_decoder(
            value_target, encodings, memory_key_padding_mask=encodings_mask
        )
        os.environ["ALLOW_WEIGHT_RECORDING"] = previous
        # print("setted to True")
        return action, value

    def forward_actor(self, features):
        encodings, encodings_mask, action_target, action_mask, _ = features
        encodings = self.encoder(encodings, src_key_padding_mask=encodings_mask)
        action = self.action_decoder(
            action_target,
            encodings,
            tgt_key_padding_mask=action_mask,
            memory_key_padding_mask=encodings_mask,
        )
        return action

    def forward_critic(self, features):
        encodings, encodings_mask, _, _, value_target = features
        encodings = self.encoder(encodings, src_key_padding_mask=encodings_mask)
        value = self.value_decoder(
            value_target, encodings, memory_key_padding_mask=encodings_mask
        )
        return value

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class PredictiveTransformerExtractor(TransformerExtractor):

    def forward_prediction(self, obs_features, action_features):
        obs_encodings, obs_mask, action_target, action_mask, value_target = obs_features
        action_encodings, action_mask = action_features
        concated_encodings = torch.cat((obs_encodings, action_encodings), dim=-2)
        concated_encoding_mask = torch.cat((obs_mask, action_mask), dim=-1)
        concated_encodings = self.encoder(
            concated_encodings, src_key_padding_mask=concated_encoding_mask
        )
        next_obs_pred = self.action_decoder(
            obs_encodings,
            concated_encodings,
            tgt_key_padding_mask=obs_mask,
            memory_key_padding_mask=concated_encoding_mask,
        )
        return next_obs_pred


class BilateralTransformerExtractor(TransformerExtractor):

    def __init__(
        self,
        embedding_size=16,
        num_heads=1,
        num_encoder_layers=0,
        num_decoder_layers=1,
        layer_norm_eps=1e-5,
        dim_feedforward=32,
        dropout=0,
        norm_first=False,
        activation_fn=torch.nn.ReLU,
        share_head=False,
        device=None,
        **kwargs
    ):

        super().__init__(
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            layer_norm_eps=layer_norm_eps,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            activation_fn=activation_fn,
            share_decoder=share_head,
            device=device,
        )

        muscle_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            activation=activation_fn(),
        )
        muscle_layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.muscle_encoder = nn.TransformerEncoder(
            muscle_encoder_layer, num_encoder_layers, muscle_layer_norm
        ).to(self.device)

        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            activation=activation_fn(),
        )
        time_layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.time_encoder = nn.TransformerEncoder(
            time_encoder_layer, num_encoder_layers, time_layer_norm
        ).to(self.device)

        self._reset_parameters()

    def _make_time_positional_embedding(
        self,
        input_size,
        timestep: torch.Tensor,
    ):
        time_length = input_size[1]
        d_model = input_size[3]
        # position = torch.arange(time_length, device=self.device).unsqueeze(1)
        position = timestep.unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=self.device)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(input_size, device=self.device)
        pe[:, :, :, 0::2] = torch.sin(position * div_term)[:, :, None, :]
        pe[:, :, :, 1::2] = torch.cos(position * div_term)[:, :, None, :]
        return pe

    def forward(
        self,
        features,
        src_mask: torch.Tensor,
        obs_timestep: torch.Tensor,
        act_timestep: torch.Tensor,
    ):

        assert (
            obs_timestep == act_timestep
        ).all(), "Current version: Obs and Act timestep must be the same"

        encodings, encodings_mask, action_target, action_mask, value_target = features

        shape = encodings.shape
        if len(shape) == 4:
            encodings = encodings.view(-1, *shape[2:])
            muscle_encodings = self.muscle_encoder(
                encodings,
                src_key_padding_mask=(
                    encodings_mask.view(-1, *encodings_mask.shape[2:])
                    if encodings_mask is not None
                    else None
                ),
            )
            muscle_encodings = muscle_encodings.view(
                *shape[:2], *muscle_encodings.shape[1:]
            )
        else:
            raise NotImplementedError("This model is only implemented for 4D inputs")
            # muscle_encodings = self.muscle_encoder(encodings, src_key_padding_mask=encodings_mask)

        # Add time positional encoding
        obs_time_embedding = self._make_time_positional_embedding(
            muscle_encodings.shape, obs_timestep
        )
        muscle_encodings += obs_time_embedding
        act_time_embedding = self._make_time_positional_embedding(
            action_target.shape, act_timestep
        )
        action_target += act_time_embedding

        shape = muscle_encodings.shape
        if len(shape) == 4:
            muscle_encodings = muscle_encodings.view(-1, *shape[2:])
            muscle_encodings = muscle_encodings.permute(1, 0, 2)
            # Test Time Encoding!
            time_encodings = self.time_encoder(
                muscle_encodings,
                src_key_padding_mask=(
                    encodings_mask.view(-1, *encodings_mask.shape[2:]).permute(1, 0)
                    * -1e10
                    if encodings_mask is not None
                    else None
                ),
                mask=src_mask * -1e10,
            )
            time_encodings = time_encodings.permute(1, 0, 2)
            time_encodings = time_encodings.view(*shape[:2], *time_encodings.shape[1:])
        else:
            raise NotImplementedError("This model is only implemented for 4D inputs")
            # muscle_encodings = muscle_encodings.permute(1, 0, 2)
            # time_encodings = self.time_encoder(muscle_encodings, src_key_padding_mask=encodings_mask.permute(1, 0))
            # time_encodings = time_encodings.permute(1, 0, 2)
        if encodings_mask is not None:
            time_encodings = torch.where(encodings_mask[..., None], 0, time_encodings)

        shape = time_encodings.shape
        if len(shape) == 4:
            time_encodings = time_encodings.view(-1, *shape[2:])
            action = self.decoder(
                action_target.view(-1, *action_target.shape[2:]),
                time_encodings,
                tgt_key_padding_mask=(
                    action_mask.view(-1, *action_mask.shape[2:])
                    if action_mask is not None
                    else None
                ),
                memory_key_padding_mask=(
                    encodings_mask.view(-1, *encodings_mask.shape[2:])
                    if encodings_mask is not None
                    else None
                ),
            )
            value = self.decoder(
                value_target.view(-1, *value_target.shape[2:]),
                encodings,
                memory_key_padding_mask=(
                    encodings_mask.view(-1, *encodings_mask.shape[2:])
                    if encodings_mask is not None
                    else None
                ),
            )
            action = action.view(*shape[:2], *action.shape[1:])
            value = value.view(*shape[:2], *value.shape[1:])
        else:
            raise NotImplementedError("This model is only implemented for 4D inputs")
            # action = self.decoder(
            #     action_target,
            #     time_encodings,
            #     tgt_key_padding_mask=action_mask,
            #     memory_key_padding_mask=encodings_mask,
            # )
            # value = self.decoder(
            #     value_target, encodings, memory_key_padding_mask=encodings_mask
            # )
        return action, value
