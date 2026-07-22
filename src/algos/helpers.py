from models.ppo.policies import (
    BilateralMuscleTransformerPolicy,
)
import torch.nn as nn


def get_policy(observation_space, action_space, poilcy_class, args):

    feature_extractor_config = {
        "num_layers": 0,
        "num_heads": 0,
        "embedding_size": args.embedding_size,
        "layer_norm_eps": 1e-5,
        "dim_feedforward": args.dim_feedforward,
        "dropout": 0,
        "position_embedding": "learned",
        "norm_first": True,
    }

    network_config = {
        "num_encoder_layers": args.num_layers,
        "num_decoder_layers": args.num_layers,
        "num_heads": args.num_heads,
        "layer_norm_eps": 1e-5,
        "dim_feedforward": args.dim_feedforward,
        "dropout": 0,
        "norm_first": True,
    }

    if poilcy_class == BilateralMuscleTransformerPolicy:
        network_config["k_bilateral"] = args.k_bilateral

    policy = poilcy_class(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor_kwargs=feature_extractor_config,
        lr_schedule=lambda x: args.learning_rate,
        use_lattice=False,
        use_expln=True,
        ortho_init=False,
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=network_config,
        policy_outputs_variance=args.policy_outputs_variance,
        device=args.device,
    )

    return policy
