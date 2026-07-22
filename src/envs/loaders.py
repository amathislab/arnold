import os
import json
import pickle
import torch.nn as nn
from definitions import (
    ARGS_FILE_NAME,
    ENV_CONFIG_PATH,
    ENV_FILE_NAME,
    EXPERT_PER_CONFIG_PATH,
    EXPERT_POLICIES_PATH,
    MODEL_CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    VOCABULARY_FILE_NAME,
    ENV_CONFIG_NAME,
)
from envs.environment_factory import EnvironmentFactory

from models.ppo.policies import LatticeRecurrentActorCriticPolicy
from train.algo_factory import AlgoFactory


def load_expert_policy_and_env(imitation_task, device="cuda", custom_expert_config_dict=None):
    """
    Load the expert policy and environment for a given imitation task.
    Args:
        imitation_task (str): The name of the imitation task.
        device (str): The device to load the policy on (default: "cuda").
        custom_expert_config_path (str): Path to the custom expert config file (optional).
    Returns:
        expert_policy: The loaded expert policy.
        train_env: The environment used for training.
        vecnormalize: The VecNormalize object for the environment.
    """
    expert_config = None
    if custom_expert_config_dict is not None:
        expert_config = custom_expert_config_dict.get(imitation_task)
    if expert_config is None:
        # Load the default expert config
        with open(EXPERT_PER_CONFIG_PATH, "r") as f:
            default_expert_config_dict = json.load(f)
            expert_config = default_expert_config_dict.get(imitation_task)
    if expert_config is None:
        raise ValueError(
            f"imitation task {imitation_task} not found in custom_expert_config_dict nor in {EXPERT_PER_CONFIG_PATH}"
        )

    expert_path = os.path.join(EXPERT_POLICIES_PATH, expert_config["expert"])
    model_path = os.path.join(expert_path, MODEL_FILE_NAME)
    train_env_config_path = os.path.join(expert_path, ENV_CONFIG_NAME)
    if os.path.exists(train_env_config_path):
        train_env_config = json.load(open(train_env_config_path, "r"))
    else:
        train_env_config = {}

    train_env_config["env_name"] = expert_config["train_env"]
    train_env = EnvironmentFactory.create(**train_env_config)

    if expert_config["use_arnold"]:
        vocabulary = load_vocabulary(expert_path)
    else:
        vocabulary = None
    train_model_config_path = os.path.join(expert_path, MODEL_CONFIG_FILE_NAME)
    if os.path.exists(train_model_config_path):
        train_model_config = json.load(open(train_model_config_path, "r"))
    else:
        train_model_config = None
    custom_objects = {
        "vocabulary": vocabulary,
    }

    print(f"Loading policy from {model_path}")
    expert_policy = load_expert_policy(
        algo=expert_config["algo"],
        model_path=model_path,
        custom_objects=custom_objects,
        train_model_config=train_model_config,
        device=device,
    )
    vecnormalize_path = os.path.join(expert_path, ENV_FILE_NAME)

    with open(vecnormalize_path, "rb") as f:
        vecnormalize = pickle.load(f)

    return expert_policy, train_env, vecnormalize, expert_config["use_arnold"]


def load_vocabulary(experiment_path):
    vocabulary_path = os.path.join(experiment_path, VOCABULARY_FILE_NAME)
    vocabulary = json.load(open(vocabulary_path, "r"))
    return vocabulary


def load_args(experiment_path):
    args_path = os.path.join(experiment_path, ARGS_FILE_NAME)
    args = json.load(open(args_path, "r"))
    return args


def load_expert_policy(
    algo, model_path, custom_objects=None, train_model_config=None, device="cuda"
):
    if custom_objects is None:
        custom_objects = {}
    if train_model_config is not None:
        policy_kwargs = train_model_config.get("policy_kwargs")
        if policy_kwargs is not None:
            if policy_kwargs.get("use_lattice"):
                assert (
                    algo == "recurrent_ppo"
                ), "Lattice only supported for Recurrent PPO"
                policy_class = LatticeRecurrentActorCriticPolicy
                custom_objects["policy_class"] = policy_class
            activation_fn = policy_kwargs.get("activation_fn")
            if activation_fn is not None:
                policy_kwargs["activation_fn"] = getattr(nn, activation_fn)
            custom_objects["policy_kwargs"] = policy_kwargs
    model = AlgoFactory.get_algo_class(algo).load(
        model_path, custom_objects=custom_objects, device=device
    )
    return model.policy
