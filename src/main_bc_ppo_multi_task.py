import os
import shutil
import argparse
import json
import wandb
import torch.nn as nn
import numpy as np
from wandb.integration.sb3 import WandbCallback
from definitions import ROOT_DIR, ENV_CONFIG_PATH, ENV_INFO
from envs.environment_factory import ENV_NAME_TO_ID
from envs.utilities import create_vec_env, get_model_env_vocabulary_path
from metrics.custom_callbacks import TensorboardCallback, CustomCheckpointCallback
from train.trainer import Trainer
from utilities import merge_task_names
from models.ppo.policies import MuscleTransformerPolicy
from stable_baselines3.common.type_aliases import Schedule
from vocabulary import VOCABULARY


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--seed", type=int, default=0, help="Seed for random number generator"
)
parser.add_argument(
    "--log_std_init", type=float, default=0.0, help="Initial log standard deviation"
)
parser.add_argument(
    "--reset_std",
    action="store_true",
    help="Reset the standard deviation of the policy network",
)
parser.add_argument("--tasks", type=str, nargs="*", help="Name of the tasks")
parser.add_argument(
    "--load_path", type=str, default=None, help="Path to the experiment to load"
)
parser.add_argument(
    "--checkpoint_num", type=int, default=None, help="Checkpoint number to load"
)
parser.add_argument(
    "--log_root",
    type=str,
    default=os.path.join(ROOT_DIR, "output"),
    help="Path to save the loggings",
)
parser.add_argument("--project_name", type=str, help="Name of wandb project")
parser.add_argument(
    "--num_envs_per_task",
    type=int,
    default=1,
    help="Number of parallel environments per task",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size",
)
parser.add_argument(
    "--ent_coef", type=float, default=0.0, help="Entropy coefficient for PPO"
)
parser.add_argument(
    "--vf_coef", type=float, default=0.5, help="Value function coefficient for PPO"
)
parser.add_argument(
    "--pg_coef", type=float, default=1.0, help="Policy gradient coefficient for PPO"
)
parser.add_argument(
    "--imitation_coef", type=float, default=0.0, help="Imitation loss coefficient"
)
parser.add_argument(
    "--loss",
    type=str,
    default="mse",
    help="Imitation loss type",
    choices=["mse", "neglogp"],
)
parser.add_argument(
    "--constant_loss_weight",
    action="store_true",
    help="Do not divide the loss by the size of the environment's action space",
)
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--min_cosine_lr", type=float, default=None, help="Minimum learning rate")
parser.add_argument(
    "--rollout_steps", type=int, default=128, help="Number of steps for each rollout"
)
parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of layers for the policy network"
)
parser.add_argument(
    "--num_heads", type=int, default=1, help="Number of heads for the policy network"
)
parser.add_argument(
    "--dim_feedforward",
    type=int,
    default=256,
    help="Number of units in the feedforward layers",
)
parser.add_argument(
    "--embedding_size", type=int, default=64, help="Size of the embedding layer"
)
parser.add_argument(
    "--policy_outputs_variance",
    action="store_true",
    help="Use variance for the policy outputs",
)
parser.add_argument(
    "--critic_only_training",
    action="store_true",
    help="Use critic only training",
)
parser.add_argument("--norm_reward", action="store_true", help="Normalize reward")
parser.add_argument("--device", type=str, default="cuda", help="Device, cuda or cpu")
parser.add_argument(
    "--num_steps",
    type=int,
    default=10_000_000,
    help="Number of training steps once an environment is sampled",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=10,
    help="Number of epochs using the same rollouts",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=100_000,
    help="Frequency to save model per rollouts",
)
parser.add_argument("--local", action="store_true", help="Run locally without wandb")
parser.add_argument(
    "--log_interval", type=int, default=16, help="How many rollowts between loggings"
)
parser.add_argument(
    "--out_prefix", type=str, default="", help="Prefix for output files"
)
parser.add_argument(
    "--out_suffix", type=str, default="", help="Suffix for output files"
)
parser.add_argument(
    "--linear_schedule_coefs",
    action="store_true",
    help="Linearly schedule coefficients from imitation to RL",
)
parser.add_argument(
    "--separate_vf_decoder",
    action="store_true",
    help="Use separate decoders for policy and value function",
)
parser.add_argument(
    "--ablate_obs_norm",
    action="store_true",
    help="Disable observation normalization",
)
parser.add_argument(
    "--dense_reward",
    action="store_true",
    help="Use dense reward instead of sparse reward",
)
parser.add_argument(
    "--num_memory_steps",
    type=int,
    default=5,
    help="Number of past observations to include in the policy input",
)
parser.add_argument(
    "--use_expert_actions",
    action="store_true",
    help="Whether to use expert actions in environments",
)
parser.add_argument(
    "--custom_experts",
    type=str,
    default=None,
    help="Path to custom expert policies",
)
parser.add_argument(
    "--positional_encoding",
    type=str,
    default="learned",
    help="Type of positional encoding to use",
    choices=["learned", "sin_cos", "task_specific"],
)
task_embedding_group = parser.add_mutually_exclusive_group()
task_embedding_group.add_argument(
    "--task_specific_learnable",
    action="store_true",
    help="Use learnable task-specific token embeddings (the default for task_specific)",
)
task_embedding_group.add_argument(
    "--task_specific_non_learnable",
    action="store_true",
    help="Freeze task-specific token embeddings after random initialization",
)
args = parser.parse_args()
if (
    args.positional_encoding != "task_specific"
    and (args.task_specific_learnable or args.task_specific_non_learnable)
):
    parser.error("Task-specific embedding flags require --positional_encoding task_specific")

if args.load_path is not None:
    experiment_name = args.load_path.split("/")[-1]
else:
    experiment_name = None

prefix = f"{args.out_prefix}arnold_"
tasks_string = merge_task_names(args.tasks)
if args.positional_encoding == "task_specific":
    embedding_label = "nonlearn" if args.task_specific_non_learnable else "learn"
    run_name = (
        f"{args.out_prefix}arnold_{tasks_string}_task_emb_{embedding_label}"
        f"_seed_{args.seed}{args.out_suffix}"
    )
else:
    run_name = (
        f"{args.out_prefix}arnold_{tasks_string}_bc_ppo_seed_{args.seed}{args.out_suffix}"
    )
log_path = os.path.join(args.log_root, "training", "ongoing", run_name)

policy = MuscleTransformerPolicy
feature_extractor_config = {
    "num_layers": 0,
    "num_heads": 0,
    "embedding_size": args.embedding_size,
    "layer_norm_eps": 1e-5,
    "dim_feedforward": args.dim_feedforward,
    "dropout": 0,
    "position_embedding": args.positional_encoding,
    "norm_first": True,
}
if args.positional_encoding == "task_specific":
    feature_extractor_config["task_names"] = args.tasks
    feature_extractor_config["task_specific_learnable"] = (
        not args.task_specific_non_learnable
    )

network_config = {
    "num_encoder_layers": args.num_layers,
    "num_decoder_layers": args.num_layers,
    "num_heads": args.num_heads,
    "layer_norm_eps": 1e-5,
    "dim_feedforward": args.dim_feedforward,
    "dropout": 0,
    "norm_first": True,
    "share_decoder": not args.separate_vf_decoder,
}

policy_kwargs = dict(
    log_std_init=args.log_std_init,
    activation_fn=nn.ReLU,
    net_arch=network_config,
    features_extractor_kwargs=feature_extractor_config,
    policy_outputs_variance=args.policy_outputs_variance,
    critic_only_training=args.critic_only_training,
    device=args.device,
)


def linear_schedule(initial_value: float, final_value: float) -> Schedule:
    """
    Linear schedule from initial_value to final_value over the course of training.
    """

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


def double_cosine_schedule(min_value: float, max_value: float) -> Schedule:
    """
    Cosine schedule that starts at min_value, increases to max_value, and then decreases back to min_value
    over the course of training.
    
    :param min_value: Minimum learning rate.
    :param max_value: Maximum learning rate.
    :return: A function that takes progress_remaining (1.0 -> 0.0) and returns the learning rate.
    """
    def func(progress_remaining: float) -> float:
        return min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * (1 - 2 * progress_remaining)))
    
    return func

if args.min_cosine_lr is not None:
    lr = double_cosine_schedule(args.min_cosine_lr, args.lr)
else:
    lr = args.lr
model_config = dict(
    policy=policy,
    device=args.device,
    seed=args.seed,
    batch_size=args.batch_size,
    n_steps=args.rollout_steps,
    learning_rate=lr,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=(
        linear_schedule(0.0, args.vf_coef)
        if args.linear_schedule_coefs
        else args.vf_coef
    ),
    pg_coef=(
        linear_schedule(0.0, args.pg_coef)
        if args.linear_schedule_coefs
        else args.pg_coef
    ),
    ent_coef=args.ent_coef,
    imitation_coef=(
        linear_schedule(args.imitation_coef, 0.0)
        if args.linear_schedule_coefs
        else args.imitation_coef
    ),
    imitation_loss=args.loss,
    constant_loss_weight=args.constant_loss_weight,
    n_epochs=args.n_epochs,
    use_sde=False,
    policy_kwargs=policy_kwargs,
)


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(log_path, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), log_path)
    with open(os.path.join(log_path, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    env_config_list = []
    for task in args.tasks:
        task_cfg_name = f"arnold_{task}_config.json"
        if args.dense_reward:
            task_cfg_name = "dense_" + task_cfg_name
        env_config_path = os.path.join(ENV_CONFIG_PATH, task_cfg_name)
        with open(env_config_path, "r") as f:
            env_config = json.load(f)
        env_config["num_memory_steps"] = args.num_memory_steps
        env_config_list.append(env_config)

    model_path, env_path, vocabulary_path = get_model_env_vocabulary_path(
        log_path, args.load_path, args.checkpoint_num
    )

    if vocabulary_path is not None:
        with open(vocabulary_path, "r") as file:
            old_vocabulary = json.load(file)
        print("Vocabulary loaded from", vocabulary_path)
        with open(os.path.join(log_path, "vocabulary.json"), "w") as file:
            json.dump(
                VOCABULARY, file, indent=4, default=lambda _: "<not serializable>"
            )
    else:
        old_vocabulary = None
        with open(os.path.join(log_path, "vocabulary.json"), "w") as file:
            json.dump(
                VOCABULARY, file, indent=4, default=lambda _: "<not serializable>"
            )

    envs = create_vec_env(
        env_config_list=env_config_list,
        num_envs_per_config=args.num_envs_per_task,
        seed=args.seed,
        load_env_path=env_path,
        multi_env=True,
        old_vocabulary=old_vocabulary,
        norm_reward=args.norm_reward,
        norm_obs=not args.ablate_obs_norm,  # Add this line
        expert_task_list=(
            args.tasks if args.imitation_coef > 0 else [None] * len(args.tasks)
        ),
        expert_device=args.device,
        custom_expert_config_path=args.custom_experts,
    )

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    envs.save(os.path.join(log_path, "env.pkl"))

    # Define callbacks for evaluation and saving the agent
    save_freq = max(args.save_freq // (args.num_envs_per_task * len(args.tasks)), 1)
    checkpoint_callback = CustomCheckpointCallback(
        save_freq=save_freq,
        save_path=log_path,
        save_vecnormalize=True,
        verbose=2,
    )

    info_key_set = set(
        [
            f"{ENV_NAME_TO_ID[config['env_name']]}/{el}"
            for config in env_config_list
            for el in ENV_INFO[config["env_name"]]
        ]
    )
    tensorboard_callback = TensorboardCallback(info_keywords=info_key_set)

    if args.local:
        callbacks_list = [checkpoint_callback, tensorboard_callback]
    else:
        run = wandb.init(
            project=args.project_name,
            name=run_name,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb_callback = WandbCallback(
            model_save_path=f"{log_path}/{run.id}",
            gradient_save_freq=100,
            log="all",
        )
        callbacks_list = [checkpoint_callback, tensorboard_callback, wandb_callback]

    # Define trainer
    trainer = Trainer(
        algo="multi_task_bc_ppo",
        envs=envs,
        env_config_list=env_config_list,
        load_model_path=model_path,
        log_dir=log_path,
        model_config=model_config,
        callbacks=callbacks_list,
        old_vocabulary=old_vocabulary,
        log_interval=args.log_interval,
        use_expert_actions=args.use_expert_actions,
        reset_std=args.reset_std,
    )

    # Train agent
    trainer.train(total_timesteps=args.num_steps)
    trainer.save()
