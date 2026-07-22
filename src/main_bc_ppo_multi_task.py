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
    "--ent_coef", type=float, default=0, help="Entropy coefficient for PPO"
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
    choices=["learned", "sin_cos"],
)
args = parser.parse_args()

if args.load_path is not None:
    experiment_name = args.load_path.split("/")[-1]
else:
    experiment_name = None

prefix = f"{args.out_prefix}arnold_"
tasks_string = merge_task_names(args.tasks)
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


"""

python src/main_bc_ppo_multi_task.py --tasks elbow_pose baoding_p1_cw --imitation_coef 1 --ent_coef 0 --pg_coef 0 --vf_coef 0

runai submit \
    --name arnold-000 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose relocate \
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 000_ \
    --num_steps 50_000_000
    "
    
runai submit \
    --name arnold-104 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 104_ \
    --num_steps 50_000_000
    "
    
runai submit \
    --name arnold-105 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks pen \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 105_ \
    --num_steps 50_000_000
    "
    
runai submit \
    --name arnold-106 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks reorient \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 106_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-107 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks relocate \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 107_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-108 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p1_ccw \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 108_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-109 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p1_cw \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 109_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-110 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p2 \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 110_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-111 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p2_overlap \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 111_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-112 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 112_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-113 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_index_reach \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 113_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-114 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_middle_reach \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 114_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-115 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_ring_reach \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 115_ \
    --num_steps 50_000_000
    "
    
runai submit \
    --name arnold-116 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_little_reach \
    --num_envs_per_task 32 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 116_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-117 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.25 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 16 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 117_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-120 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.35 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p1_cw baoding_p1_ccw baoding_p2_overlap baoding_p2 \
    --num_envs_per_task 8 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 120_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-119 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach\
    --num_envs_per_task 6 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 119_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-121 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 1 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen relocate elbow_pose baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
    --num_envs_per_task 4 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 121_ \
    --num_steps 50_000_000
    "
runai submit \
    --name arnold-122 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_index_reach pen \
    --num_envs_per_task 4 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 122_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-123 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 16 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --num_layers 4 \
    --num_heads 2 \
    --policy_outputs_variance \
    --out_prefix 123_ \
    --num_steps 50_000_000
    "
runai submit \
    --name arnold-124 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 1 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
    --num_envs_per_task 4 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 124_ \
    --num_steps 50_000_000
    "

runai submit \
    --name arnold-125 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 16 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --pg_coef 1.0 \
    --imitation_coef 1.0 \
    --policy_outputs_variance \
    --num_steps 1_000_000 \
    --linear_schedule_coefs
    --out_prefix 125_ \
    "
runai submit \
    --name arnold-126 \
    --image registry.rcp.epfl.ch/arnold/bc:latest \
    --gpu 0.3 \
    --run-as-uid 174516 \
    --run-as-gid 79678 \
    --existing-pvc claimname=upamathis-scratch,path=/users \
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
    cd /users/alberto/arnold; \
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 16 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --pg_coef 1.0 \
    --imitation_coef 1.0 \
    --policy_outputs_variance \
    --batch_size 256 \
    --rollout_steps 2048 \
    --num_steps 1_000_000 \
    --linear_schedule_coefs
    --out_prefix 126_ \
    "
    
runai submit\
    --name arnold-127 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks \
                hand_index_reach \
                elbow_pose \
            --num_envs_per_task=16\
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --out_prefix=127_ \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3
            "
    
runai submit\
    --name arnold-128 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=128_ \
            "

runai submit\
    --name arnold-129 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=129_ \
            "

runai submit\
    --name arnold-130 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=130_ \
            "
runai submit\
    --name arnold-131 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --linear_schedule_coefs \
            --out_prefix=131_ \
            "
runai submit\
    --name arnold-132 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 32 --memory 48Gi --cpu-limit 32 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.01 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=132_ \
            "

runai submit\
    --name arnold-135 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=2e-8 \
            --vf_coef=0.01 \
            --pg_coef=0.02 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --linear_schedule_coefs \
            --out_prefix=135_ \
            "
runai submit\
    --name arnold-136 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=2e-8 \
            --vf_coef=0.01 \
            --pg_coef=0.02 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --linear_schedule_coefs \
            --out_prefix=136_ \
            "
runai submit\
    --name arnold-137 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=2 \
            --num_layers=4 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=137_ \
            "
            
runai submit\
    --name arnold-138 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=138_ \
            "
runai submit\
    --name arnold-139 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=139_ \
            "

runai submit\
    --name arnold-140 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=1 \
            --out_prefix=140_ \
            "

runai submit\
    --name arnold-141 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=1 \
            --separate_vf_decoder \
            --out_prefix=141_ \
            "

runai submit\
    --name arnold-142 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --load_path output/training/ongoing/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0 \
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=142_ \
            "

runai submit\
    --name arnold-143 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=30_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=143_ \
            --seed=1 \
            "

runai submit\
    --name arnold-144 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap relocate elbow_pose \
            --load_path output/training/ongoing/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0 \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=144_ \
            "

runai submit\
    --name arnold-145 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw relocate elbow_pose \
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=145_ \
            "

runai submit\
    --name arnold-146 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=146_ \
            "

runai submit\
    --name arnold-147 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=147_ \
            --seed 1 \
            "

runai submit\
    --name arnold-148 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=148_ \
            --seed 2 \
            " 

runai submit\
    --name arnold-149 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --out_prefix=149_ \
            --seed 0 \
            "

runai submit\
    --name arnold-150 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=1e-4 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --out_prefix=150_ \
            --seed 0 \
            "

runai submit\
    --name arnold-151 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=1e-8 \
            --vf_coef=1e-4 \
            --pg_coef=1e-1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --out_prefix=151_ \
            --seed 0 \
            "
runai submit\
    --name arnold-152 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=1e-8 \
            --vf_coef=1e-4 \
            --pg_coef=1e-1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --policy_outputs_variance \
            --out_prefix=152_ \
            --seed 0 \
            "
runai submit\
    --name arnold-153 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=1e-8 \
            --vf_coef=1e-4 \
            --pg_coef=1e-1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=153_ \
            --seed 0 \
            "
runai submit\
    --name arnold-154 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=1e-4 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --norm_reward \
            --out_prefix=154_ \
            --seed 0 \
            "

runai submit\
    --name arnold-155 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --norm_reward \
            --out_prefix=155_ \
            --seed 0 \
            "
            
runai submit\
    --name arnold-156 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --ablate_obs_norm \
            --out_prefix=156_ \
            --seed 0 \
            "
runai submit\
    --name arnold-157 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=30_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --norm_reward \
            --out_prefix=157_ \
            --seed 0 \
            --num_memory_steps 20 \
            "
runai submit\
    --name arnold-158 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=30_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --norm_reward \
            --out_prefix=158_ \
            --seed 0 \
            --num_memory_steps 50 \
            "

runai submit\
    --name arnold-159 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=30_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --norm_reward \
            --out_prefix=159_ \
            --seed 0 \
            --num_memory_steps 0 \
            "
runai submit\
    --name arnold-160 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=160_ \
            --seed 0 \
            "

runai submit\
    --name arnold-161 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate\
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=161_ \
            --seed 0 \
            "

runai submit\
    --name arnold-162 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks kinesis \
            --num_envs_per_task 32 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=162_ \
            "

runai submit\
    --name arnold-162 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks kinesis \
            --num_envs_per_task 32 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=162_ \
            "
runai submit\
    --name arnold-163 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=163_ \
            "

runai submit\
    --name arnold-164 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis\
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=164_ \
            --seed 0 \
            "

runai submit\
    --name arnold-165 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis\
            --num_envs_per_task 4 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --norm_reward \
            --constant_loss_weight \
            --out_prefix=165_ \
            --seed 0 \
            "

runai submit\
    --name arnold-166 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis\
            --num_envs_per_task 4 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --constant_loss_weight \
            --norm_reward \
            --out_prefix=166_ \
            --seed 0 \
            "

runai submit\
    --name arnold-167 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/163_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=0.001 \
            --log_interval=1 \
            --n_epochs=3 \
            --out_prefix=167_ \
            "
runai submit\
    --name arnold-168 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.5 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/164_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=168_ \
            --seed 0 \
            "

runai submit\
    --name arnold-169 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.65 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=169_ \
            --seed 0 \
            "

runai submit\
    --name arnold-170 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=170_ \
            --seed 0 \
            "

runai submit\
    --name arnold-171 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=171_ \
            --seed 0 \
            "
runai submit\
    --name arnold-172 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.65 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --ablate_obs_norm \
            --out_prefix=172_ \
            --seed 0 \
            "
runai submit\
    --name arnold-173 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --ablate_obs_norm \
            --out_prefix=173_ \
            --seed 0 \
            "

runai submit\
    --name arnold-174 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --out_prefix=174_ \
            --seed 0 \
            "

runai submit\
    --name arnold-175 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=175_ \
            --seed 0 \
            "

runai submit\
    --name arnold-176 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_ring_reach \
                baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=176_ \
            --seed 0 \
            "

runai submit\
    --name arnold-177 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.65 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks \
                hand_thumb_reach \
                hand_index_reach \
                hand_middle_reach \
                hand_ring_reach \
                hand_little_reach \
                elbow_pose \
                reorient reorient reorient \
                pen pen pen \
                baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw \
                baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw \
                baoding_p2 baoding_p2 baoding_p2 baoding_p2 baoding_p2 \
                baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap \
                relocate relocate relocate relocate relocate relocate relocate relocate relocate relocate \
                kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis\
            --num_envs_per_task 1 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=177_ \
            --seed 0 \
            "

runai submit\
    --name arnold-178 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks \
                hand_thumb_reach \
                hand_index_reach \
                hand_middle_reach \
                hand_ring_reach \
                hand_little_reach \
                elbow_pose \
                reorient reorient reorient \
                pen pen pen \
                baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw \
                baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw \
                baoding_p2 baoding_p2 baoding_p2 baoding_p2 baoding_p2 \
                baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap \
                relocate relocate relocate relocate relocate relocate relocate relocate relocate relocate \
                kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis\
            --num_envs_per_task 1 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=178_ \
            --seed 0 \
            "

runai submit\
    --name arnold-179 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks \
                hand_thumb_reach \
                hand_index_reach \
                hand_middle_reach \
                hand_ring_reach \
                hand_little_reach \
                elbow_pose \
                reorient reorient reorient \
                pen pen pen \
                baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw baoding_p1_cw \
                baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw baoding_p1_ccw \
                baoding_p2 baoding_p2 baoding_p2 baoding_p2 baoding_p2 \
                baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap baoding_p2_overlap \
                relocate relocate relocate relocate relocate relocate relocate relocate relocate relocate \
                kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis kinesis\
            --num_envs_per_task 1 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=179_ \
            --seed 0 \
            "

runai submit\
    --name arnold-180 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks pen \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=180_ \
            --seed 0 \
            "

runai submit\
    --name arnold-181 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks reorient \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=181_ \
            --seed 0 \
            "

runai submit\
    --name arnold-182 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_middle_reach \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=182_ \
            --seed 0 \
            "

runai submit\
    --name arnold-183 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_little_reach \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=183_ \
            --seed 0 \
            "

runai submit\
    --name arnold-184 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks pen \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=184_ \
            --seed 0 \
            "

runai submit\
    --name arnold-185 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.3 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks reorient \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=185_ \
            --seed 0 \
            "

runai submit\
    --name arnold-186 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_middle_reach \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=186_ \
            --seed 0 \
            "

runai submit\
    --name arnold-187 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_little_reach \
            --load_path output/training/ongoing/176_arnold_htr_hir_hrr_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=187_ \
            --seed 0 \
            "

runai submit\
    --name arnold-169 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.65 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=169_ \
            --seed 0 \
            "

runai submit\
    --name arnold-170 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=170_ \
            --seed 0 \
            "
            
runai submit\
    --name arnold-188 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks pen \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=188_ \
            --seed 0 \
            "

runai submit\
    --name arnold-189 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks reorient \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=189_ \
            --seed 0 \
            "

runai submit\
    --name arnold-190 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_middle_reach \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=190_ \
            --seed 0 \
            "

runai submit\
    --name arnold-191 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_little_reach \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-4 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=191_ \
            --seed 0 \
            "

runai submit\
    --name arnold-192 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks elbow_pose \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=1_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=192_ \
            --seed 0 \
            --reset_std \
            "

runai submit\
    --name arnold-193 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=193_ \
            --seed 0 \
            --reset_std \
            "
runai submit\
    --name arnold-194 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2 \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=194_ \
            --seed 0 \
            --reset_std \
            "

runai submit\
    --name arnold-195 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=195_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "
runai submit\
    --name arnold-196 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2 \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=196_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-197 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.25 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks elbow_pose \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=1_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=197_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-198 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks relocate \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=198_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            "

runai submit \
    --name arnold-199 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=199_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-200 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=200_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            "

runai submit\
    --name arnold-201 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=201_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-202 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=202_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-203 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 0 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=203_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            "

runai submit\
    --name arnold-204 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks elbow_pose \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=204_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "
            
runai submit\
    --name arnold-205 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=205_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-206 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=206_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-207 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks relocate \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=207_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            "

runai submit\
    --name arnold-208 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=208_ \
            --seed 0 \
            --reset_std \
            --log_std_init -2 \
            --dense_reward \
            --critic_only_training \
            "

runai submit\
    --name arnold-209 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=209_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            --critic_only_training \
            "

runai submit\
    --name arnold-210 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=210_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-211 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks relocate \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=211_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-212 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/209_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=212_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-213 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=213_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            --critic_only_training \
            "

runai submit\
    --name arnold-214 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p1_ccw \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=214_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-215 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p1_cw \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=215_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "
runai submit\
    --name arnold-216 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2 \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=216_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-217 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks baoding_p2_overlap \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=217_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-218 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks elbow_pose \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=218_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-219 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_index_reach \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=219_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-220 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=220_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-221 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_middle_reach \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=221_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-222 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_ring_reach \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=222_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-223 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_little_reach \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=223_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-224 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks kinesis \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=224_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "
            
runai submit\
    --name arnold-225 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks relocate \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=225_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-226 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks reorient \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=226_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "

runai submit\
    --name arnold-227 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.4 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0 \
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks pen \
            --load_path output/training/ongoing/213_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 32 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=0 \
            --num_steps=20_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=2e-5 \
            --min_cosine_lr 1e-8 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --out_prefix=227_ \
            --seed 0 \
            --reset_std \
            --log_std_init -3 \
            --dense_reward \
            "
runai submit\
    --name arnold-246 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --custom_experts data/expert_configs/arnold_experts_0.json \
            --out_prefix=246_ \
            --seed 0 \
            "

runai submit\
    --name arnold-247 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --load_path output/training/ongoing/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=10_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --custom_experts data/expert_configs/arnold_experts_0.json \
            --dense_reward \
            --out_prefix=247_ \
            --seed 0 \
            "

runai submit\
    --name arnold-248 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --dense_reward \
            --out_prefix=248_ \
            --seed 0 \
            "

runai submit\
    --name arnold-249 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --dense_reward \
            --out_prefix=249_ \
            --seed 1 \
            "

runai submit\
    --name arnold-250 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --dense_reward \
            --out_prefix=250_ \
            --seed 2 \
            "

runai submit\
    --name arnold-251 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=0 \
            --vf_coef=0.5 \
            --pg_coef=0 \
            --imitation_coef=1 \
            --num_steps=50_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-3 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --dense_reward \
            --out_prefix=251_ \
            --seed 0 \
            "

runai submit\
    --name arnold-287 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.6 \
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 64Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_bc_ppo_multi_task.py \
            --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
                reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis\
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
                relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
            --num_envs_per_task 2 \
            --ent_coef=1e-6 \
            --vf_coef=0.5 \
            --pg_coef=1 \
            --imitation_coef=1 \
            --num_steps=5_000_000 \
            --batch_size=128 \
            --rollout_steps=512 \
            --embedding_size=128 \
            --dim_feedforward=512 \
            --num_heads=4 \
            --num_layers=6 \
            --lr=1e-5 \
            --log_interval=1 \
            --n_epochs=3 \
            --separate_vf_decoder \
            --policy_outputs_variance \
            --norm_reward \
            --dense_reward \
            --out_prefix=287_ \
            --seed 0 \
            "
            
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 1 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --pg_coef 1.0 \
    --imitation_coef 1.0 \
    --num_steps 1_000_000 \
    --local \
    --out_prefix test_ \
    --save_freq 1000 \
    --num_memory_steps 0
    
    
    python src/main_bc_ppo_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix test_ \
    --num_steps 50_000_000

    --local
    
    
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap \
    --num_envs_per_task 1 \
    --ent_coef=0 \
    --vf_coef=0 \
    --pg_coef=0 \
    --imitation_coef=1 \
    --num_steps=10_000_000 \
    --batch_size=128 \
    --rollout_steps=512 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --lr=0.001 \
    --log_interval=1 \
    --n_epochs=3 \
    --out_prefix=128_ \
    --save_freq=1 \
    --local

    python src/main_bc_ppo_multi_task.py \
    --tasks baoding_p1_cw baoding_p1_ccw baoding_p2_overlap baoding_p2 \
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix 120_ \
    --num_steps 50_000_000 \
    --local

    python src/main_bc_ppo_multi_task.py \
    --tasks hand_index_reach relocate \
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix test_ \
    --num_steps 50_000_000 \
    --save_freq 100000 \
    --local

    python src/main_bc_ppo_multi_task.py \
    --tasks kinesis \
    --num_envs_per_task 4 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --load_path data/student_policies/arnold_multi_task/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0 \
    --policy_outputs_variance \
    --out_prefix test_ \
    --num_steps 40_000_000 \
    --save_freq 100000 \
    --batch_size 32 \
    --rollout_steps 128 \
    --log_interval 1 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --num_heads=4 \
    --num_layers=6 \
    --separate_vf_decoder \
    --local

    python src/main_bc_ppo_multi_task.py \
    --tasks reorient \
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix test_ \
    --num_steps 40_000_000 \
    --save_freq 100000 \
    --batch_size 32 \
    --n_epochs 1 \
    --rollout_steps 1500 \
    --log_interval 1 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --num_heads=4 \
    --num_layers=6 \
    --separate_vf_decoder \
    --custom_experts data/expert_configs/arnold_experts_0.json \
    --use_expert_actions \
    --local

    python src/main_bc_ppo_multi_task.py \
    --tasks reorient kinesis baoding_p2_overlap baoding_p1_cw\
    --num_envs_per_task 1 \
    --ent_coef 0 \
    --vf_coef 0 \
    --pg_coef 0 \
    --imitation_coef 1 \
    --policy_outputs_variance \
    --out_prefix test_ \
    --num_steps 40_000_000 \
    --save_freq 100000 \
    --batch_size 32 \
    --n_epochs 1 \
    --rollout_steps 1500 \
    --log_interval 1 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --num_heads=4 \
    --num_layers=6 \
    --separate_vf_decoder \
    --custom_experts data/expert_configs/arnold_experts_0.json \
    --use_expert_actions \
    --local

"""