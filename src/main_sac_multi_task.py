import os
import shutil
import argparse
import json
import wandb
import numpy as np
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from definitions import ROOT_DIR, ENV_CONFIG_PATH, ENV_INFO
from envs.environment_factory import ENV_NAME_TO_ID
from envs.utilities import create_vec_env, get_model_env_vocabulary_path
from metrics.custom_callbacks import TensorboardCallback, CustomCheckpointCallback
from utilities import merge_task_names
from models.feature_extractors import MultiTaskFlatFeaturesExtractor
from models.ppo.ppo import MultiEnvPPO
from vocabulary import VOCABULARY


parser = argparse.ArgumentParser(description="Multi-task SAC/PPO with MLP policy")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tasks", type=str, nargs="*", help="Names of the tasks")
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--checkpoint_num", type=int, default=None)
parser.add_argument(
    "--log_root",
    type=str,
    default=os.path.join(ROOT_DIR, "output"),
)
parser.add_argument("--project_name", type=str, help="Name of wandb project")
parser.add_argument("--num_envs_per_task", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument(
    "--hidden_size", type=int, default=256, help="Hidden layer width for MLP"
)
parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of MLP hidden layers"
)
parser.add_argument(
    "--algo",
    type=str,
    default="sac",
    choices=["sac", "ppo"],
    help="RL algorithm to use",
)
# SAC-specific args
parser.add_argument(
    "--buffer_size", type=int, default=1_000_000, help="Replay buffer size (SAC only)"
)
parser.add_argument("--tau", type=float, default=0.005, help="Polyak update coefficient (SAC only)")
parser.add_argument(
    "--ent_coef",
    type=str,
    default="auto",
    help="SAC entropy coefficient (float or 'auto'); for PPO, use a float",
)
parser.add_argument(
    "--learning_starts",
    type=int,
    default=100,
    help="Steps before SAC starts learning (SAC only)",
)
parser.add_argument(
    "--train_freq",
    type=int,
    default=1,
    help="Update the model every train_freq steps (SAC only)",
)
parser.add_argument(
    "--gradient_steps",
    type=int,
    default=-1,
    help="Gradient steps per environment step (SAC only, -1 = train_freq steps)",
)
# PPO-specific args
parser.add_argument(
    "--rollout_steps", type=int, default=2048, help="Steps per rollout collection (PPO only)"
)
parser.add_argument(
    "--n_epochs", type=int, default=10, help="Epochs per rollout update (PPO only)"
)
parser.add_argument(
    "--clip_range", type=float, default=0.2, help="PPO clipping range (PPO only)"
)
parser.add_argument(
    "--gae_lambda", type=float, default=0.95, help="GAE lambda (PPO only)"
)
parser.add_argument(
    "--vf_coef", type=float, default=0.5, help="Value function coefficient (PPO only)"
)
# Shared args
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--norm_reward", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num_steps", type=int, default=10_000_000)
parser.add_argument("--save_freq", type=int, default=100_000)
parser.add_argument("--local", action="store_true", help="Run without wandb")
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--out_prefix", type=str, default="")
parser.add_argument("--out_suffix", type=str, default="")
parser.add_argument("--dense_reward", action="store_true")
parser.add_argument(
    "--num_memory_steps",
    type=int,
    default=5,
    help="Number of past observations stacked in the policy input",
)
parser.add_argument("--custom_experts", type=str, default=None)
args = parser.parse_args()

tasks_string = merge_task_names(args.tasks)
run_name = (
    f"{args.out_prefix}arnold_{tasks_string}_{args.algo}_mlp_seed_{args.seed}{args.out_suffix}"
)
log_path = os.path.join(args.log_root, "training", "ongoing", run_name)

policy_kwargs = dict(
    features_extractor_class=MultiTaskFlatFeaturesExtractor,
    net_arch=[args.hidden_size] * args.num_layers,
)

# SAC ent_coef can be "auto" or a float; PPO requires a float
ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)


if __name__ == "__main__":
    os.makedirs(log_path, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), log_path)
    with open(os.path.join(log_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4, default=lambda _: "<not serializable>")
    with open(os.path.join(log_path, "vocabulary.json"), "w") as f:
        json.dump(VOCABULARY, f, indent=4)

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

    model_path, env_path, _ = get_model_env_vocabulary_path(
        log_path, args.load_path, args.checkpoint_num
    )

    envs = create_vec_env(
        env_config_list=env_config_list,
        num_envs_per_config=args.num_envs_per_task,
        load_env_path=env_path,
        multi_env=True,
        old_vocabulary=None,
        norm_reward=args.norm_reward,
        norm_obs=True,
        expert_task_list=[None] * len(args.tasks),
        expert_device=args.device,
        custom_expert_config_path=args.custom_experts,
    )

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    envs.save(os.path.join(log_path, "env.pkl"))

    save_freq = max(
        args.save_freq // (args.num_envs_per_task * len(args.tasks)), 1
    )
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
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            model_save_path=f"{log_path}/{run.id}",
            gradient_save_freq=100,
            log="all",
        )
        callbacks_list = [checkpoint_callback, tensorboard_callback, wandb_callback]

    if args.algo == "sac":
        if model_path is not None:
            print(f"Loading model from {model_path}")
            agent = SAC.load(
                model_path,
                env=envs,
                tensorboard_log=log_path,
                device=args.device,
            )
        else:
            print("\nNo model path provided. Initializing new SAC model.\n")
            agent = SAC(
                policy="MultiInputPolicy",
                env=envs,
                learning_rate=args.lr,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=args.gamma,
                ent_coef=ent_coef,
                learning_starts=args.learning_starts,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                policy_kwargs=policy_kwargs,
                verbose=2,
                tensorboard_log=log_path,
                device=args.device,
                seed=args.seed,
            )
    else:  # ppo
        if model_path is not None:
            print(f"Loading model from {model_path}")
            agent = MultiEnvPPO.load(
                model_path,
                env=envs,
                tensorboard_log=log_path,
                device=args.device,
            )
        else:
            print("\nNo model path provided. Initializing new PPO model.\n")
            agent = MultiEnvPPO(
                policy="MultiInputPolicy",
                env=envs,
                learning_rate=args.lr,
                n_steps=args.rollout_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=float(ent_coef) if ent_coef != "auto" else 0.0,
                vf_coef=args.vf_coef,
                policy_kwargs=policy_kwargs,
                verbose=2,
                tensorboard_log=log_path,
                device=args.device,
                seed=args.seed,
            )

    agent.learn(
        total_timesteps=args.num_steps,
        callback=callbacks_list,
        reset_num_timesteps=False,
        log_interval=args.log_interval,
    )

    agent.save(os.path.join(log_path, "final_model.zip"))
    envs.save(os.path.join(log_path, "final_env.pkl"))


"""
python src/main_sac_multi_task.py \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 1 \
    --local \
    --num_steps 1_000_000 \
    --hidden_size 256 \
    --num_layers 2

python src/main_sac_multi_task.py \
    --algo ppo \
    --tasks elbow_pose hand_index_reach \
    --num_envs_per_task 1 \
    --local \
    --num_steps 1_000_000 \
    --hidden_size 256 \
    --num_layers 2
"""
