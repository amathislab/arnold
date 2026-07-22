import os
import shutil
import argparse
import torch.nn as nn
import json
import wandb
from definitions import ROOT_DIR, ENV_INFO, ENV_CONFIG_PATH
from metrics.custom_callbacks import TensorboardCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from train.trainer import Trainer
from models.ppo.policies import MuscleTransformerPolicy
from models.rollout_buffers import MultiEnvDictRolloutBuffer
from envs.environment_factory import ENV_NAME_TO_ID
from envs.utilities import create_vec_env
from wandb.integration.sb3 import WandbCallback
from imitation.util import logger as imit_logger
from vocabulary import VOCABULARY
from utilities import get_model_env_vocabulary_path, merge_strings


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for random number generator",
)
parser.add_argument(
    "--saveroot",
    help="Root directory for saving models",
    default=os.path.join(ROOT_DIR, "output"),
)
parser.add_argument(
    "--logroot",
    help="Root directory for logging",
    default=os.path.join(ROOT_DIR, "output"),
)
parser.add_argument(
    "--log_std_init",
    type=float,
    default=0.0,
    help="Initial log standard deviation",
)
parser.add_argument(
    "--env_names",
    type=str,
    nargs="*",
    default=["MuscleElbowPoseFixed"],
    help="List of environment names",
)
parser.add_argument(
    "--load_path",
    type=str,
    help="Path to the experiment to load",
)
parser.add_argument(
    "--checkpoint_num",
    type=int,
    default=None,
    help="Number of the checkpoint to load",
)
parser.add_argument(
    "--num_envs_per_class",
    type=int,
    default=1,
    help="Number of parallel environments",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Learning device, cuda or cpu",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=10_000_000,
    help="Number of training steps once an environment is sampled",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=500_000,
    help="Save a checkpoint every N number of steps",
)
parser.add_argument(
    "--algo",
    type=str,
    default="ppo",
    help="Which algorithm to use",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Size of the batch",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="Learning rate",
)
parser.add_argument(
    "--policy_outputs_variance",
    action="store_true",
    default=False,
    help="Flag to use SDE",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=4,
    help="Number of the encoder and decoder layers",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=1,
    help="Number of heads of the encoder and of the decoder",
)
parser.add_argument(
    "--embedding_size",
    type=int,
    default=64,
    help="Size of the embedding in the transformer",
)
parser.add_argument(
    "--dim_feedforward",
    type=int,
    default=256,
    help="Size of the fully-connected hidden layer in the transformer",
)
parser.add_argument(
    "--out_suffix",
    type=str,
    default="",
    help="Suffix added to the experiment folder name",
)
parser.add_argument(
    "--project_name",
    type=str,
    default="arnold",
    help="Project name for wandb",
)
parser.add_argument(
    "--norm_reward",
    action="store_true",
    default=False,
    help="Flag to use per-environment reward normalization",
)
parser.add_argument(
    "--local",
    action="store_true",
    default=False,
    help="Run experiment locally withouth wandb",
)

args = parser.parse_args()

env_names_str = merge_strings(args.env_names)

run_name = f"{env_names_str}_{args.algo}_seed_{args.seed}_nl_{args.num_layers}_nh_{args.num_heads}_es_{args.embedding_size}_df_{args.dim_feedforward}{args.out_suffix}"
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", "ongoing", run_name)


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

env_config_list = []
for imitation_task in args.env_names:
    eval_env_config_path = os.path.join(
        ENV_CONFIG_PATH, f"{imitation_task}_config.json"
    )
    with open(eval_env_config_path, "r") as f:
        eval_env_config = json.load(f)
    env_config_list.append(eval_env_config)

# PCGrad requires a special rollout buffer
if args.algo == "pcgrad_ppo":
    rollout_buffer_class = MultiEnvDictRolloutBuffer
    rollout_buffer_kwargs = dict(
        num_env_classes=len(env_config_list),
        num_envs_per_class=args.num_envs_per_class,
    )
else:
    rollout_buffer_class = None
    rollout_buffer_kwargs = {}

model_config = dict(
    policy=MuscleTransformerPolicy,
    seed=args.seed,
    device=args.device,
    batch_size=args.batch_size,
    n_steps=128,
    learning_rate=args.learning_rate,
    ent_coef=1e-06,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.8,
    n_epochs=10,
    use_sde=False,
    sde_sample_freq=-1,  # number of steps
    rollout_buffer_class=rollout_buffer_class,
    rollout_buffer_kwargs=rollout_buffer_kwargs,
    policy_kwargs=dict(
        features_extractor_kwargs=feature_extractor_config,
        use_lattice=False,
        use_expln=True,
        ortho_init=False,
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=network_config,
        lattice_kwargs=dict(
            std_clip=(1e-3, 10),
            std_reg=0,
        ),
        policy_outputs_variance=args.policy_outputs_variance,
        device=args.device,
    ),
)


if __name__ == "__main__":
    if not args.local:
        run = wandb.init(
            project=args.project_name,
            name=run_name,
            config=model_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    # ensure tensorboard log directory exists and copy this file to track
    save_path = os.path.join(args.saveroot, "models", run_name)
    log_path = os.path.join(args.logroot, "training", run_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), save_path)
    with open(os.path.join(save_path, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    # TODO: we need to find a way to deal with changes in the vocabulary when the training
    # restarts due to interrruption in the cluster
    with open(os.path.join(save_path, "vocabulary.json"), "w") as file:
        json.dump(VOCABULARY, file, indent=4, default=lambda _: "<not serializable>")

    # Define the callbacks
    save_freq = max(
        args.save_every // (args.num_envs_per_class * len(args.env_names)), 1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        save_vecnormalize=True,
        verbose=1,
    )
    # info_key_set = set(
    #     [
    #         f"{ENV_NAME_TO_ID[env_name]}/{el}"
    #         for env_name in args.env_names
    #         for el in ENV_INFO[env_name]
    #     ]
    # )
    # tensorboard_callback = TensorboardCallback(info_keywords=info_key_set)

    if not args.local:
        wandb_callback = WandbCallback(
            model_save_path=f"{save_path}/{run.id}",
            gradient_save_freq=100,
            log="all",
        )

    model_path, env_path, vocabulary_path = get_model_env_vocabulary_path(
        save_path, args.load_path, args.checkpoint_num
    )

    if vocabulary_path is not None:
        with open(vocabulary_path, "r") as file:
            old_vocabulary = json.load(file)
        print("Vocabulary loaded from", vocabulary_path)
    else:
        old_vocabulary = None

    # Create the environment
    envs = create_vec_env(
        env_config_list=env_config_list,
        num_envs_per_config=args.num_envs_per_class,
        load_env_path=env_path,
        tensorboard_log=TENSORBOARD_LOG,
        old_vocabulary=old_vocabulary,
        norm_reward=args.norm_reward,
        seed=args.seed,
    )

    # Define trainer
    callbacks = [checkpoint_callback]
    if not args.local:
        callbacks.append(wandb_callback)
    trainer = Trainer(
        algo=args.algo,
        envs=envs,
        env_config_list=env_config_list,
        load_model_path=model_path,
        log_dir=log_path,
        model_config=model_config,
        custom_logger=imit_logger.configure(
            folder=TENSORBOARD_LOG,
            format_strs=(
                ["stdout", "log", "csv", "tensorboard"]
                if args.local
                else ["stdout", "log", "csv", "wandb"]
            ),
        ),
        callbacks=callbacks,
        old_vocabulary=old_vocabulary,
    )

    # Train agent
    trainer.train(total_timesteps=args.num_steps)

    if not args.local:
        run.finish()

#####
# python src/main_arnold.py --env_names hand_index_middle_reach --num_envs_per_class 2 --device cpu --policy_outputs_variance --num_layers 2 --embedding_size 128 --out_suffix two_finger_scratch --norm_reward --local
#####
