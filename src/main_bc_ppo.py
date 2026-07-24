import os
import shutil
import argparse
import json
import wandb
import torch.nn as nn
from datetime import datetime
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from datetime import datetime
from definitions import ROOT_DIR, ENV_CONFIG_PATH
from envs.utilities import create_vec_env
from metrics.custom_callbacks import EnvDumpCallback, TensorboardCallback
from train.trainer import Trainer, SingleEnvTrainer
from stable_baselines3.common.policies import ActorCriticPolicy
from models.ppo.policies import MuscleTransformerPolicy
from models.ppo.policies import MuscleMlpPolicy, LatticeRecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from models.csi_model import CSIActionNet

parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument(
    "--seed", type=int, default=0, help="Seed for random number generator"
)
parser.add_argument(
    "--log_std_init", type=float, default=0.0, help="Initial log standard deviation"
)
parser.add_argument("--task", type=str, help="Name of the task", choices=[
    "elbow_pose",
    "hand_index_reach",
    "hand_little_reach",
    "hand_middle_reach",
    "hand_ring_reach",
    "hand_thumb_reach",
    "kinesis",
    "pen",
    "relocate",
    "reorient",
    "baoding_p1_ccw",
    "baoding_p1_cw",
    "baoding_p2",
    "baoding_p2_overlap"
])
parser.add_argument(
    "--use_arnold", action="store_true", help="Use Arnold policy and env"
)
parser.add_argument(
    "--load_path", type=str, default=None, help="Path to the experiment to load"
)
parser.add_argument(
    "--log_root",
    type=str,
    default=os.path.join(ROOT_DIR, "output"),
    help="Path to save the loggings",
)
parser.add_argument("--project_name", type=str, help="Name of wandb project")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of parallel environments"
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
    "--rollout_steps", type=int, default=128, help="Number of steps for each rollout"
)
parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of layers for the policy network"
)
parser.add_argument(
    "--num_heads", type=int, default=1, help="Number of heads for the policy network"
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="Batch size for the policy network"
)
parser.add_argument(
    "--network",
    type=str,
    default="mlp",
    help="Network architecture",
    choices=["mlp", "lstm", "lattice_lstm", "csi"],
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
    "--load_csi_subspace", type=str, default=None, help="Path to the CSI subspace to load"
)
parser.add_argument(
    "--csi_subspace", type=int, default=None, help="Number of components to keep in the CSI subspace"
)
parser.add_argument(
    "--load_vecnormalize",
    action="store_true",
    help="Also load the VecNormalize statistics saved next to --load_path, instead "
    "of starting from fresh observation normalization. Required to reproduce RL "
    "fine-tuning of an existing checkpoint.",
)

args = parser.parse_args()

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

if args.load_path is not None:
    experiment_name = args.load_path.split("/")[-1]
else:
    experiment_name = None

now = datetime.now()
current_time = now.strftime(r"%Y%m%d%H%M%S")
if args.use_arnold:
    prefix = f"{args.out_prefix}arnold_"
    config_prefix = "arnold_"
else:
    prefix = args.out_prefix
    config_prefix = "dense_"
run_name = f"{prefix}{args.task}_bc_ppo_seed_{args.seed}_{current_time}"
log_path = os.path.join(args.log_root, "training", "ongoing", run_name)

if args.use_arnold:
    policy_class = MuscleTransformerPolicy
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
    policy_kwargs = dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=network_config,
        features_extractor_kwargs=feature_extractor_config,
        policy_outputs_variance=args.policy_outputs_variance,
        device=args.device,
    )
else:
    if args.network == "lstm":
        policy_class = RecurrentActorCriticPolicy
        algorithm = "recurrent_ppo"
    elif args.network == "lattice_lstm":
        policy_class = LatticeRecurrentActorCriticPolicy
        algorithm = "recurrent_ppo"
    elif args.network == "mlp":
        policy_class = MuscleMlpPolicy
        algorithm = "ppo"
    elif args.network == "csi":
        policy_class = CSIActionNet
        algorithm = "ppo"
    else:
        raise ValueError(f"Network {args.network} not supported")
    policy_kwargs = dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

model_config = dict(
    policy=policy_class,
    device=args.device,
    batch_size=args.batch_size,
    n_steps=args.rollout_steps,
    learning_rate=2e-05,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=args.vf_coef,
    pg_coef=args.pg_coef,
    ent_coef=float(args.ent_coef),
    imitation_coef=args.imitation_coef,
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

    env_config_path = os.path.join(ENV_CONFIG_PATH, f"{config_prefix}{args.task}_config.json")
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    # When fine-tuning an existing checkpoint, optionally restore the VecNormalize
    # statistics saved alongside it so training continues with the same observation
    # normalization instead of re-estimating it from scratch.
    if args.load_vecnormalize:
        if args.load_path is None:
            raise ValueError("--load_vecnormalize requires --load_path")
        load_env_path = os.path.join(
            ROOT_DIR,
            args.load_path.replace("rl_model_", "rl_model_vecnormalize_").replace(
                ".zip", ".pkl"
            ),
        )
    else:
        load_env_path = None

    envs = create_vec_env(
        env_config_list=[env_config],
        num_envs_per_config=args.num_envs,
        load_env_path=load_env_path,
        multi_env=False,
        old_vocabulary=None,
        norm_reward=False,
        expert_task_list=[args.task] if args.imitation_coef > 0 else [None],
        expert_device=args.device,
    )

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    envs.save(os.path.join(log_path, "env.pkl"))

    # Define callbacks for evaluation and saving the agent
    eval_callback = EvalCallback(
        eval_env=envs,
        callback_on_new_best=EnvDumpCallback(log_path, verbose=0),
        n_eval_episodes=10,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq / args.num_envs,
        save_path=log_path,
        save_vecnormalize=True,
        verbose=2,
    )

    tensorboard_callback = TensorboardCallback(
        info_keywords=(
            "solved",
            "rwd_dense",
            "done",
        ),
    )

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
    trainer = SingleEnvTrainer(
        algo="bc_ppo",
        envs=envs,
        env_config=env_config,
        load_model_path=os.path.join(ROOT_DIR, args.load_path) if args.load_path is not None else None,
        log_dir=log_path,
        model_config=model_config,
        callbacks=callbacks_list,
        timesteps=args.num_steps,
        log_interval=args.log_interval,
    )
    # Load CSI
    import numpy as np
    import torch
    if args.network == "csi":
        if args.load_csi_subspace is not None:
            csi_projection = np.load(os.path.join(ROOT_DIR, args.load_csi_subspace))
            csi_mean = np.load(os.path.join(ROOT_DIR, args.load_csi_subspace.replace("subspace.npy", "mean.npy")))
            csi_projection = torch.from_numpy(csi_projection)
            csi_mean = torch.from_numpy(csi_mean)
            if args.csi_subspace is not None and args.csi_subspace > csi_projection.shape[0]:
                csi_subspace = csi_projection.shape[0]
            else :
                csi_subspace = args.csi_subspace
                # raise ValueError(f"CSI subspace is greater than the number of components in the subspace. {args.csi_subspace} > {csi_projection.shape[0]}")
            trainer.agent.policy.change_projection(csi_projection, csi_mean, args.csi_subspace)

    # Train agent
    trainer.train()
    trainer.save()


"""
python src/main_bc_ppo.py --task elbow_pose --imitation_coef 1 --ent_coef 1e-6 --pg_coef 1 --vf_coef 0.5

python src/main_bc_ppo.py --task elbow_pose --imitation_coef 1 --ent_coef 0 --pg_coef 0 --vf_coef 0

python src/main_bc_ppo.py --task elbow_pose --imitation_coef 0 --ent_coef 1e-6 --pg_coef 1 --vf_coef 0.5


python src/main_bc_ppo.py --task baoding_p1_ccw --imitation_coef 1 --ent_coef 1e-6 --pg_coef 1 --vf_coef 0.5

python src/main_bc_ppo.py --task baoding_p1_ccw --imitation_coef 1 --ent_coef 0 --pg_coef 0 --vf_coef 0

python src/main_bc_ppo.py --task baoding_p1_ccw --imitation_coef 0 --ent_coef 1e-6 --pg_coef 1 --vf_coef 0.5

python src/main_bc_ppo.py --task baoding_p2_overlap --num_envs 16 --imitation_coef 1 --ent_coef 0 --pg_coef 0 --vf_coef 0

python src/main_bc_ppo.py --task elbow_pose --imitation_coef 1 --ent_coef 1e-6 --pg_coef 1 --vf_coef 0.5 --use_arnold

python src/main_bc_ppo.py --task elbow_pose --imitation_coef 1 --ent_coef 0 --pg_coef 0 --vf_coef 0 --use_arnold

"""
