import os
import shutil
import argparse
import json
import wandb
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from datetime import datetime
from definitions import ROOT_DIR, ENV_CONFIG_PATH
from envs.utilities import create_vec_env
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback, TensorboardCallback
from train.trainer import SingleEnvTrainer
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
    "--load_policy", type=str, default=None, help="Path to the experiment to load"
)
parser.add_argument(
    "--load_csi_subspace", type=str, default=None, help="Path to the CSI subspace to load"
)
parser.add_argument(
    "--csi_subspace", type=int, default=None, help="Number of components to keep in the CSI subspace"
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
    "--network",
    type=str,
    default="mlp",
    help="Network architecture",
    choices=["mlp", "lstm", "lattice_lstm", "csi"],
)
parser.add_argument(
    "--rollout_steps", type=int, default=100, help="Number of steps for each rollout"
)
parser.add_argument("--device", type=str, default="cuda", help="Device, cuda or cpu")
parser.add_argument(
    "--num_steps",
    type=int,
    default=10_000_000,
    help="Number of training steps once an environment is sampled",
)
parser.add_argument(
    "--save_freq", type=int, default=100_000, help="Frequency to save model per rollouts"
)
parser.add_argument("--local", action="store_true", help="Run locally without wandb")
parser.add_argument(
    "--log_interval", type=int, default=16, help="How many rollouts between loggings"
)
parser.add_argument(
    "--out_prefix", type=str, default="", help="Prefix for output files"
)
parser.add_argument(
    "--out_suffix", type=str, default="", help="Suffix for output files"
)
args = parser.parse_args()

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

if args.load_policy is not None:
    experiment_name = args.load_policy.split("/")[-1]
else:
    experiment_name = None

now = datetime.now()
current_time = now.strftime(r"%Y%m%d%H%M%S")
run_name = f"{args.out_prefix}{args.task}_{args.network}_ppo_seed_{args.seed}{args.out_suffix}_{current_time}"
log_path = os.path.join(args.log_root, "training", "ongoing", run_name)

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
if args.network == "lattice_lstm":
    policy_kwargs["use_lattice"] = True
    policy_kwargs["use_expln"] = True
    policy_kwargs["ortho_init"] = False
    policy_kwargs["std_reg"] = 0

model_config = dict(
    policy=policy_class,
    device=args.device,
    batch_size=256,
    n_steps=args.rollout_steps,
    learning_rate=2e-05,
    ent_coef=1e-06,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.8,
    n_epochs=10,
    use_sde=False,
    policy_kwargs=policy_kwargs,
)

if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(log_path, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), log_path)
    with open(os.path.join(log_path, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    config_prefix = "dense_"
    env_config_path = os.path.join(ENV_CONFIG_PATH, f"{config_prefix}{args.task}_config.json")
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    
    envs = create_vec_env(
        env_config_list=[env_config],
        num_envs_per_config=args.num_envs,
        load_env_path=os.path.join(ROOT_DIR, args.load_policy.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl")),
        multi_env=False,
        old_vocabulary=None,
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
        algo=algorithm,
        envs=envs,
        env_config=env_config,
        load_model_path=os.path.join(ROOT_DIR, args.load_policy),
        log_dir=log_path,
        model_config=model_config,
        callbacks=callbacks_list,
        timesteps=args.num_steps,
    )
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
        # python src/main_csi_get_subspace.py --policy_path output/training/ongoing/CustomMyoHandPenTwirlRandom_csi_ppo_seed_0_20250918125732/rl_model_5000000_steps.zip --vecnorm_path output/training/ongoing/CustomMyoHandPenTwirlRandom_csi_ppo_seed_0_20250918125732/rl_model_vecnormalize_5000000_steps.pkl --env_name CustomMyoHandPenTwirlRandom --num_envs 4 --num_steps 100000 --save_path output/pen_csi
        # else :
        #     trainer.agent.policy.change_projection(torch.eye(trainer.agent.policy.action_space.shape[0]), torch.zeros(trainer.agent.policy.action_space.shape[0]), trainable=False)
    # if args.load_csi_subspace is not None:
    #     csi_projection = np.load(args.load_csi_subspace)
    #     csi_projection = torch.from_numpy(csi_projection)
    #     csi_projection = csi_projection[:args.csi_subspace]
        
    #     csi_agent_policy = CSI_wrapper(trainer.agent.policy, csi_projection)
    #     trainer.agent.policy = csi_agent_policy
    #     # Add projection matrix to optimizer

    # Train agent
    trainer.train()
    trainer.save()


    """
runai submit\
    --name arnold-133 \
    --image registry.rcp.epfl.ch/arnold/bc\
    --run-as-uid 174516 \
    --run-as-gid 79678\
    --gpu 0.2\
    --cpu 64 --memory 64Gi --cpu-limit 64 --memory-limit 128Gi\
    --existing-pvc claimname=upamathis-scratch,path=/users\
    --environment WANDB_API_KEY=98ef4d616aed16f83a910072a0b36f39d6a720d5 \
    --environment WANDB_ENTITY=albertochiappa\
    --backoff-limit 0\
    --command -- /bin/bash -ic " \
        cd /users/alberto/arnold; \
        python src/main_ppo.py \
            --seed 0 \
            --log_std_init 0.0 \
            --env_name HandReachRandom \
            --num_envs 64 \
            --network lattice_lstm \
            --rollout_steps 100 \
            --device cuda \
            --num_steps 50000000 \
            --save_freq 100000 \
            --log_interval 16 \
            --out_prefix "_133" \
            --out_suffix "" \
            "
            
        python src/main_ppo.py \
            --seed 0 \
            --log_std_init 0.0 \
            --env_name HandPoseRandom \
            --num_envs 1 \
            --network lattice_lstm \
            --rollout_steps 100 \
            --device cuda \
            --num_steps 50000000 \
            --save_freq 100000 \
            --log_interval 16 \
            --out_prefix "_133" \
            --out_suffix "" \
            --local
    """
