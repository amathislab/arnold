import numpy as np
import imitation
import torch.nn as nn
import torch
import wandb
import tempfile
import gymnasium as gym
import imitation.util
from algos.bc_pure import BCPure
from algos.bc_value import BCValue
from algos.bc_predict import BCPredict
from algos.bc_bilateral import BCBilateral
from algos.dagger_value import MultiEnvDAggerTrainer
from algos.dagger_bilateral import MultiEnvBilateralDAggerTrainer
from envs.utilities import load_env_and_expert_policy
from metrics.custom_loggers import WandbBCLogger
from utilities import merge_strings
from torch_utils import my_safe_to_tensor
from definitions import (
    ROOT_DIR,
)

imitation.util.util.safe_to_tensor = my_safe_to_tensor  # Override a buggy func
from imitation.util import logger as imit_logger
from models.ppo.policies import (
    BilateralMuscleTransformerPolicy,
    MuscleTransformerPolicy,
    PredictiveMuscleTransformerPolicy,
)
from gymnasium import spaces
import datetime
import pickle
import argparse
import json
import os
import shutil
from dataloaders.multi_dataloader import MultiEnvDataLoader
from metrics.custom_loggers import WandbBCLogger
from models.ppo.policies import MuscleTransformerPolicy
from vocabulary import VOCABULARY


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Dagger",
        description="Clone the behavior of an expert policy for muscle control",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--name", help="Name of the experiment", default="test")
    parser.add_argument(
        "--algo",
        type=str,
        default="bc_value",
        help="Algorithm to use, bc_value or bc_predict",
        choices=["bc_value", "bc_predict", "bc_bilateral", "bc"],
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load already-trained model",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="transformer",
        help="Policy network to use, transformer or predictive_transformer",
        choices=["transformer", "predictive_transformer", "bilateral_transformer"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Path to the dataset to use for training",
    )
    parser.add_argument(
        "--imitation_tasks",
        nargs="+",
        help="Tasks to imitate",
        default=["hand_index_reach"],
    )
    parser.add_argument(
        "--dataroot",
        help="Root directory of the project",
        default=os.path.join(ROOT_DIR, "data/datasets"),
    )
    parser.add_argument(
        "--saveroot",
        help="Root directory for saving models",
        default=os.path.join(ROOT_DIR, "output"),
    )
    parser.add_argument(
        "--logroot",
        help="Root directory for logging",
        default=os.path.join(ROOT_DIR, "logs"),
    )
    parser.add_argument(
        "--traj_per_dataset",
        type=int,
        default=100000,
        help="Number of trajectories per dataset",
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
        "--k_bilateral",
        type=int,
        default=256,
        help="Number of time steps to consider in the bilateral attention",
    )
    parser.add_argument(
        "--time_skip",
        type=int,
        default=6,
        help="Number of time steps between two input tokens",
    )
    parser.add_argument(
        "--log_std_init",
        type=float,
        default=-0.3,
        help="Initial log standard deviation",
    )
    parser.add_argument(
        "--policy_outputs_variance",
        action="store_true",
        default=False,
        help="Flag to use SDE",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Learning device, cuda or cpu",
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
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--action_loss_mode",
        type=str,
        default="neg_log_p",
        help="Action loss mode",
        choices=["neg_log_p", "l2"],
    )
    parser.add_argument(
        "--ent_weight",
        type=float,
        default=1e-6,
        help="Scaling applied to the entropy regularization",
    )
    parser.add_argument(
        "--l2_weight",
        type=float,
        default=0.0,
        help="Scaling applied to the L2 regularization",
    )
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.5,
        help="Masking observations and actions with a specific rate for bilateral transformer",
    )
    parser.add_argument(
        "--deterministic_expert",
        default=False,
        action="store_true",
        help="Masking observations and actions with a specific rate for bilateral transformer",
    )
    parser.add_argument(
        "--value_weight",
        type=float,
        default=0.5,
        help="Scaling applied to the value function learning",
    )
    parser.add_argument(
        "--obs_pred_weight",
        type=float,
        default=0.5,
        help="Loss weight for next observation prediction",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Epochs to train the model for",
    )
    parser.add_argument(
        "--beta_rounds",
        type=int,
        default=15,
        help="How many rounds of dagger before beta decreases linearly to 0",
    )
    parser.add_argument(
        "--rollout_reuse_epoch",
        type=int,
        default=4,
        help="How many epochs to use each rollout in DAgger",
    )
    parser.add_argument(
        "--rollout_round_min_timesteps",
        type=int,
        default=2048,
        help="How many steps to collect per rollout",
    )
    parser.add_argument(
        "--max_rollout_storage",
        type=int,
        default=1024000,
        help="How many steps to collect per rollout",
    )
    parser.add_argument(
        "--omit_history_rollout",
        action="store_true",
        help="Whether to throw away all history data, to do online training",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=256,
        help="Batches between two logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save the model every n epochs",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="How many envs for each environment",
    )
    parser.add_argument(
        "--no_action_clipping",
        action="store_true",
        help="Whether not to clip expert actions to [-1,1]",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run experiments locally or on server, only have effects on outputting and logging",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(0)

    task_names_str = merge_strings(args.imitation_tasks)

    now = datetime.datetime.now()
    current_time = now.strftime(r"%Y%m%d%H%M%S")
    run_name = f"{args.name}_{task_names_str}__seed_{args.seed}_nl_{args.num_layers}_nh_{args.num_heads}_es_{args.embedding_size}_df_{args.dim_feedforward}_{current_time}"

    save_path = os.path.join(args.saveroot, run_name)
    os.makedirs(save_path)
    TENSORBOARD_LOG = os.path.join(args.logroot, "training", run_name)

    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)
    with open(os.path.join(TENSORBOARD_LOG, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")
    with open(os.path.join(save_path, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    # save vocabularies
    with open(os.path.join(TENSORBOARD_LOG, "vocabulary.json"), "w") as file:
        json.dump(VOCABULARY, file, indent=4, default=lambda _: "<not serializable>")
    with open(os.path.join(save_path, "vocabulary.json"), "w") as file:
        json.dump(VOCABULARY, file, indent=4, default=lambda _: "<not serializable>")

    # prepare model saving directories
    if not os.path.exists(os.path.join(args.saveroot)):
        os.makedirs(os.path.join(args.saveroot))

    # if args.datasets is not None:
    #     print("Loading trajectories")
    #     data_paths = [os.path.join(args.dataroot, dataset) for dataset in args.datasets]
    #     dataloader = MultiEnvDataLoader(
    #         paths=data_paths,
    #         traj_per_dataset=args.traj_per_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=True,
    #         stochastic_df=True,
    #     )
    #     obs_space = dataloader.obs_space
    #     act_space = dataloader.act_space
    #     print("Trajectories loaded")
    # else:
    #     assert args.load is not None
    #     with open(
    #         os.path.join(os.path.dirname(args.load), "observation_space.pkl"), "rb"
    #     ) as file:
    #         obs_space = pickle.load(file)
    #     with open(
    #         os.path.join(os.path.dirname(args.load), "action_space.pkl"), "rb"
    #     ) as file:
    #         act_space = pickle.load(file)
    #     print("Spaces loaded")
    #     # Load saved spaces

    # # save spaces
    # with open(os.path.join(save_path, "observation_space.pkl"), "wb") as file:
    #     pickle.dump(obs_space, file)
    # with open(os.path.join(save_path, "action_space.pkl"), "wb") as file:
    #     pickle.dump(act_space, file)

    # all_context = {
    #     "observation_space": obs_space,
    #     "action_space": act_space,
    #     "args": args.__dict__,
    # }

    # init wandb
    if not args.local:
        # wandb_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        run = wandb.init(
            dir=TENSORBOARD_LOG,
            project=args.name,
            name=run_name,
            # config=all_context,
            sync_tensorboard=True,
            save_code=True,
        )

    eval_env, expert_policy_list, train_env_list, vecnormalize_list = (
        load_env_and_expert_policy(
            args.imitation_tasks,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            use_arnold=True
        )
    )

    obs_space = eval_env.observation_space
    act_space = eval_env.action_space

    # Avoid action clipping
    if args.no_action_clipping:
        act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=act_space.shape)
        for expert_policy in expert_policy_list:
            expert_policy.action_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=expert_policy.action_space.shape
            )
            # TODO: check that it's not necessary to change the action space of the policy
            # expert_policy.policy.action_space = expert_policy.action_space
            # eval_env.action_space = spaces.Box(
            #     low=-np.inf, high=np.inf, shape=eval_env.action_space.shape
            # )

    custom_logger = WandbBCLogger(run) if not args.local else None

    if args.policy == "transformer":
        policy_class = MuscleTransformerPolicy
    elif args.policy == "predictive_transformer":
        policy_class = PredictiveMuscleTransformerPolicy
    elif args.policy == "bilateral_transformer":
        policy_class = BilateralMuscleTransformerPolicy
    else:
        raise ValueError("Unknown policy {}".format(args.policy))
    bc_policy = get_policy(obs_space, act_space, policy_class, args)

    extra_kwargs = {}
    if args.algo == "bc_value":
        bc_class = BCValue
    elif args.algo == "bc_predict":
        bc_class = BCPredict
    elif args.algo == "bc":
        bc_class = BCPure
    elif args.algo == "bc_bilateral":
        bc_class = BCBilateral
        extra_kwargs["time_skip"] = args.time_skip
        extra_kwargs["mask_rate"] = args.mask_rate
    else:
        raise ValueError("Unknown algorithm")

    if args.load is not None:
        bc_policy.load(args.load)
        print(f"Policy loaded from {args.load}")

    bc_trainer = bc_class(
        observation_space=obs_space,
        action_space=act_space,
        rng=np.random.default_rng(args.seed),
        policy=bc_policy,
        # demonstrations=dataloader,
        device=args.device,
        batch_size=args.batch_size,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=dict(lr=args.learning_rate),
        action_loss_mode=args.action_loss_mode,
        ent_weight=args.ent_weight,
        l2_weight=args.l2_weight,
        value_weight=args.value_weight,
        obs_pred_weight=args.obs_pred_weight,
        custom_logger=custom_logger,
        **extra_kwargs,
    )

    num_epoch = 0

    def save_policy_callback():
        from imitation.data import types
        from imitation.util import util

        global num_epoch, bc_trainer, args
        # save policy
        num_epoch += 1
        # if args.local:
        #     err_list = []
        #     for batch in dataloader:
        #         obs_tensor = types.map_maybe_dict(
        #             lambda x: util.safe_to_tensor(x).to(bc_policy.device),
        #             types.maybe_unwrap_dictobs(batch["obs"]),
        #         )
        #         acts = util.safe_to_tensor(batch["acts"]).to(bc_policy.device)
        #         acts_pred, _, _ = bc_policy(obs_tensor, deterministic=True)
        #         err_list.append((acts - acts_pred).cpu().abs().mean().item())
        #     print("\n=====================")
        #     print("Relative error:", np.mean(err_list))

        if num_epoch % args.save_every == 0:
            policy_path = os.path.join(save_path, f"policy_epoch{num_epoch}")
            bc_trainer.policy.save(policy_path + ".zip")
            with open(policy_path + ".pkl", "wb") as f:
                pickle.dump(bc_trainer.policy, f)
            print(f"Policy saved at {policy_path}")

    with tempfile.TemporaryDirectory(prefix="dagger") as tempdir:

        print("training dagger in temp dir", tempdir)
        extra_kwargs = {}
        if args.algo == "bc_bilateral":
            dagger_cls = MultiEnvBilateralDAggerTrainer
            extra_kwargs["time_skip"] = args.time_skip
        else:
            dagger_cls = MultiEnvDAggerTrainer
        dagger_trainer = dagger_cls(
            venv=eval_env,
            env_names=args.imitation_tasks,
            train_env_list=train_env_list,
            scratch_dir=tempdir,
            expert_policy_list=expert_policy_list,
            deterministic_expert=args.deterministic_expert,
            bc_trainer=bc_trainer,
            vec_normalize_list=vecnormalize_list,
            rng=np.random.default_rng(args.seed),
            # expert_trajs=dataloader,
            beta_schedule=args.beta_rounds,
            omit_history_rollout=args.omit_history_rollout,
            # custom_logger=WandbBCLogger(run) if not args.local else None,
            custom_logger=imit_logger.configure(
                folder=TENSORBOARD_LOG,
                format_strs=(
                    ["stdout", "log", "csv", "tensorboard"]
                    if args.local
                    else ["stdout", "log", "csv", "wandb", "tensorboard"]
                ),
            ),
            **extra_kwargs,
        )
        print("Training DAgger")
        dagger_trainer.train(
            total_round_num=args.n_epochs,
            max_rollout_storage=args.max_rollout_storage,
            rollout_round_min_episodes=1,
            rollout_round_min_timesteps=args.rollout_round_min_timesteps,
            bc_train_kwargs={
                "on_epoch_end": save_policy_callback,
                "n_epochs": args.rollout_reuse_epoch,
                "log_interval": args.log_interval,
            },
        )
        print("Training finished successfully")

    if not args.local:
        wandb.finish()


"""
python src/main_bc.py --dataset baoding_p1_cw_no_clip_100k hand_index_reach_no_clip_100k \
    --traj_per_dataset=5 --batch_size=10 --n_epochs=1000 --save_every=1000 --learning_rate=1e-4

python src/main_dagger.py --env_name MuscleHandIndexReachRandom MuscleBaodingP1CW \
--dataset hand_index_reach_no_clip_100k baoding_p1_cw_no_clip_100k \
    --expert_policy_path data/expert_policies/hand_reach_single_finger \
        data/expert_policies/baoding_phase_1 \
            --rollout_round_min_timesteps=20 \
                --train_env_name MuscleHandIndexReachRandom CleanBaodingBalls

python src/main_dagger.py \
--datasets hand_index_reach_no_clip_100k baoding_p1_cw_no_clip_100k \
    --imitation_tasks baoding_p1_ccw hand_index_reach \
            --rollout_round_min_timesteps=2048 --n_epochs=1000 --device cpu \
                --traj_per_dataset 100000

"""
