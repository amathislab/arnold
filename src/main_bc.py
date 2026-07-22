import numpy as np
import imitation
import torch.nn as nn
import torch
import wandb
import tempfile
import glob
import gymnasium as gym
import imitation.util
from imitation.algorithms.bc import BC
from envs.loaders import load_vocabulary
from metrics.custom_loggers import WandbBCLogger
from utilities import merge_strings
from definitions import ROOT_DIR
from torch_utils import my_safe_to_tensor

imitation.util.util.safe_to_tensor = my_safe_to_tensor  # Override a buggy func
from models.ppo.policies import MuscleTransformerPolicy
from vocabulary import VOCABULARY
from gymnasium import spaces
import datetime
import pickle
import pandas as pd
import argparse
import json
import os
import shutil
from algos.dagger_value import MultiEnvDAggerTrainer
from dataloaders.multi_dataloader import MultiEnvDataLoader
from metrics.custom_loggers import WandbBCLogger
from models.ppo.policies import MuscleTransformerPolicy
from models.ppo.policies import LatticeRecurrentActorCriticPolicy
from main_eval_arnold import get_number
from envs.utilities import load_vecnormalize, make_parallel_envs
from envs.environment_factory import EnvironmentFactory
from train.algo_factory import AlgoFactory
from vocabulary import VOCABULARY
from definitions import MODEL_PATTERN


def get_policy(observation_space, action_space, args):

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

    policy = MuscleTransformerPolicy(
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


def load_expert_policy(
    algo, experiment_path, checkpoint_number, custom_objects=None, custom_config=None
):
    if custom_objects is None:
        custom_objects = {}
    if custom_config is not None:
        policy_kwargs = custom_config.get("policy_kwargs")
        if policy_kwargs is not None:
            if policy_kwargs.get("use_lattice"):
                policy_class = LatticeRecurrentActorCriticPolicy
                custom_objects["policy_class"] = policy_class
            activation_fn = policy_kwargs.get("activation_fn")
            if activation_fn is not None:
                policy_kwargs["activation_fn"] = getattr(nn, activation_fn)
            custom_objects["policy_kwargs"] = policy_kwargs
    model_file = MODEL_PATTERN.replace("*", str(checkpoint_number))
    model_path = os.path.join(experiment_path, model_file)
    model = AlgoFactory.get_algo_class(algo).load(
        model_path, custom_objects=custom_objects
    )
    return model.policy


def load_expert_policy_and_env(
    experiment_path, env_name=None, use_vec_env=False, num_envs=1, seed=0
):
    model_list = sorted(
        glob.glob(os.path.join(experiment_path, MODEL_PATTERN)),
        key=get_number,
    )
    checkpoints = [get_number(el) for el in model_list]
    if len(checkpoints) >= 1:
        checkpoint = checkpoints[-1]
    else:
        checkpoint = None
        raise FileNotFoundError("Checkpoint not found in directory", experiment_path)

    try:
        print("Loading configuration file...")
        with open(os.path.join(experiment_path, "args.json"), "r") as f:
            loaded_args = json.load(f)
        print("Done")
    except Exception as e:
        loaded_args = None
        print("Configuration not loaded:", e)

    try:
        print("Loading vocabulary...")
        vocabulary = load_vocabulary(experiment_path, checkpoint)
        print("Done")
    except Exception as e:
        vocabulary = None
        print("Vocabulary not loaded:", e)

    try:
        print("Loading environment config of trained policy...")
        eval_env_config_path = os.path.join(experiment_path, f"{env_name}_config.json")
        eval_env_config = json.load(open(eval_env_config_path, "r"))
        print("Done")
    except Exception as e:
        print("Environment config loading failed:", e)
        eval_env_config = {
            "env_name": env_name,
            "include_adapt_state": True,
            "num_memory_steps": 5,
            "seed": seed,
        }
        print("Using new config:", eval_env_config)

    if use_vec_env:
        eval_env = make_parallel_envs(
            env_config_list=[eval_env_config],
            num_envs_per_config=num_envs,
            tensorboard_log=None,
            seed=seed,
            multi_env=True,
        )
    else:
        if "env_name" in eval_env_config:
            eval_env_config["env_name"] = env_name
            # eval_env_config.pop("env_name")
        eval_env = EnvironmentFactory.create(**eval_env_config)

    try:
        model_file = MODEL_PATTERN.replace("*", str(checkpoint))
        model_path = os.path.join(experiment_path, model_file)
        print(f"Loading policy as MuscleTransformerPolicy from {model_path}")
        expert_policy = MuscleTransformerPolicy.load(model_path)
        vecnormalize = None
        print("Done")
        return expert_policy, eval_env, vecnormalize
    except Exception as e:
        print("Loading policy as MuscleTransformerPolicy failed:", e)

    try:
        print(f"Loading policy as PPO Wrapped Policy from {experiment_path}")
        custom_objects = {
            "observation_space": eval_env.observation_space,
            "action_space": eval_env.action_space,
            "vocabulary": vocabulary,
        }
        expert_policy = load_expert_policy(
            algo=loaded_args["algo"] if loaded_args else "recurrent_ppo",
            experiment_path=experiment_path,
            checkpoint_number=checkpoint,
            custom_objects=custom_objects,
            # custom_config=loaded_args,
        )
        vecnormalize = load_vecnormalize(
            experiment_path=experiment_path,
            checkpoint_number=checkpoint,
            env_config_list=[eval_env_config],
            vocabulary=vocabulary,
            single_env=False,
        )
        vecnormalize.training = False
        print("Done")
        return expert_policy, eval_env, vecnormalize
    except Exception as e:
        print("Loading policy as PPO Wrapped Policy failed:", e)
        return None, None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Behavior Cloning",
        description="Clone the behavior of an expert policy for muscle control",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Path to the dataset to use for training",
        default=["hand_index_reach_no_clip_100k"],
    )
    parser.add_argument(
        "--env_name",
        nargs="+",
        help="Names of environments corresponding to the datasets",
        default=["MuscleHandIndexReachRandom"],
    )
    parser.add_argument(
        "--expert_policy_path",
        nargs="+",
        help="Path to expert policies corresponding to the datasets",
        default=[],
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
    parser.add_argument("--name", help="Name of the experiment", default="test")
    parser.add_argument("--algo", help="Which algorithm to use", default="bc")
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
        "--n_epochs",
        type=int,
        default=100,
        help="Epochs to train the model for",
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
        "--rollout_round_min_timesteps",
        type=int,
        default=20480,
        help="How many steps to collect per rollout",
    )
    parser.add_argument(
        "--rollout_reuse_epoch",
        type=int,
        default=4,
        help="How many epochs to use each rollout in DAgger",
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

    data_paths = [os.path.join(args.dataroot, dataset) for dataset in args.dataset]
    task_names_str = merge_strings(args.dataset)

    now = datetime.datetime.now()
    current_time = now.strftime(r"%Y%m%d%H%M%S")
    run_name = f"{args.name}_{task_names_str}_{args.algo}_seed_{args.seed}_nl_{args.num_layers}_nh_{args.num_heads}_es_{args.embedding_size}_df_{args.dim_feedforward}_{current_time}"

    save_path = os.path.join(args.saveroot, "models", run_name)
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
    if not os.path.exists(os.path.join(args.saveroot, "models")):
        os.makedirs(os.path.join(args.saveroot, "models"))

    print("Loading trajectories")
    dataloader = MultiEnvDataLoader(
        paths=data_paths,
        traj_per_dataset=args.traj_per_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        stochastic_df=True,
    )
    print("Trajectories loaded")

    # save spaces
    obs_space = dataloader.obs_space
    act_space = dataloader.act_space
    with open(os.path.join(save_path, "observation_space.pkl"), "wb") as file:
        pickle.dump(obs_space, file)
    with open(os.path.join(save_path, "action_space.pkl"), "wb") as file:
        pickle.dump(act_space, file)

    all_context = {
        "observation_space": obs_space,
        "action_space": act_space,
        "args": args.__dict__,
    }

    # init wandb
    if not args.local:
        wandb_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        run = wandb.init(
            dir=wandb_dir,
            project=args.name,
            name=run_name,
            config=all_context,
            save_code=True,
        )

    if args.algo == "bc":

        # start training pipeline
        policy = get_policy(obs_space, act_space, args)

        trainer = BC(
            observation_space=obs_space,
            action_space=act_space,
            rng=np.random.default_rng(args.seed),
            policy=policy,
            demonstrations=dataloader,
            device=args.device,
            batch_size=args.batch_size,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=dict(lr=args.learning_rate),
            ent_weight=args.ent_weight,
            l2_weight=args.l2_weight,
            custom_logger=WandbBCLogger(run) if not args.local else None,
        )

        num_epoch = 0

        def save_policy_callback():
            from imitation.data import types
            from imitation.util import util

            global num_epoch, trainer, args
            # save policy
            num_epoch += 1

            if args.local:
                err_list = []
                for batch in dataloader:
                    obs_tensor = types.map_maybe_dict(
                        lambda x: util.safe_to_tensor(x),
                        types.maybe_unwrap_dictobs(batch["obs"]),
                    )
                    acts = util.safe_to_tensor(batch["acts"])
                    acts_pred, _, _ = policy(obs_tensor, deterministic=True)
                    err_list.append((acts - acts_pred).abs().mean().item())
                print("\n=====================")
                print(np.mean(err_list))

            if num_epoch % args.save_every == 0:
                policy_path = os.path.join(save_path, f"policy_epoch{num_epoch}")
                trainer.policy.save(policy_path + ".zip")
                with open(policy_path + ".pkl", "wb") as f:
                    pickle.dump(trainer.policy, f)
                print(f"Policy saved at {policy_path}")

        trainer.train(
            n_epochs=args.n_epochs,
            on_epoch_end=save_policy_callback,
            log_interval=args.log_interval,
        )

    elif args.algo == "dagger":

        assert len(args.dataset) == len(args.env_name)
        assert len(args.dataset) == len(args.expert_policy_path)
        if len(args.env_name) != 1:
            raise ValueError("Currently DAgger can only be run on a single environment")

        expert_policy, eval_env, vecnormalize = load_expert_policy_and_env(
            args.expert_policy_path[0],
            args.env_name[0],
            use_vec_env=True,
            num_envs=args.num_envs,
            seed=args.seed,
        )

        # Avoid action clipping
        if args.no_action_clipping:
            act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=act_space.shape)
            expert_policy.action_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=expert_policy.action_space.shape
            )
            # expert_policy.policy.action_space = expert_policy.action_space
            eval_env.action_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=eval_env.action_space.shape
            )

        bc_policy = get_policy(obs_space, act_space, args)
        bc_trainer = BC(
            observation_space=obs_space,
            action_space=act_space,
            rng=np.random.default_rng(args.seed),
            policy=bc_policy,
            demonstrations=dataloader,
            device=args.device,
            batch_size=args.batch_size,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=dict(lr=args.learning_rate),
            ent_weight=args.ent_weight,
            l2_weight=args.l2_weight,
            # custom_logger=WandbBCLogger(run),
        )

        num_epoch = 0

        def save_policy_callback():
            from imitation.data import types
            from imitation.util import util

            global num_epoch, bc_trainer, args
            # save policy
            num_epoch += 1
            if args.local:
                err_list = []
                for batch in dataloader:
                    obs_tensor = types.map_maybe_dict(
                        lambda x: util.safe_to_tensor(x).to(bc_policy.device),
                        types.maybe_unwrap_dictobs(batch["obs"]),
                    )
                    acts = util.safe_to_tensor(batch["acts"]).to(bc_policy.device)
                    acts_pred, _, _ = bc_policy(obs_tensor, deterministic=True)
                    err_list.append((acts - acts_pred).cpu().abs().mean().item())
                print("\n=====================")
                print(np.mean(err_list))

            if num_epoch % args.save_every == 0:
                policy_path = os.path.join(save_path, f"policy_epoch{num_epoch}")
                bc_trainer.policy.save(policy_path + ".zip")
                with open(policy_path + ".pkl", "wb") as f:
                    pickle.dump(bc_trainer.policy, f)
                print(f"Policy saved at {policy_path}")

        with tempfile.TemporaryDirectory(prefix="dagger") as tempdir:

            print("training dagger in temp dir", tempdir)
            dagger_trainer = MyDAggerTrainer(
                venv=eval_env,
                scratch_dir=tempdir,
                expert_policy=expert_policy,
                bc_trainer=bc_trainer,
                vec_normalize=vecnormalize,
                rng=np.random.default_rng(args.seed),
                expert_trajs=dataloader,
                custom_logger=WandbBCLogger(run) if not args.local else None,
            )
            dagger_trainer.train(
                total_round_num=args.n_epochs,
                rollout_round_min_episodes=1,
                rollout_round_min_timesteps=args.rollout_round_min_timesteps,
                bc_train_kwargs={
                    "on_epoch_end": save_policy_callback,
                    "n_epochs": args.rollout_reuse_epoch,
                },
            )

    if not args.local:
        wandb.finish()


"""
python src/main_bc.py --dataset baoding_p1_cw_no_clip_100k hand_index_reach_no_clip_100k \
    --traj_per_dataset=5 --batch_size=10 --n_epochs=1000 --save_every=1000 --learning_rate=1e-4

"""
