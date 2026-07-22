import os
import argparse
import json
import datetime
import wandb
import pickle
from torch import nn
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from envs.loaders import load_args, load_vocabulary
from models.ppo.policies import MuscleTransformerPolicy
from definitions import ROOT_DIR, ENV_INFO, ENV_CONFIG_PATH
from envs.environment_factory import ENV_NAME_TO_ID
from envs.environment_factory import EnvironmentFactory
from envs.utilities import create_vec_env, get_model_env_vocabulary_path
from train.finetune import FinetuneTrainer
from metrics.custom_callbacks import TensorboardCallback
from imitation.util import logger as imit_logger
from envs.utilities import load_env_and_expert_policy
from utilities import merge_strings


class MyCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(
                checkpoint_type="policy", extension="zip"
            )
            self.model.policy.save(model_path)
            if self.verbose >= 2:
                print(f"Saving policy checkpoint to {model_path}")

            model_path = self._checkpoint_path(checkpoint_type="algo", extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving algo checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path(
                    "replay_buffer_", extension="pkl"
                )
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(
                        f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                    )

            if (
                self.save_vecnormalize
                and self.model.get_vec_normalize_env() is not None
            ):
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path(
                    "vecnormalize_", extension="pkl"
                )
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Finetune Transformer Policy",
        description="Finetune a transformer policy on a new task",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="finetune",
        help="Project name for wandb",
    )
    parser.add_argument(
        "--policy_name", type=str, help="The path of the policy to load and finetune"
    )
    parser.add_argument(
        "--experiment_root",
        type=str,
        default=ROOT_DIR,
        help="The root dir to find policy, vocabulary, etc",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=os.path.join(ROOT_DIR, "output"),
        help="Default root dir to save models",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=os.path.join(ROOT_DIR, "output"),
        help="Default root dir to save logs",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="What algorithm to use for fine-tune",
        choices=["ppo", "separated_value_ppo"],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["relocate"],
        help="The tasks to finetune the policy",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of instances per type of env"
    )
    parser.add_argument(
        "--n_steps", type=int, default=1024, help="Number of steps for each rollout"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log every N number of steps"
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=102400,
        help="Number of total timesteps to learn",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Seed of this run")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to do the finetuning"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=102400,
        help="Save a checkpoint every N number of steps",
    )
    parser.add_argument(
        "--local", action="store_true", help="Local experiments disable wandb"
    )

    args = parser.parse_args()

    # policy = MuscleTransformerPolicy.load(policy_path)

    env_config_list = []
    for task_name in args.tasks:
        eval_env_config_path = os.path.join(ENV_CONFIG_PATH, f"{task_name}_config.json")
        with open(eval_env_config_path, "r") as f:
            eval_env_config = json.load(f)
        env_config_list.append(eval_env_config)

    task_names_str = merge_strings(args.tasks)
    now = datetime.datetime.now()
    current_time = now.strftime(r"%Y%m%d%H%M%S")
    run_name = f"finetune_{task_names_str}_{current_time}"
    save_path = os.path.join(args.save_root, "models", run_name)
    TENSORBOARD_LOG = os.path.join(args.log_root, "training", run_name)

    old_vocabulary = load_vocabulary(args.experiment_root)
    pretrain_args = load_args(args.experiment_root)

    # save vocabularies
    os.makedirs(save_path)
    with open(os.path.join(save_path, "vocabulary.json"), "w") as file:
        json.dump(
            old_vocabulary, file, indent=4, default=lambda _: "<not serializable>"
        )
    with open(os.path.join(save_path, "args.json"), "w") as file:
        json.dump(pretrain_args, file, indent=4, default=lambda _: "<not serializable>")
    with open(os.path.join(save_path, "finetune_args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    # envs = EnvironmentFactory.create(**eval_env_config)
    # envs.mujoco_render_frames = True
    envs, expert_policy_list, train_env_list, vecnormalize_list = (
        load_env_and_expert_policy(
            args.tasks,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            normalize_reward=True,
        )
    )
    # envs = create_vec_env(
    #     env_config_list=env_config_list,
    #     num_envs_per_config=args.num_envs,
    #     load_env_path=None,
    #     tensorboard_log=TENSORBOARD_LOG,
    #     old_vocabulary=old_vocabulary,
    #     norm_reward=False,
    #     seed=args.seed,
    #     multi_env=True
    # )

    # policy.observation_space = envs.observation_space

    policy_args = {
        # "observation_space": envs.observation_space,
        # "action_space": envs.action_space,
        "features_extractor_kwargs": {
            "num_layers": 0,
            "num_heads": 0,
            "embedding_size": pretrain_args["embedding_size"],
            "layer_norm_eps": 1e-5,
            "dim_feedforward": pretrain_args["dim_feedforward"],
            "dropout": 0,
            "position_embedding": "learned",
            "norm_first": True,
        },
        # "lr_schedule": lambda x: pretrain_args["learning_rate"],
        "use_lattice": False,
        "use_expln": True,
        "ortho_init": False,
        "log_std_init": pretrain_args["log_std_init"],
        "activation_fn": nn.ReLU,
        "net_arch": {
            "num_encoder_layers": pretrain_args["num_layers"],
            "num_decoder_layers": pretrain_args["num_layers"],
            "num_heads": pretrain_args["num_heads"],
            "layer_norm_eps": 1e-5,
            "dim_feedforward": pretrain_args["dim_feedforward"],
            "dropout": 0,
            "norm_first": True,
        },
        "policy_outputs_variance": pretrain_args["policy_outputs_variance"],
        "expert_policy_list": expert_policy_list,
        "train_env_list": train_env_list,
        "vecnormalize_list": vecnormalize_list,
        # "device": args.device,
    }

    policy_path = os.path.join(args.experiment_root, args.policy_name)

    save_freq = max(args.save_every // (args.num_envs * len(args.tasks)), 1)
    checkpoint_callback = MyCheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        save_vecnormalize=True,
        verbose=2,
    )
    # info_key_list = []
    # for env_config in env_config_list :
    #     for el in ENV_INFO[env_config["env_name"]] :
    #         import ipdb
    #         ipdb.set_trace()
    #         # info_key_list.append(f"{ENV_NAME_TO_ID[env_config["env_name"]]}/{el}")
    info_key_set = set(
        [
            ENV_NAME_TO_ID[env_config["env_name"]] + f"/{el}"
            for env_config in env_config_list
            for el in ENV_INFO[env_config["env_name"]]
        ]
    )
    tensorboard_callback = TensorboardCallback(info_keywords=info_key_set)
    if args.local:
        callbacks = [checkpoint_callback, tensorboard_callback]
    else:
        run = wandb.init(
            project=args.project_name,
            name=run_name,
            config=policy_args,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb_callback = WandbCallback(
            model_save_path=f"{TENSORBOARD_LOG}/{run.id}",
            gradient_save_freq=100,
            log="all",
        )
        callbacks = [checkpoint_callback, wandb_callback]

    trainer = FinetuneTrainer(
        algo=args.algo,
        policy_path=policy_path,
        envs=envs,
        model_config=policy_args,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        vf_coef=args.vf_coef,
        batch_size=args.batch_size,
        env_config_list=env_config_list,
        log_interval=args.log_interval,
        custom_logger=imit_logger.configure(
            folder=TENSORBOARD_LOG,
            format_strs=(
                ["stdout", "log", "csv", "tensorboard"]
                if args.local
                else ["stdout", "log", "csv", "wandb"]
            ),
        ),
        device=args.device,
        callbacks=callbacks,
        old_vocabulary=old_vocabulary,
    )
    # print(policy)
    # print(envs.observation_space)
    # print(envs)

    trainer.train(args.total_steps)
