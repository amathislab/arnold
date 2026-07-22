import os
import glob
import json
from sys import prefix
from envs.environment_factory import EnvironmentFactory
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.loaders import load_expert_policy, load_vocabulary
from envs.multi_env import (
    MultiSubprocVecEnv,
    FlexibleMultiVecNormalize,
    RewardNormalizer,
    MyVecNormalize,
    EnvIDWrapper
)
from utilities import get_number, get_remote_checkpoint
from definitions import (
    ENV_CONFIG_PATH,
    ENV_FILE_NAME,
    EXPERT_PER_CONFIG_PATH,
    EXPERT_POLICIES_PATH,
    MODEL_CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    MODEL_PATTERN,
    ENV_PATTERN,
    VOCABULARY_FILE_NAME,
)
from envs.expert_wrapper import ExpertWrapper


def get_last_checkpoint(path):
    model_list = sorted(
        glob.glob(os.path.join(path, MODEL_PATTERN)),
        key=get_number,
    )
    checkpoints_list = [get_number(el) for el in model_list]
    if len(checkpoints_list) > 0:
        return max(checkpoints_list)
    else:
        return None


def get_model_env_vocabulary_path(tensorboard_log, load_path, checkpoint_num):
    """This function is used to robustly recover the checkpoint when the training is interrupted.
    When tensorboard_log already exists, the function looks for the latest checkpoint in such
    folder. This is done so that an automatic restart of the training resumes from the last
    available checkpoint, ignoring load_path and checkpoint_num. If tensorboard_log does not
    exist or has no checkpoint, the training starts either from the specified checkpoint_num, or,
    if no checkpoint_num is specified, from the last checkpoint of load_path. If load_path has
    no checkpoint, the training starts from scratch

    :param tensorboard_log: path of the Tensorboard log directory
    :param load_path: path of the directory of the experiment we want to resume
    :param checkpoint_num: number of the checkpoint to load
    :return: model_path and env_path
    """
    if os.path.isdir(tensorboard_log):
        # The folder already exists, then we resume the training if there are already checkpoints
        checkpoint_num = get_last_checkpoint(tensorboard_log)
        if checkpoint_num is None:
            print(
                f"WARNING: A training at {tensorboard_log} already exists, but no checkpoint was found. "
                f"Searching for a checkpoint at {load_path}."
            )
        else:
            load_path = tensorboard_log
            if load_path is not None:
                print(
                    f"WARNING: A checkpoint was found at {tensorboard_log}, so we are resuming from there."
                )

    if load_path is not None:
        if checkpoint_num is None:
            checkpoint_num = get_last_checkpoint(load_path)
        if checkpoint_num is None:
            print(
                f"WARNING: No checkpoints at the given path {load_path}, starting a new training"
            )
        else:
            model_path = os.path.join(
                load_path, MODEL_PATTERN.replace("*", str(checkpoint_num))
            )
            env_path = os.path.join(
                load_path, ENV_PATTERN.replace("*", str(checkpoint_num))
            )
            vocabulary_path = os.path.join(load_path, VOCABULARY_FILE_NAME)
            print(f"Loading model from {model_path} and environment from {env_path}")
    else:
        model_path = None
        env_path = None
        vocabulary_path = None
    return model_path, env_path, vocabulary_path


class ExpertMonitor(Monitor):
    @property
    def id(self):
        return self.unwrapped.env.id

    def get_obs_ids(self):
        return self.unwrapped.env.get_obs_ids()


def make_parallel_envs(
    env_config_list,
    num_envs_per_config,
    tensorboard_log,
    seed,
    multi_env=True,
    normalize_reward=False,
    reward_normalizer_list=None,
    expert_task_list=(None,),
    expert_device="cuda",
    custom_expert_config_path=None,
):
    if reward_normalizer_list is None:
        reward_normalizer_list = [None] * len(env_config_list)

    def make_env(env_id, config_id, env_config, reward_normalizer, expert_task):
        def _thunk():
            env_config["seed"] = seed + env_id
            env = EnvironmentFactory.create(**env_config)
            if expert_task is not None:
                # if expert_task == "kinesis":
                #     env = KinesisExpertWrapper(env)
                # else:
                env = ExpertWrapper(env, expert_task, device=expert_device, custom_expert_config_path=custom_expert_config_path)
            if normalize_reward:
                env = RewardNormalizer(env, reward_normalizer, config_id)
            env = ExpertMonitor(env, tensorboard_log)
            if multi_env :
                env = EnvIDWrapper(env, config_id)
            return env

        return _thunk

    if multi_env:
        env_fn_list = []
        if expert_task_list[0] is None:
            expert_task_list = [None] * len(env_config_list)
        for config_id, (env_config, reward_normalizer, expert_task) in enumerate(
            zip(env_config_list, reward_normalizer_list, expert_task_list)
        ):
            for env_id in range(num_envs_per_config):
                env_fn_list.append(
                    make_env(
                        env_id,
                        config_id,
                        env_config,
                        reward_normalizer,
                        expert_task,
                    )
                )
        return MultiSubprocVecEnv(env_fns=env_fn_list)
    else:
        assert len(env_config_list) == 1
        return SubprocVecEnv(
            [
                make_env(
                    i,
                    0,
                    env_config_list[0],
                    reward_normalizer_list[0],
                    expert_task_list[0],
                )
                for i in range(num_envs_per_config)
            ]
        )


def create_vec_env(
    env_config_list,
    num_envs_per_config=1,
    load_env_path=None,
    tensorboard_log=None,
    multi_env=True,
    old_vocabulary=None,
    norm_reward=True,
    norm_obs=True,
    seed=0,
    expert_task_list=(None,),
    expert_device="cuda",
    custom_expert_config_path=None,
):
    envs = make_parallel_envs(
        env_config_list=env_config_list,
        num_envs_per_config=num_envs_per_config,
        tensorboard_log=tensorboard_log,
        seed=seed,
        multi_env=multi_env,
        expert_task_list=expert_task_list,
        expert_device=expert_device,
        custom_expert_config_path=custom_expert_config_path
    )
    if load_env_path is None or not os.path.exists(load_env_path):
        print(
            f"Creating a new environment for {[config['env_name'] for config in env_config_list]}"
        )
        if multi_env:
            envs = FlexibleMultiVecNormalize(envs, norm_reward=norm_reward, norm_obs=norm_obs)  # Add norm_obs
        else:
            envs = MyVecNormalize(envs, norm_obs=norm_obs)  # Add norm_obs
    else:
        if multi_env:
            envs = FlexibleMultiVecNormalize.load(
                load_env_path, envs, vocabulary=old_vocabulary
            )
            print(
                f"Warning: ignoring norm_reward and norm_obs setup {norm_reward}, {norm_obs}. "
                f"The loaded environment normalizations are set to {envs.norm_reward}, {envs.norm_obs}"
            )
        else:
            envs = MyVecNormalize.load(load_env_path, envs)
    return envs


def load_env_and_expert_policy(
    imitation_task_list,
    num_envs=1,
    seed=0,
    device="cuda",
    normalize_reward=False,
    use_arnold=True
):
    with open(EXPERT_PER_CONFIG_PATH, "r") as f:
        expert_config_dict = json.load(f)

    policy_list = []
    eval_env_config_list = []
    train_env_list = []
    vecnormalize_list = []
    vecnormalize_path_list = []
    expert_config_list = []
    vocabulary_list = []

    for imitation_task in imitation_task_list:
        if use_arnold :
            prefix = "arnold_"
        eval_env_config_path = os.path.join(
            ENV_CONFIG_PATH, f"{prefix}{imitation_task}_config.json"
        )
        with open(eval_env_config_path, "r") as f:
            eval_env_config = json.load(f)
        single_eval_env = EnvironmentFactory.create(**eval_env_config)
        eval_env_config_list.append(eval_env_config)
        if imitation_task in expert_config_dict:
            expert_config = expert_config_dict[imitation_task]
            # if expert_config["use_arnold"]:
            #     train_env = single_eval_env  # The params of the env define the number of time steps, which cannot change
            # else:
            train_env = EnvironmentFactory.create(expert_config["train_env"])

            expert_path = os.path.join(EXPERT_POLICIES_PATH, expert_config["expert"])
            model_path = os.path.join(expert_path, MODEL_FILE_NAME)
            print(f"Loading policy from {model_path}")
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
                "observation_space": train_env.observation_space,
                "action_space": single_eval_env.action_space,
                "vocabulary": vocabulary,
            }
            expert_policy = load_expert_policy(
                algo=expert_config["algo"],
                model_path=model_path,
                custom_objects=custom_objects,
                train_model_config=train_model_config,
                device=device,
            )
            vecnormalize_path = os.path.join(expert_path, ENV_FILE_NAME)
        else:
            vecnormalize_path = None
            expert_policy = None
            expert_config = None
            vocabulary = None
            train_env = None

        vecnormalize_path_list.append(vecnormalize_path)
        expert_config_list.append(expert_config)
        vocabulary_list.append(vocabulary)

        # vecnormalize = create_vec_env(
        #     env_config_list=[eval_env_config],
        #     load_env_path=vecnormalize_path,
        #     multi_env=expert_config["use_arnold"],
        #     old_vocabulary=vocabulary,
        # )
        policy_list.append(expert_policy)
        train_env_list.append(train_env)
        # vecnormalize_list.append(vecnormalize)

    for eval_env_config, vecnormalize_path, expert_config, vocabulary in zip(
        eval_env_config_list,
        vecnormalize_path_list,
        expert_config_list,
        vocabulary_list,
    ):
        vecnormalize = create_vec_env(
            env_config_list=(
                eval_env_config_list
                if expert_config and expert_config["use_arnold"]
                else [eval_env_config]
            ),
            load_env_path=vecnormalize_path,
            multi_env=False,
            old_vocabulary=vocabulary,
        )
        vecnormalize_list.append(vecnormalize)

    eval_env = make_parallel_envs(
        env_config_list=eval_env_config_list,
        num_envs_per_config=num_envs,
        tensorboard_log=None,
        seed=seed,
        multi_env=use_arnold,
        normalize_reward=normalize_reward,
        reward_normalizer_list=vecnormalize_list,
    )
    return eval_env, policy_list, train_env_list, vecnormalize_list


def load_vecnormalize(
    experiment_path,
    checkpoint_number,
    env_config_list,
    vocabulary,
    single_env=False,
    host=None,
    host_project_root=None,
):
    env_file = ENV_PATTERN.replace("*", str(checkpoint_number))
    env_path = os.path.join(experiment_path, env_file)
    if (
        not os.path.exists(env_path)
        and host is not None
        and host_project_root is not None
    ):
        get_remote_checkpoint(
            host, host_project_root, experiment_path, checkpoint_number
        )
    vecnormalize = create_vec_env(
        env_config_list,
        load_env_path=env_path,
        multi_env=not single_env,
        old_vocabulary=vocabulary,
    )
    return vecnormalize
