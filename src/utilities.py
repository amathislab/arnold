import os
from unittest.mock import patch
import numpy as np
import subprocess
import re
from definitions import (
    ROOT_DIR,
    MODEL_PATTERN,
    ENV_PATTERN,
    VOCABULARY_FILE_NAME,
)
from vocabulary import KEYS_LIST
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Iterable
import torch
import json
from torch import nn
from models.ppo.policies import LatticeRecurrentActorCriticPolicy
from models.ppo.policies import MuscleTransformerPolicy
from train.algo_factory import AlgoFactory


def get_number(filename):
    return int(filename.split("_steps.zip")[0].split("_")[-1])


def load_model(
    algo,
    experiment_path,
    checkpoint_number,
    custom_objects=None,
    custom_config=None,
    custom_model_name=None,
):
    if custom_objects is None:
        custom_objects = {}
    if custom_config is not None:
        custom_config_path = os.path.join(experiment_path, custom_config)
        custom_config = json.load(open(custom_config_path, "r"))
        policy_kwargs = custom_config.get("policy_kwargs")
        if policy_kwargs is not None:
            if policy_kwargs.get("use_lattice"):
                policy_class = LatticeRecurrentActorCriticPolicy
                custom_objects["policy_class"] = policy_class

            activation_fn = policy_kwargs.get("activation_fn")
            if activation_fn is not None:
                policy_kwargs["activation_fn"] = getattr(nn, activation_fn)
            custom_objects["policy_kwargs"] = policy_kwargs
    model_file = (
        MODEL_PATTERN.replace("*", str(checkpoint_number))
        if custom_model_name is None
        else custom_model_name
    )
    model_path = os.path.join(experiment_path, model_file)
    print("Trying to load model from", model_path)
    if not os.path.exists(model_path):
        get_remote_checkpoint(experiment_path, checkpoint_number)
    try:
        model = AlgoFactory.get_algo_class(algo).load(
            model_path,
            custom_objects=custom_objects,
        )
    except:
        pass

    try:
        model = MuscleTransformerPolicy.load(model_path)
    except:
        pass

    return model


def get_best_checkpoint(steps, rewards, checkpoints, verbose=1):
    # Lowpass filter the rewards to avoid choosing a checkpoint at a peak due to noise
    clean_rewards = savgol_filter(rewards, window_length=51, polyorder=3)
    steps = list(steps)
    # Get the list of the closest steps to the checkpoints and the corresponding rewards
    closest_step_list = [
        min(steps, key=lambda x: abs(x - ckpt)) for ckpt in checkpoints
    ]
    closest_reward_list = [
        clean_rewards[steps.index(closest_step)] for closest_step in closest_step_list
    ]
    reward_ckpt_max = max(closest_reward_list)
    step_ckpt_max_approx = closest_step_list[closest_reward_list.index(reward_ckpt_max)]
    step_ckpt_max = min(checkpoints, key=lambda x: abs(x - step_ckpt_max_approx))
    if verbose:
        print(
            "Best checkpoint:",
            step_ckpt_max,
            ", corresponding reward:",
            reward_ckpt_max,
        )
    return step_ckpt_max


def get_data_from_tb_log(path, y, x="step", tb_config=None):
    if tb_config is None:
        tb_config = {}

    event_acc = EventAccumulator(path, tb_config)
    event_acc.Reload()

    if not isinstance(y, Iterable) or isinstance(y, str):
        y = [y]

    out_dict = {}
    for attr_name in y:
        if attr_name in event_acc.Tags()["scalars"]:
            x_vals, y_vals = np.array(
                [(getattr(el, x), el.value) for el in event_acc.Scalars(attr_name)]
            ).T
            out_dict[attr_name] = (x_vals, y_vals)
        else:
            out_dict[attr_name] = None
    return out_dict


def get_experiment_data(tb_dir_path, attributes, tb_config=None):
    experiment_data = {}
    folder_content = os.listdir(tb_dir_path)
    assert len(folder_content) == 1
    tb_file_name = folder_content[0]
    tb_file_path = os.path.join(tb_dir_path, tb_file_name)
    data_dict = get_data_from_tb_log(tb_file_path, attributes, tb_config=tb_config)
    for key, values in data_dict.items():
        if values is not None:
            x_vals, y_vals = values
            experiment_data_el = experiment_data.get(key)
            if experiment_data_el is None:
                experiment_data[key] = {}
                experiment_data[key]["x"] = [x_vals]
                experiment_data[key]["y"] = [y_vals]
            else:
                experiment_data[key]["x"].append(x_vals)
                experiment_data[key]["y"].append(y_vals)
    return experiment_data


def get_remote_checkpoint(
    host, host_project_root, experiment_path, checkpoint_num, verbose=True
):
    if verbose:
        print("Attempting to fetch remote experiment...")
    if checkpoint_num is None:
        raise NotImplementedError(
            "Selection of best checkpoint from the remote not implemented"
        )
    file_names = [
        "args.json",
        "*_config.json",
        VOCABULARY_FILE_NAME,
        MODEL_PATTERN.replace("*", str(checkpoint_num)),
        ENV_PATTERN.replace("*", str(checkpoint_num)),
    ]
    file_paths = [
        os.path.join(f"{host}:{host_project_root}", experiment_path, f)
        for f in file_names
    ]
    os.makedirs(os.path.join(ROOT_DIR, experiment_path), exist_ok=True)
    return_code = subprocess.run(
        ["rsync", *file_paths, os.path.join(ROOT_DIR, experiment_path)]
    )
    if return_code != 0:
        for path in file_paths:
            subprocess.run(["rsync", path, os.path.join(ROOT_DIR, experiment_path)])


def merge_strings(string_list):
    ret_list = []
    if len(string_list) > 2:
        for s in string_list:
            sub1 = re.sub("[^A-Z]", "", s)
            sub2 = "".join([a[0] for a in s.split("_")])
            if sub1 == "" and sub2 == "":
                ret_list.append("X")
            elif sub1 != "":
                ret_list.append(sub1)
            elif sub2 != "":
                ret_list.append(sub2)
            else:
                raise NotImplementedError
    else:
        ret_list = string_list
    return "_".join(ret_list)


def merge_task_names(task_list):
    if len(task_list) <= 2:
        # Concatenate tasks with underscores if 2 or fewer tasks
        return "_".join(task_list)
    else:
        # Shorten each task name by taking the first letter of each substring
        shortened_tasks = [
            "".join(word[0] for word in task.split("_")) for task in task_list
        ]
        # Concatenate the shortened task names with underscores
        return "_".join(shortened_tasks)


def token_to_string(token, vocabulary=KEYS_LIST):
    cur_name = []
    for i in range(len(token)):
        val = token[i].item()
        if val:
            cur_name.append(vocabulary[int(val)])
    cur_name = "|".join(cur_name)
    return cur_name


# if __name__ == "__main__" :

#     class CustomDataset(th_data.Dataset):
#         def __init__(self, data):
#             self.data = data

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             return self.data[idx]

#     # Example usage
#     data = torch.arange(100)  # Sample data from 0 to 99
#     dataset = CustomDataset(data)
#     batch_size = 5

#     continuous_sampler = ContinuousBatchSampler(dataset, batch_size)
#     data_loader = th_data.DataLoader(dataset, batch_sampler=continuous_sampler)

#     # Fetch and print batches
#     for batch in data_loader:
#         print(batch)
