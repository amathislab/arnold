from torch.utils.data import DataLoader
from imitation.data import rollout, types
from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.types import DictObs
from envs.multi_env_utilites import (
    infer_spaces,
    merge_action_spaces,
    merge_observation_spaces,
)
from envs.multi_env_utilites import (
    pad_observation,
    compute_box_mask,
    compute_dict_mask,
    pad_action,
)
from definitions import (
    STOCHASTIC_DF_NAME,
    DETERMINISTIC_DF_NAME,
)
import pandas as pd
import numpy as np
import tqdm
import os
from algos.bc_value import rew_transitions_collate_fn


class MultiEnvDataLoader(DataLoader):
    def __init__(
        self,
        paths,
        traj_per_dataset,
        batch_size,
        shuffle=False,
        mask_last_obs_dim=False,
        stochastic_df=True,
        **kwargs,
    ):
        self.paths = paths
        self.traj_per_dataset = traj_per_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask_last_obs_dim = mask_last_obs_dim
        self.stochastic_df = stochastic_df
        dataset, self.raw_transitions, self.obs_space, self.act_space = self.build_dataset()

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=rew_transitions_collate_fn,
            **kwargs,
        )

    def build_dataset(self):
        num_dataset = len(self.paths)
        raw_trajectories = []
        obs_spaces = []
        act_spaces = []
        for id, path in enumerate(self.paths):
            traj = self.load_single_env_dataset(path)
            obs_space, act_space = infer_spaces(traj)
            raw_trajectories.append(traj)
            obs_spaces.append(obs_space)
            act_spaces.append(act_space)
        obs_space = merge_observation_spaces(obs_spaces, mask_last_obs_dim=False)
        act_space = merge_action_spaces(act_spaces)

        # For each environment, pad the observations and actions to the maximum space
        random_obs = obs_space.sample()
        random_action = act_space.sample()
        raw_obs_list = []
        raw_act_list = []
        raw_next_obs_list = []
        raw_infos_list = []
        raw_dones_list = []
        raw_rews_list = []
        raw_obs_mask_list = []
        raw_act_mask_list = []
        for traj, smaller_obs_space, smaller_act_space in zip(
            raw_trajectories, obs_spaces, act_spaces
        ):
            cur_obs_list = []
            cur_act_list = []
            cur_next_obs_list = []
            cur_infos_list = []
            cur_dones_list = []
            cur_rews_list = []
            cur_obs_mask = compute_dict_mask(
                obs_space, smaller_obs_space, ignore_last_dim=not self.mask_last_obs_dim
            )
            cur_act_mask = compute_box_mask(
                act_space, smaller_act_space, ignore_last_dim=False
            )
            for transition in traj:
                padded_obs = pad_observation(
                    transition["obs"],
                    random_obs,
                    cur_obs_mask,
                    ignore_last_dim=not self.mask_last_obs_dim,
                )
                cur_obs_list.append(DictObs(padded_obs))
                padded_next_obs = pad_observation(
                    transition["next_obs"],
                    random_obs,
                    cur_obs_mask,
                    ignore_last_dim=not self.mask_last_obs_dim,
                )
                cur_next_obs_list.append(DictObs(padded_next_obs))
                padded_act = pad_action(transition["acts"], random_action, cur_act_mask)
                cur_act_list.append(padded_act)
                cur_infos_list.append(transition["infos"])
                cur_dones_list.append(transition["dones"])
                cur_rews_list.append(transition["rews"])
            
            raw_obs_list.append(cur_obs_list)
            raw_act_list.append(cur_act_list)
            raw_next_obs_list.append(cur_next_obs_list)
            raw_infos_list.append(cur_infos_list)
            raw_dones_list.append(cur_dones_list)
            raw_rews_list.append(cur_rews_list)
            raw_obs_mask_list.append(cur_obs_mask)
            raw_act_mask_list.append(cur_act_mask)
        
        # Concatenate the observations, actions, next_observations, dones, rewards
        raw_transitions = []
        cat_obs_list = []
        cat_act_list = []
        cat_next_obs_list = []
        cat_dones_list = []
        cat_rews_list = []
        cat_infos_list = []
        for obs_list, act_list, next_obs_list, dones_list, rews_list, infos_list in zip(
            raw_obs_list, raw_act_list, raw_next_obs_list, raw_dones_list, raw_rews_list, raw_infos_list
        ):
            raw_transitions.append(
                types.TransitionsWithRew(
                    obs=DictObs.stack(obs_list),
                    acts=np.stack(act_list),
                    infos=infos_list,
                    next_obs=DictObs.stack(next_obs_list),
                    dones=np.stack(dones_list),
                    rews=np.stack(rews_list),
                )
            )
            cat_obs_list.extend(obs_list)
            cat_act_list.extend(act_list)
            cat_next_obs_list.extend(next_obs_list)
            cat_dones_list.extend(dones_list)
            cat_rews_list.extend(rews_list)
            cat_infos_list.extend(infos_list)
            
        # Problem: the concatenation is removing the size of the observation, action, ... because there is no "batch" dimension
        # obs = DictObs.stack(obs_list)
        # next_obs = DictObs.stack(next_obs_list)
        # acts = np.stack(act_list)
        # dones = np.stack(dones_list)
        # rews = np.stack(rews_list)
        # dataset = types.TransitionsWithRew(obs, acts, infos_list, next_obs, dones, rews)
        dataset = types.TransitionsWithRew(
            obs=DictObs.stack(cat_obs_list),
            acts=np.stack(cat_act_list),
            infos=cat_infos_list,
            next_obs=DictObs.stack(cat_next_obs_list),
            dones=np.stack(cat_dones_list),
            rews=np.stack(cat_rews_list)
        )
        return dataset, raw_transitions, obs_space, act_space

    def load_single_env_dataset(self, path):
        if self.stochastic_df:
            df_path = os.path.join(path, STOCHASTIC_DF_NAME)
        else:
            df_path = os.path.join(path, DETERMINISTIC_DF_NAME)

        print("Loading trajectories from", df_path)
        dataset = pd.read_pickle(df_path)[: self.traj_per_dataset]
        print("Trajectories in memory, analyzing")
        trajectories_accum = TrajectoryAccumulator()

        unique_episodes = dataset["episode"].unique()
        trajectories = []
        with tqdm.tqdm(total=len(dataset)) as bar:
            for episode_id in unique_episodes:
                episode_data = dataset[dataset["episode"] == episode_id]
                obs = DictObs(episode_data.iloc[0]["observation"])

                # add the first observation (after reset)
                trajectories_accum.add_step(dict(obs=obs), key=episode_id)

                # iterate through action, reward and next_observation of the episode data
                for _, step in episode_data.iterrows():
                    step_dict = dict(
                        obs=DictObs(step["next_observation"]),
                        acts=step["action"],
                        rews=step["reward"],
                        infos="",
                    )
                    trajectories_accum.add_step(step_dict=step_dict, key=episode_id)
                    bar.update(1)
                traj = trajectories_accum.finish_trajectory(
                    key=episode_id, terminal=True
                )
                trajectories.append(traj)
        del trajectories_accum
        print("Loaded", len(trajectories), "episodes from", path)

        flattened_trajectories = rollout.flatten_trajectories_with_rew(trajectories)
        return flattened_trajectories


def cut_trajectory(path, save_path, length):

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    print("Loading trajectories from", path)
    dataset = pd.read_pickle(path)[:length]
    dataset.to_pickle(save_path)


def cut_trajectories(load_path, save_path, length):

    for load, save in zip(load_path, save_path):
        cut_trajectory(load, save, length)
