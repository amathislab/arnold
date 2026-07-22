import gymnasium as gym
import numpy as np
import torch
from definitions import MASK_SUFFIX
from imitation.data.types import DictObs

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def infer_spaces(trajectories):
    observation_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(low=-np.inf, high=np.inf, shape=val.shape)
            for key, val in trajectories[0]["obs"].items()
        }
    )
    action_space = gym.spaces.Box(
        low=-1, high=1, shape=(trajectories[0]["acts"].shape[0],)
    )
    return observation_space, action_space


def get_larger_box(box_list):
    dim_list = []
    for box in box_list:
        for i, s in enumerate(box.shape):
            if len(dim_list) <= i:
                dim_list.append(s)
            else:
                dim_list[i] = max(dim_list[i], s)

    low = np.min([np.min(b.low) for b in box_list])
    high = np.max([np.max(b.high) for b in box_list])
    return gym.spaces.Box(low, high, shape=dim_list)


def fill_larger_box(larger_box, box, mask, ignore_last_dim=True):
    # print("DEBUG:", larger_box.shape, box.shape, mask.shape, larger_box[~mask].shape, larger_box[~mask, :box.shape[-1]].shape)
    # larger_box[~mask] = box.reshape(larger_box[~mask].shape)
    if len(box.shape) != len(larger_box.shape) :
        assert(len(box.shape) == len(larger_box.shape) + 1) # first dim is batch size
        if isinstance(larger_box, np.ndarray) :
            larger_box = np.repeat(larger_box[np.newaxis, ...], box.shape[0], axis=0)
        else :
            larger_box = torch.repeat_interleave(larger_box.unsqueeze(0), box.shape[0], dim=0)

    if len(box.shape) - int(ignore_last_dim) == 1:
        if ignore_last_dim:
            larger_box[~mask, : box.shape[-1]] = box.reshape(-1, box.shape[-1])
            larger_box[~mask, box.shape[-1] :] = 0
        else:
            larger_box[~mask] = box.flatten()
        larger_box[mask] = 0
    elif len(box.shape) == 2 :  # with batch dim
        b = box.shape[0]
        if ignore_last_dim:
            larger_box[:, ~mask, : box.shape[-1]] = box.reshape(b, -1, box.shape[-1])
            larger_box[:, ~mask, box.shape[-1] :] = 0
        else:
            larger_box[:, ~mask] = box.reshape(b, -1)
        larger_box[:, mask] = 0
    else :
        raise NotImplementedError("Only implemented for single-dim or 2-dim box observation / actions")

    return larger_box


def compute_box_mask(larger_box, smaller_box, ignore_last_dim=True):
    """Returns a mask with the same shape of larger_box, with True outside of the range
    of the shape of smaller_box. If smaller_box is None, the mask is False everywhere. If
    ignore_last_dim is True, then the mask is not computed along that dimension and the
    mask will have one fewer dim than larger_box.

    :param larger_box: A gym Box
    :param smaller_box: Another gym Box, whose size is smaller or equal than that of
    larger_box in all dimensions
    :return: the mask to remove values in larger_box which are not in smaller_box
    """
    if ignore_last_dim:
        mask_shape = larger_box.shape[:-1]
        if smaller_box is not None:
            smaller_box_shape = smaller_box.shape[:-1]
    else:
        mask_shape = larger_box.shape
        if smaller_box is not None:
            smaller_box_shape = smaller_box.shape

    mask = np.ones(mask_shape).astype(bool)
    if smaller_box is not None:
        valid_slice = tuple(slice(l) for l in smaller_box_shape)
    else:
        valid_slice = slice(0)  # No valid value
    mask[valid_slice] = False
    return mask


def compute_dict_mask(larger_dict, smaller_dict, ignore_last_dim=True):
    """Returns a mask with the same shape of larger_dict, with True outside of the range
    of the shape of smaller_dict. If smaller_dict is None, the mask is False everywhere. If
    ignore_last_dim is True, then the mask is not computed along that dimension and the
    mask will have one fewer dim than larger_dict.

    :param larger_dict: A gym Dict
    :param smaller_dict: Another gym Dict, whose size is smaller or equal than that of
    larger_dict in all dimensions
    :return: the mask to remove values in larger_dict which are not in smaller_dict
    """
    mask_dict = {}
    for key, larger_box in larger_dict.items():
        if not key.endswith(MASK_SUFFIX):
            smaller_box = smaller_dict.get(key)
            mask = compute_box_mask(larger_box, smaller_box, ignore_last_dim)
            mask_dict[key + MASK_SUFFIX] = mask
    return mask_dict


def merge_observation_spaces(observation_spaces, mask_last_obs_dim=False):
    if all([isinstance(x, gym.spaces.Box) for x in observation_spaces]):
        return get_larger_box(observation_spaces)
    elif all([isinstance(x, gym.spaces.Dict) for x in observation_spaces]):
        spaces_dict = {}
        all_keys = set.union(*[set(o.keys()) for o in observation_spaces])
        for key in all_keys:
            boxes_per_key = [
                o.get(key) for o in observation_spaces if o.get(key) is not None
            ]
            larger_box = get_larger_box(boxes_per_key)
            spaces_dict[key] = larger_box
            if mask_last_obs_dim:
                spaces_dict[key + MASK_SUFFIX] = larger_box
            else:
                spaces_dict[key + MASK_SUFFIX] = gym.spaces.Box(
                    -float("inf"), float("inf"), shape=larger_box.shape[:-1]
                )
        return gym.spaces.Dict(spaces_dict)
    else:
        raise NotImplementedError("Only implemented for Box or Dict observation spaces")


def merge_action_spaces(action_spaces):
    if not all([isinstance(x, gym.spaces.Box) for x in action_spaces]):
        raise NotImplementedError("Only implemented for Box action spaces")
    else:
        return get_larger_box(action_spaces)


def pad_observation(observation, larger_obs, obs_mask, ignore_last_dim=True):
    if isinstance(observation, np.ndarray) or isinstance(observation, torch.Tensor):
        obs = larger_obs.copy()
        obs = fill_larger_box(obs, observation, obs_mask, ignore_last_dim)
        return obs
    elif isinstance(observation, dict) or isinstance(observation, DictObs):
        obs = {key: val.copy() for key, val in larger_obs.items()}
        for key, value in observation.items():
            obs[key] = fill_larger_box(
                obs[key], value, obs_mask[key + MASK_SUFFIX], ignore_last_dim
            )
        obs.update(obs_mask)
        return obs
    else:
        raise NotImplementedError("Only implemented for Box or Dict observation spaces")


def pad_action(action, larger_act, act_mask, ignore_last_dim=False):
    if isinstance(action, np.ndarray) :
        act = larger_act.copy()
        act = fill_larger_box(act, action, act_mask, ignore_last_dim)
        return act
    elif isinstance(action, torch.Tensor) :
        act = larger_act.clone()
        act = fill_larger_box(act, action, act_mask, ignore_last_dim)
        return act
    else:
        raise NotImplementedError("Only implemented for Box action spaces")
