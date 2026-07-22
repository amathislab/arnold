from imitation.data import rollout, types
from typing import Dict, List, Union
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common import vec_env
from envs.multi_env import FlexibleMultiVecNormalize
from envs.multi_env_utilites import compute_box_mask, merge_action_spaces, pad_action
from models.ppo.policies import MuscleTransformerPolicy, to_tensor_dict


import numpy as np
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def call_muscle_transformer_policy(
    policy: MuscleTransformerPolicy, observation, deterministic, train_mode=False
):

    policy.set_training_mode(train_mode)
    if train_mode:
        obs_i_normalized = to_tensor_dict(observation, device=policy.device)
        action, value, _ = policy(
            obs_i_normalized,
            deterministic=deterministic,
        )
    else:
        with torch.no_grad():
            obs_i_normalized = to_tensor_dict(observation, device=policy.device)
            action, value, _ = policy(
                obs_i_normalized,
                deterministic=deterministic,
            )

    return action, value


def get_muscle_transformer_value(
    policy: MuscleTransformerPolicy,
    observation,
    train_mode=False,
):

    policy.set_training_mode(train_mode)
    if train_mode:
        obs_i_normalized = to_tensor_dict(observation, device=policy.device)
        action, value, _ = policy(
            obs_i_normalized,
            deterministic=False,
        )
    else:
        with torch.no_grad():
            obs_i_normalized = to_tensor_dict(observation, device=policy.device)
            action, value, _ = policy(
                obs_i_normalized,
                deterministic=False,
            )

    return value


def evaluate_muscle_transformer_action(
    policy: MuscleTransformerPolicy, observation, action, train_mode=False
):

    policy.set_training_mode(train_mode)
    if train_mode:
        obs_i_normalized = to_tensor_dict(observation, device=policy.device)
        values, log_prob, entropy = policy.evaluate_actions(obs_i_normalized, action)
    else:
        with torch.no_grad():
            obs_i_normalized = to_tensor_dict(observation, device=policy.device)
            values, log_prob, entropy = policy.evaluate_actions(
                obs_i_normalized,
                action,
            )

    return values, log_prob, entropy


def call_sb3_policy(
    policy: RecurrentActorCriticPolicy,
    observation,
    states,
    episode_start,
    deterministic,
    train_mode=False,
):

    policy.set_training_mode(train_mode)

    observation, vectorized_env = policy.obs_to_tensor(observation)
    episode_start = torch.tensor(
        episode_start, dtype=torch.float32, device=policy.device
    )

    if isinstance(observation, dict):
        n_envs = observation[next(iter(observation.keys()))].shape[0]
    else:
        n_envs = observation.shape[0]

    if states is None:
        # Initialize hidden states to zeros
        states = torch.concatenate(
            [
                torch.zeros(policy.lstm_hidden_state_shape, device=policy.device)
                for _ in range(n_envs)
            ],
            axis=1,
        )
        states = (states, states.clone())
        states = RNNStates(pi=states, vf=states)

    if train_mode:
        actions, values, _, states = policy(
            observation,
            lstm_states=states,
            episode_starts=episode_start,
            deterministic=deterministic,
        )
    else:
        with torch.no_grad():
            actions, values, _, states = policy(
                observation,
                lstm_states=states,
                episode_starts=episode_start,
                deterministic=deterministic,
            )

    return actions, values, states


def get_sb3_value(
    policy: RecurrentActorCriticPolicy,
    observation,
    states,
    episode_start,
    train_mode=False,
):

    policy.set_training_mode(train_mode)

    observation, vectorized_env = policy.obs_to_tensor(observation)
    episode_start = torch.tensor(
        episode_start, dtype=torch.float32, device=policy.device
    )

    if train_mode:
        values = policy.predict_values(
            observation, lstm_states=states.vf, episode_starts=episode_start
        )
    else:
        with torch.no_grad():
            values = policy.predict_values(
                observation, lstm_states=states.vf, episode_starts=episode_start
            )

    return values


def evaluate_sb3_action(
    policy: RecurrentActorCriticPolicy,
    observation,
    action,
    states,
    episode_start,
    train_mode=False,
):

    policy.set_training_mode(train_mode)

    observation, vectorized_env = policy.obs_to_tensor(observation)
    episode_start = torch.tensor(
        episode_start, dtype=torch.float32, device=policy.device
    )

    if states is None:
        # Initialize hidden states to zeros
        states = torch.zeros(policy.lstm_hidden_state_shape, device=policy.device)
        states = (states, states.clone())
        states = RNNStates(pi=states, vf=states)

    if train_mode:
        values, log_prob, entropy = policy.evaluate_actions(
            observation, action, lstm_states=states, episode_starts=episode_start
        )
    else:
        with torch.no_grad():
            values, log_prob, entropy = policy.evaluate_actions(
                observation, action, lstm_states=states, episode_starts=episode_start
            )

    return values, log_prob, entropy


class policy_list_to_callable:

    def __init__(
        self,
        policy_list,
        venv,
        train_env_list,
        vecnormalize_list,
        deterministic_policy=False,
        train_mode=False,
    ):
        """For N environments and M policies, each policy predicts an action for M/N environments.
        We also need to convert the actions to the format expected by the environment.
        """
        self.num_envs = venv.num_envs
        self.envs_per_policy = self.num_envs // len(policy_list)
        self.policy_list = policy_list
        self.train_env_list = train_env_list
        self.vecnormalize_list = vecnormalize_list
        self.deterministic_policy = deterministic_policy
        self.train_mode = train_mode
        self.action_space_list = [
            venv.action_space if not train_env else train_env.action_space
            for train_env in train_env_list
        ]
        self.large_action_space = merge_action_spaces(self.action_space_list)
        self.large_action_sample = torch.from_numpy(self.large_action_space.sample())
        self.action_mask_list = [
            torch.from_numpy(
                compute_box_mask(
                    self.large_action_space, action_space_i, ignore_last_dim=False
                )
            )
            for action_space_i in self.action_space_list
        ]

    def evaluate_actions(
        self, env_ids, obs, actions, obs_dicts, states, episode_starts
    ):

        value_list = []
        log_prob_list = []
        entropy_list = []

        for key in obs.keys():
            if isinstance(obs[key], torch.Tensor):
                obs[key] = obs[key].cpu().numpy()

        for i, (env_i, act_i, obs_dict_i, state_i, episode_start_i) in enumerate(
            zip(env_ids, actions, obs_dicts, states, episode_starts)
        ):
            expert_id = env_i // self.envs_per_policy
            policy_i = self.policy_list[expert_id]
            train_env_i = self.train_env_list[expert_id]
            vecnormalize_i = self.vecnormalize_list[expert_id]

            if train_env_i is None:
                obs_i = {key: obs[key][env_i : env_i + 1] for key in obs}
                assert isinstance(vecnormalize_i, FlexibleMultiVecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_single_obs_dict(
                    obs_i, env_idx=expert_id
                )

            else:
                obs_i_list = []

                obs_train_style = np.zeros(0)
                for key in train_env_i.obs_keys:
                    obs_train_style = np.concatenate(
                        [obs_train_style, obs_dict_i[key].ravel()]
                    )
                obs_i_list.append(obs_train_style)
                obs_i = np.stack(obs_i_list)
                assert isinstance(vecnormalize_i, VecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_obs(obs_i)

            if isinstance(policy_i, MuscleTransformerPolicy):
                value, log_prob, entropy = evaluate_muscle_transformer_action(
                    policy_i, obs_i_normalized, act_i, train_mode=self.train_mode
                )
            elif isinstance(policy_i, RecurrentActorCriticPolicy):
                value, log_prob, entropy = evaluate_sb3_action(
                    policy_i,
                    obs_i_normalized,
                    act_i,
                    state_i,
                    episode_start_i,
                    train_mode=self.train_mode,
                )
            else:
                raise NotImplementedError(
                    f"Policy type {type(policy_i)} is not supported"
                )

            value_list.append(value)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)

        return torch.cat(value_list), torch.cat(log_prob_list), torch.cat(entropy_list)

    def predict_values(self, obs, obs_dicts, states, episode_starts):

        value_list = []
        for i, (policy, train_env_i, vecnormalize_i) in enumerate(
            zip(self.policy_list, self.train_env_list, self.vecnormalize_list)
        ):
            if train_env_i is None:
                obs_i = {
                    key: obs[key][
                        i * self.envs_per_policy : (i + 1) * self.envs_per_policy
                    ]
                    for key in obs
                }
                assert isinstance(vecnormalize_i, FlexibleMultiVecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_single_obs_dict(
                    obs_i, env_idx=i
                )

            else:
                obs_i_list = []
                for d in obs_dicts[
                    i * self.envs_per_policy : (i + 1) * self.envs_per_policy
                ]:
                    obs_train_style = np.zeros(0)
                    for key in train_env_i.obs_keys:
                        obs_train_style = np.concatenate(
                            [obs_train_style, d[key].ravel()]
                        )
                    obs_i_list.append(obs_train_style)
                obs_i = np.stack(obs_i_list)
                assert isinstance(vecnormalize_i, VecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_obs(obs_i)

            if states is None:
                states_i = None
            else:
                states_i = states[i]

            episode_starts_i = episode_starts[
                i * self.envs_per_policy : (i + 1) * self.envs_per_policy
            ]

            if isinstance(policy, MuscleTransformerPolicy):
                value = get_muscle_transformer_value(
                    policy,
                    obs_i_normalized,
                    train_mode=self.train_mode,
                )
            elif isinstance(policy, RecurrentActorCriticPolicy):
                value = get_sb3_value(
                    policy,
                    obs_i_normalized,
                    states_i,
                    episode_starts_i,
                    train_mode=self.train_mode,
                )
            else:
                raise NotImplementedError(
                    f"Policy type {type(policy)} is not supported"
                )

            # Unormalize value!!!
            # if isinstance(vecnormalize_i, FlexibleMultiVecNormalize) :
            #     value = vecnormalize_i.unnormalize_single_reward(value, i)
            # elif isinstance(vecnormalize_i, vec_env.VecNormalize) :
            #     value = vecnormalize_i.unnormalize_reward(value)
            # else :
            #     raise NotImplementedError(f"VecNormalize type {type(vecnormalize_i)} is not supported")

            value_list.append(value.reshape(-1))

        return torch.cat(value_list)

    def predict_single_value(self, obs, obs_dict, env_idx, states, episode_starts):

        expert_id = env_idx // self.envs_per_policy
        policy = self.policy_list[expert_id]
        train_env_i = self.train_env_list[expert_id]
        vecnormalize_i = self.vecnormalize_list[expert_id]
        if train_env_i is None:
            obs_i = {key: obs[key][None, ...] for key in obs}
            assert isinstance(vecnormalize_i, FlexibleMultiVecNormalize)
            obs_i_normalized = vecnormalize_i.normalize_single_obs_dict(
                obs_i, env_idx=expert_id
            )

        else:
            obs_i_list = []

            obs_train_style = np.zeros(0)
            for key in train_env_i.obs_keys:
                obs_train_style = np.concatenate(
                    [obs_train_style, obs_dict[key].ravel()]
                )
            obs_i_list.append(obs_train_style)
            obs_i = np.stack(obs_i_list)
            assert isinstance(vecnormalize_i, VecNormalize)
            obs_i_normalized = vecnormalize_i.normalize_obs(obs_i)

        states_i = states
        episode_starts_i = episode_starts

        if isinstance(policy, MuscleTransformerPolicy):
            value = get_muscle_transformer_value(
                policy, obs_i_normalized, train_mode=self.train_mode
            )
        elif isinstance(policy, RecurrentActorCriticPolicy):
            _, value, _ = call_sb3_policy(
                policy,
                obs_i_normalized,
                states_i,
                episode_starts_i,
                self.deterministic_policy,
                train_mode=self.train_mode,
            )
        else:
            raise NotImplementedError(f"Policy type {type(policy)} is not supported")

        # Unormalize value!!!
        # if isinstance(vecnormalize_i, FlexibleMultiVecNormalize) :
        #     value = vecnormalize_i.unnormalize_single_reward(value, i)
        # elif isinstance(vecnormalize_i, vec_env.VecNormalize) :
        #     value = vecnormalize_i.unnormalize_reward(value)
        # else :
        #     raise NotImplementedError(f"VecNormalize type {type(vecnormalize_i)} is not supported")
        return value

    def __call__(self, obs, obs_dicts, states, episode_starts):

        actions_list = []
        states_list = []
        value_list = []
        for i, (
            policy,
            train_env_i,
            vecnormalize_i,
            action_space_i,
            action_mask_i,
        ) in enumerate(
            zip(
                self.policy_list,
                self.train_env_list,
                self.vecnormalize_list,
                self.action_space_list,
                self.action_mask_list,
            )
        ):
            if train_env_i is None:
                obs_i = {
                    key: obs[key][
                        i * self.envs_per_policy : (i + 1) * self.envs_per_policy
                    ]
                    for key in obs
                }
                assert isinstance(vecnormalize_i, FlexibleMultiVecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_single_obs_dict(
                    obs_i, env_idx=i
                )

            else:
                obs_i_list = []
                for d in obs_dicts[
                    i * self.envs_per_policy : (i + 1) * self.envs_per_policy
                ]:
                    obs_train_style = np.zeros(0)
                    for key in train_env_i.obs_keys:
                        obs_train_style = np.concatenate(
                            [obs_train_style, d[key].ravel()]
                        )
                    obs_i_list.append(obs_train_style)
                obs_i = np.stack(obs_i_list)
                assert isinstance(vecnormalize_i, VecNormalize)
                obs_i_normalized = vecnormalize_i.normalize_obs(obs_i)

            if states is None:
                states_i = None
            else:
                states_i = states[i]

            episode_starts_i = episode_starts[
                i * self.envs_per_policy : (i + 1) * self.envs_per_policy
            ]

            if isinstance(policy, MuscleTransformerPolicy):
                action, value = call_muscle_transformer_policy(
                    policy,
                    obs_i_normalized,
                    self.deterministic_policy,
                    train_mode=self.train_mode,
                )
                states_i_new = None
            elif isinstance(policy, RecurrentActorCriticPolicy):
                action, value, states_i_new = call_sb3_policy(
                    policy,
                    obs_i_normalized,
                    states_i,
                    episode_starts_i,
                    self.deterministic_policy,
                    train_mode=self.train_mode,
                )
            else:
                raise NotImplementedError(
                    f"Policy type {type(policy)} is not supported"
                )

            # Unormalize value!!!
            # if isinstance(vecnormalize_i, FlexibleMultiVecNormalize) :
            #     value = vecnormalize_i.unnormalize_single_reward(value, i)
            # elif isinstance(vecnormalize_i, vec_env.VecNormalize) :
            #     value = vecnormalize_i.unnormalize_reward(value)
            # else :
            #     raise NotImplementedError(f"VecNormalize type {type(vecnormalize_i)} is not supported")

            actions_list.append(
                pad_action(
                    action,
                    self.large_action_sample.to(action.device),
                    action_mask_i.to(action.device),
                )
            )
            states_list.append(states_i_new)
            value_list.append(value.reshape(-1))

        return torch.cat(actions_list), torch.cat(value_list), states_list


def beta_mix_action(expert_actions, policy_actions, beta):

    dist = np.random.uniform(low=0.0, high=1.0, size=(expert_actions.shape[0],))
    sel = dist < beta
    sel = sel[:, None]
    return expert_actions * sel + policy_actions * (1 - sel)


class MyTrajectoryAccumulator(rollout.TrajectoryAccumulator):
    """
    This TrajectoryAccumulator will ignore terminal observations
    """

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: Union[types.Observation, Dict[str, np.ndarray]],
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ) -> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.

        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs: List[types.TrajectoryWithRew] = []
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # iterate through environments
        for env_idx in range(len(wrapped_obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        # iterate through steps
        zip_iter = enumerate(zip(acts, wrapped_obs, rews, dones, infos))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                # real_ob = types.maybe_wrap_in_dictobs(info["terminal_observation"])
                pass
            else:
                real_ob = ob
                self.add_step(
                    dict(
                        acts=act,
                        rews=rew,
                        # this is not the obs corresponding to `act`, but rather the obs
                        # *after* `act` (see above)
                        obs=real_ob,
                        infos=info,
                    ),
                    env_idx,
                )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


def generate_trajectoreis(
    expert_policy_list,
    dagger_policy,
    venv,
    train_env_list,
    sample_until,
    estimated_sample_nums,
    vecnormalize_list,
    deterministic_expert=False,
    beta=0,
):

    get_expert_action = policy_list_to_callable(
        expert_policy_list,
        venv,
        train_env_list,
        vecnormalize_list,
        deterministic_policy=deterministic_expert,
    )
    get_dagger_action = rollout.policy_to_callable(dagger_policy, venv)

    trajectories = []
    traj_accum = (
        MyTrajectoryAccumulator()
    )  # Use a customized accumulator to prevent "terminal_obs" errors
    obs = venv.reset()
    assert isinstance(obs, (np.ndarray, dict))
    wrapped_obs = types.maybe_wrap_in_dictobs(obs)

    for env_idx, ob in enumerate(wrapped_obs):

        traj_accum.add_step({"obs": ob}, key=env_idx)

    active = np.ones(venv.num_envs, dtype=bool)
    expert_states = None
    dagger_state = None
    dones = np.zeros(venv.num_envs, dtype=bool)
    step = 0
    env_per_policy = venv.num_envs // len(expert_policy_list)
    total_normalized_reward_per_env = np.zeros(len(expert_policy_list))
    total_unormalized_reward_per_env = np.zeros(len(expert_policy_list))
    total_active_steps_per_env = np.zeros(len(expert_policy_list))

    with tqdm.tqdm(
        total=estimated_sample_nums, desc="Generating new trajectories"
    ) as pbar:
        while np.any(active):

            step += 1
            obs_dicts = venv.get_attr("obs_dict")
            expert_actions, expert_values, expert_states = get_expert_action(
                obs, obs_dicts, expert_states, dones
            )
            expert_actions = expert_actions.cpu().numpy()
            expert_values = expert_values.cpu().numpy()

            dagger_action, dagger_state = get_dagger_action(obs, dagger_state, dones)
            mixed_action = beta_mix_action(expert_actions, dagger_action, beta)
            obs, rews, dones, _ = venv.step(mixed_action)

            rews_reshape = (rews * active).reshape(-1, env_per_policy)
            total_normalized_reward_per_env += rews_reshape.sum(axis=-1)

            for i, vecnormalize_i in enumerate(vecnormalize_list):
                rew_i = rews_reshape[i, :]
                if isinstance(vecnormalize_i, FlexibleMultiVecNormalize):
                    rew_i_unormalized = vecnormalize_i.unnormalize_single_reward(
                        rew_i, i
                    )
                elif isinstance(vecnormalize_i, vec_env.VecNormalize):
                    rew_i_unormalized = vecnormalize_i.unnormalize_reward(rew_i)
                else:
                    raise NotImplementedError(
                        f"VecNormalize type {type(vecnormalize_i)} is not supported"
                    )
                total_unormalized_reward_per_env[i] += rew_i_unormalized.sum()
            total_active_steps_per_env += active.reshape(-1, env_per_policy).sum(
                axis=-1
            )

            assert isinstance(obs, (np.ndarray, dict))
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            dones &= active

            new_trajectories = traj_accum.add_steps_and_auto_finish(
                expert_actions,
                wrapped_obs,
                expert_values,
                dones,
                [{} for _ in range(venv.num_envs)],
            )
            trajectories.extend(new_trajectories)

            if sample_until(trajectories):
                active &= ~dones

            pbar.update(sum(len(t.obs) - 1 for t in new_trajectories))

    info = {
        "total_normalized_reward_per_env": total_normalized_reward_per_env,
        "total_unormalized_reward_per_env": total_unormalized_reward_per_env,
        "total_active_steps_per_env": total_active_steps_per_env,
        "avg_normalized_reward_per_env": total_normalized_reward_per_env
        / (total_active_steps_per_env + 1e-9),
        "avg_unormalized_reward_per_env": total_unormalized_reward_per_env
        / (total_active_steps_per_env + 1e-9),
    }
    return trajectories, info
