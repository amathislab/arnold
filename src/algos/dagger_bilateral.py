from imitation.algorithms.dagger import (
    DAggerTrainer,
    SimpleDAggerTrainer,
    InteractiveTrajectoryCollector,
    _save_dagger_demo,
)
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
import torch
import numpy as np
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.vec_env import VecEnv
from torch.utils import data as th_data
from gymnasium import spaces
from imitation.algorithms import base, bc
from imitation.data import rollout, serialize, types
from imitation.util import logger as imit_logger
from imitation.data.types import DictObs
import logging
import tqdm
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union, Dict
from imitation.algorithms import base
from imitation.util import util
from imitation.algorithms.dagger import LinearBetaSchedule
from gymnasium import Env
from envs.env_mixins import SpecsObsMixin
from envs.multi_env import FlexibleMultiVecNormalize
from algos.dagger_value import policy_list_to_callable, MyTrajectoryAccumulator, beta_mix_action
from algos.bc_bilateral import BCBilateral
from models.ppo.policies import BilateralMuscleTransformerPolicy
from torch_utils import generate_token_mask_vectorized

class bilateral_policy_to_callable() :

    def __init__(
        self,
        policy: rollout.AnyPolicy,
        venv: VecEnv,
        deterministic_policy: bool = False,
        time_skip: int = 1,
        masking_ratio: float = 0.5
    ) :

        self.policy = policy
        self.venv = venv
        self.deterministic_policy = deterministic_policy

        self.stacked_observations = {}
        self.stacked_dones = None
        self.stacked_observation_num = 0
        self.time_skip = time_skip
        self.masking_ratio = masking_ratio

    def append_observation(
        self,
        observations: Union[np.ndarray, Dict[str, np.ndarray]],
        episode_starts: np.ndarray
    ) :
        # import ipdb
        # ipdb.set_trace()

        self.stacked_observation_num += 1
        for key in observations.keys():
            if key not in self.stacked_observations:
                self.stacked_observations[key] = observations[key][:, None, ...]
            else :
                self.stacked_observations[key] = np.concatenate((self.stacked_observations[key], observations[key][:, None, ...]), axis=1)
        if self.stacked_dones is None :
            self.stacked_dones = episode_starts[:, None]
        else :
            self.stacked_dones = np.concatenate((self.stacked_dones, episode_starts[:, None]), axis=1)

    def __call__(
        self,
        observations: Union[np.ndarray, Dict[str, np.ndarray]],
        states: Optional[Tuple[np.ndarray, ...]],
        episode_starts: Optional[np.ndarray],
    ) :

        self.append_observation(observations, episode_starts)

        def get_history(arr, time_skip) :
            start = (arr.shape[1]-1) % time_skip
            return arr[:, start::time_skip, ...]


        if isinstance(self.policy, BilateralMuscleTransformerPolicy) :

            k_bilateral = self.policy.k_bilateral
            keep_history = self.time_skip * (k_bilateral - 1) + 1
            obs_feed = {}
            # original_batch_shape = self.stacked_observations["obs"].shape
            for key in self.stacked_observations.keys():
                clipped_obs = self.stacked_observations[key][:, -keep_history:, ...].copy()
                self.stacked_observations[key] = clipped_obs
                obs_feed[key] = get_history(clipped_obs, self.time_skip)
            clipped_dones = self.stacked_dones[:, -keep_history:].copy()
            self.stacked_dones = clipped_dones
            dones_feed = get_history(clipped_dones, self.time_skip)

            # done_extended = np.concatenate((self.stacked_dones, np.ones((self.venv.num_envs, 1))), axis=1)

            # shape = self.stacked_observations["obs"].shape
            # obs_mask = torch.zeros(shape[1], dtype=torch.bool)
            # indices = torch.randperm(shape[1] - 1)[:int(shape[1]*self.masking_ratio)]
            # obs_mask[indices] = 1
            # # Ensure the last element is 1
            # obs_mask[-1] = 1
            shape = obs_feed["obs"].shape
            obs_mask = generate_token_mask_vectorized(shape[0], shape[1], int(shape[1]*self.masking_ratio)+1)

            action, _ = self.policy.predict(
                obs_feed,
                states,
                episode_start = None,
                deterministic = self.deterministic_policy,
                predict_value = False,
                done = dones_feed,
                obs_mask = obs_mask,
                act_mask = obs_mask
            )

        else :
            raise NotImplementedError(f"Policy type {type(self.policy)} is not supported")
        
        return action, states

def add_timestep_to_obs(obs, timestep):

        obs["timestep"] = timestep

        return obs

def generate_trajectoreis(
    expert_policy_list,
    dagger_policy,
    venv,
    train_env_list,
    min_timesteps,
    estimated_sample_nums,
    vecnormalize_list,
    beta = 0,
    time_skip = 6,
    masking_ratio = 0.5,
    deterministic_expert = False
):

    num_policy = len(expert_policy_list)
    env_per_policy = venv.num_envs // num_policy

    get_expert_action = policy_list_to_callable(
        expert_policy_list, venv, train_env_list, vecnormalize_list, deterministic_policy=deterministic_expert
    )
    get_dagger_action = bilateral_policy_to_callable(
        dagger_policy, venv, time_skip=time_skip, masking_ratio=masking_ratio, deterministic_policy=False
    )

    trajectory_list = [[] for i in range(num_policy)]
    traj_accum_list = [MyTrajectoryAccumulator() for i in range(num_policy)]

    active = np.ones(venv.num_envs, dtype=bool)
    expert_states = None
    dagger_state = None
    dones = np.ones(venv.num_envs, dtype=bool)
    timestep = np.zeros(venv.num_envs, dtype=int)
    step = 0
    total_normalized_reward_per_env = np.zeros(num_policy)
    total_unormalized_reward_per_env = np.zeros(num_policy)
    total_active_steps_per_env = np.zeros(num_policy)

    obs = venv.reset()
    assert isinstance(obs, dict)
    obs = add_timestep_to_obs(obs, timestep)
    wrapped_obs = types.maybe_wrap_in_dictobs(obs)

    for env_idx, (ob, done) in enumerate(zip(wrapped_obs, dones)):
        traj_accum_list[env_idx // env_per_policy].add_step(
            {
                "obs": ob,
            },
            key=env_idx % env_per_policy
        )

    with tqdm.tqdm(total=estimated_sample_nums, desc="Generating new trajectories") as pbar :
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
                if isinstance(vecnormalize_i, FlexibleMultiVecNormalize) :
                    rew_i_unormalized = vecnormalize_i.unnormalize_single_reward(rew_i, i)
                elif isinstance(vecnormalize_i, vec_env.VecNormalize) :
                    rew_i_unormalized = vecnormalize_i.unnormalize_reward(rew_i)
                else :
                    raise NotImplementedError(f"VecNormalize type {type(vecnormalize_i)} is not supported")
                total_unormalized_reward_per_env[i] += rew_i_unormalized.sum()
            total_active_steps_per_env += active.reshape(-1, env_per_policy).sum(axis=-1)

            assert isinstance(obs, (np.ndarray, dict))
            timestep = (timestep + 1) * (1 - dones)
            obs = add_timestep_to_obs(obs, timestep)
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            dones &= active


            for policy_idx in range(num_policy):
                l = policy_idx * env_per_policy
                r = (policy_idx + 1) * env_per_policy
                new_trajectories = traj_accum_list[policy_idx].add_steps_and_auto_finish(
                    expert_actions[l: r],
                    wrapped_obs[l: r],
                    expert_values[l: r],
                    dones[l: r],
                    [{} for _ in range(env_per_policy)]
                )
                trajectory_list[policy_idx].extend(new_trajectories)
                pbar.update(sum(len(t.obs) - 1 for t in new_trajectories))

            all_good = True
            for trajectory in trajectory_list :
                tot_len = sum(len(t) for t in trajectory)
                if tot_len < min_timesteps :
                    all_good = False
                    break
            if all_good:
                active &= ~dones

    info = {
        "total_normalized_reward_per_env": total_normalized_reward_per_env,
        "total_unormalized_reward_per_env": total_unormalized_reward_per_env,
        "total_active_steps_per_env": total_active_steps_per_env,
        "avg_normalized_reward_per_env": total_normalized_reward_per_env / (total_active_steps_per_env + 1e-9),
        "avg_unormalized_reward_per_env": total_unormalized_reward_per_env / (total_active_steps_per_env + 1e-9)
    }
    return trajectory_list, info

class MultiEnvBilateralDAggerTrainer(base.BaseImitationAlgorithm):

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        env_names: Sequence[str],
        train_env_list: Env,
        scratch_dir: types.AnyPath,
        rng: np.random.Generator,
        beta_schedule: Optional[Callable[[int], float]] = None,
        omit_history_rollout: bool = False,
        expert_policy_list: policies.BasePolicy,
        deterministic_expert: bool = False,
        bc_trainer: BCBilateral,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        vec_normalize_list: Optional[vec_env.VecNormalize] = None,
        time_skip: int = 6,
        **unused_kwargs,
    ):

        super().__init__(custom_logger=custom_logger)

        print("Unused kwargs of MultiEnvBilateralDAggerTrainer: ", unused_kwargs)

        if beta_schedule is None:
            beta_schedule = 15
        self.beta_schedule = LinearBetaSchedule(beta_schedule)
        self.scratch_dir = util.parse_path(scratch_dir)
        self.venv = venv
        self.env_names = env_names
        self.round_num = 0
        self.omit_history_rollout = omit_history_rollout
        self._last_loaded_round = -1
        self._all_demos = []
        self.rng = rng

        self.bc_trainer = bc_trainer
        self.bc_trainer.logger = self.logger
        self.time_skip = time_skip

        # print(expert_policy.observation_space)
        # print(venv.observation_space)
        # print(dagger_trainer_kwargs["bc_trainer"].observation_space)

        # TODO: probably need to set it to this
        # venv.observation_space = dagger_trainer_kwargs["bc_trainer"].observation_space
        # expert_policy.observation_space = dagger_trainer_kwargs["bc_trainer"].observation_space

        # dagger_trainer_kwargs["bc_trainer"].policy.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=dagger_trainer_kwargs["bc_trainer"].policy.action_space.shape)
        # dagger_trainer_kwargs["bc_trainer"].action_space = spaces.Box(low=-np.inf, high=np.inf, shape=dagger_trainer_kwargs["bc_trainer"].action_space.shape)

        # avoid action cropping
        # venv.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=venv.action_space.shape)

        self.expert_policy_list = expert_policy_list
        self.deterministic_expert = deterministic_expert

        self.expert_trajs = None# expert_trajs.raw_transitions if expert_trajs != None else None
        self.vec_normalize_list = vec_normalize_list
        updated_train_env_list = []

        for train_env in train_env_list:
            if isinstance(train_env.observation_space, spaces.Dict):
                updated_train_env_list.append(None)
            else:
                # Not an Arnold environment, the obs needs to be converted
                updated_train_env_list.append(train_env)
        self.train_env_list = updated_train_env_list

    # def create_trajectory_collector(self) -> MyInteractiveTrajectoryCollector:
    #     """Create trajectory collector to extend current round's demonstration set.

    #     Returns:
    #         A collector configured with the appropriate beta, imitator policy, etc.
    #         for the current round. Refer to the documentation for
    #         `InteractiveTrajectoryCollector` to see how to use this.
    #     """

    #     def get_action(obs) :
    #         return self.bc_trainer.policy.predict(self.vec_normalize.normalize_obs(obs))[0]

    #     save_dir = self._demo_dir_path_for_round()
    #     beta = self.beta_schedule(self.round_num)
    #     collector = MyInteractiveTrajectoryCollector(
    #         venv=self.venv,
    #         get_robot_acts=get_action,
    #         beta=beta,
    #         save_dir=save_dir,
    #         rng=self.rng,
    #     )
    #     return collector

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def train(
        self,
        total_round_num: int,
        *,
        max_rollout_storage: int = 102400,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        all data collected so far.

        Args:
            total_round_num: The number of rounds to train inside the environment.
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends. Also, that any
                round will always train for at least `self.batch_size` timesteps,
                because otherwise BC could fail to receive any batches.
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.
        """
        total_timestep_count = 0
        round_num = 0

        # if self.initial_trajs != None:
        #     print("Training with initial trajectories")
        #     self.extend_and_update(self.initial_trajs, bc_train_kwargs)

        while round_num < total_round_num:
            # collector = self.create_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0
            # for expert_policy, venv, train_env, vec_normalize in zip(
            #     self.expert_policy_list,
            #     self.venv_list,
            #     self.train_env_list,
            #     self.vec_normalize_list,
            # ):
            # sample_until = rollout.make_sample_until(
            #     min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
            #     min_episodes=rollout_round_min_episodes,
            # )

            trajectories_list, info = generate_trajectoreis(
                expert_policy_list=self.expert_policy_list,
                dagger_policy=self.bc_trainer.policy,
                venv=self.venv,
                train_env_list=self.train_env_list,
                min_timesteps=rollout_round_min_timesteps,
                estimated_sample_nums=max(rollout_round_min_timesteps * len(self.expert_policy_list), self.batch_size),
                vecnormalize_list=self.vec_normalize_list,
                beta=self.beta_schedule(round_num),
                time_skip=self.time_skip,
                masking_ratio=self.bc_trainer.mask_rate,
                deterministic_expert=self.deterministic_expert
            )

            for i, (env_name, normalized_reward, unnormalized_reward, active_steps) in enumerate(zip(
                self.env_names,
                info["avg_normalized_reward_per_env"],
                info["avg_unormalized_reward_per_env"],
                info["total_active_steps_per_env"]
            )):
                self.logger.record(f"{env_name}/avg_normalized_reward_per_env_{i}", normalized_reward)
                self.logger.record(f"{env_name}/avg_unnormalized_reward_per_env_{i}", unnormalized_reward)
                self.logger.record(f"{env_name}/total_active_steps_per_env_{i}", active_steps)

            total_reward = 0
            num_trajs = 0

            for trajectories in trajectories_list :
                for traj in trajectories:
                    total_reward += np.sum(traj.rews)
                    num_trajs += 1
                    round_timestep_count += len(traj)
                    total_timestep_count += len(traj)
                round_episode_count += len(trajectories)

            self.logger.record("dagger/mean_episode_reward", total_reward / (num_trajs + 1e-9))
            self.logger.record("dagger/total_timesteps", total_timestep_count)
            self.logger.record("dagger/round_num", round_num)
            self.logger.record("dagger/round_episode_count", round_episode_count)
            self.logger.record("dagger/round_timestep_count", round_timestep_count)
            self.logger.record("dagger/beta", self.beta_schedule(round_num))
            self.logger.dump()

            # `logger.dump` is called inside BC.train within the following fn call:
            print("Extending and updating")
            self.extend_and_update(trajectories_list, max_rollout_storage, bc_train_kwargs)
            round_num += 1

    def extend_and_update(
        self,
        trajectories: Union[Sequence[types.Trajectory], DataLoader],
        max_rollout_storage: int = 102400,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        # if "log_rollouts_venv" not in user_keys:
        #     bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        logging.info("Loading demonstrations")
        # self._try_load_demos()

        num_policy = len(self.expert_policy_list)
        # if isinstance(trajectories, DataLoader):
        #     new_transitions = trajectories.dataset
        # elif isinstance(trajectories, list) and isinstance(trajectories[0], types.Trajectory):
        #     new_transitions = rollout.flatten_trajectories_with_rew(trajectories)
        # else :
        #     assert(isinstance(trajectories, types.TransitionsMinimal))
        #     new_transitions = trajectories

        assert(isinstance(trajectories, list))
        assert(isinstance(trajectories[0], list))
        assert(isinstance(trajectories[0][0], types.Trajectory))

        # when flatting, dones will be added to the end of each trajectory
        new_transitions = [rollout.flatten_trajectories_with_rew(trajs) for trajs in trajectories]

        if self.omit_history_rollout :
            self.expert_trajs = [
                types.TransitionsWithRew(
                    obs=new_transitions[i].obs,
                    acts=new_transitions[i].acts,
                    infos=new_transitions[i].infos,
                    next_obs=new_transitions[i].next_obs,
                    dones=new_transitions[i].dones,
                    rews=new_transitions[i].rews
                ) for i in range(num_policy)
            ]
        else :
            if self.expert_trajs == None:
                self.expert_trajs = new_transitions
            else:
                # assert(isinstance(self.expert_trajs, list))
                # assert(isinstance(self.expert_trajs[0], types.TransitionsWithRew))
                # if isinstance(self.expert_trajs, DataLoader):
                #     old_transitions = self.expert_trajs.dataset
                # elif isinstance(self.expert_trajs, list) and isinstance(self.expert_trajs[0], types.Trajectory):
                #     old_transitions = rollout.flatten_trajectories(self.expert_trajs)
                # else :
                #     assert(isinstance(self.expert_trajs, types.TransitionsMinimal))
                #     old_transitions = self.expert_trajs
                if isinstance(self.expert_trajs, list) and isinstance(self.expert_trajs[0], types.TransitionsWithRew) :
                    old_transitions = self.expert_trajs
                else :
                    raise NotImplementedError(f"Expert trajs type {type(self.expert_trajs)} is not supported")
                # old_transitions = [
                #     rollout.flatten_trajectories(self.expert_trajs[i]) for i in range(num_policy)
                # ]

                self.expert_trajs = []

                for i in range(num_policy):

                    old_transition_i = old_transitions[i]
                    new_transition_i = new_transitions[i]
                    obs_list = [old_transition_i.obs, new_transition_i.obs]
                    next_obs_list = [
                        old_transition_i.next_obs,
                        new_transition_i.next_obs,
                    ]
                    acts_list = [old_transition_i.acts, new_transition_i.acts]
                    dones_list = [old_transition_i.dones, new_transition_i.dones]
                    infos_list = [old_transition_i.infos, new_transition_i.infos]
                    rews_list = [old_transition_i.rews, new_transition_i.rews]

                    total_length = len(old_transition_i.acts) + len(new_transition_i.acts)
                    if total_length > max_rollout_storage :
                        self.expert_trajs.append(types.TransitionsWithRew(
                            obs=DictObs.concatenate(obs_list)[-max_rollout_storage:],
                            acts=np.concatenate(acts_list)[-max_rollout_storage:],
                            infos=np.concatenate(infos_list)[-max_rollout_storage:],
                            next_obs=DictObs.concatenate(next_obs_list)[-max_rollout_storage:],
                            dones=np.concatenate(dones_list)[-max_rollout_storage:],
                            rews=np.concatenate(rews_list)[-max_rollout_storage:]
                        ))
                    else :
                        self.expert_trajs.append(types.TransitionsWithRew(
                            obs=DictObs.concatenate(obs_list),
                            acts=np.concatenate(acts_list),
                            infos=np.concatenate(infos_list),
                            next_obs=DictObs.concatenate(next_obs_list),
                            dones=np.concatenate(dones_list),
                            rews=np.concatenate(rews_list)
                        ))
            logging.info(f"Training at round {self.round_num}")
            self.bc_trainer.set_demonstrations(self.expert_trajs)
            self.bc_trainer.train(**bc_train_kwargs)
            self.round_num += 1
            logging.info(f"New round number is {self.round_num}")
        return self.round_num
