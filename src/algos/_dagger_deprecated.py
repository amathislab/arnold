from imitation.algorithms.dagger import (
    DAggerTrainer,
    SimpleDAggerTrainer,
    InteractiveTrajectoryCollector,
    _save_dagger_demo,
)
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
import numpy as np
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
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

class MyTrajectoryAccumulator(rollout.TrajectoryAccumulator) :
    '''
    This TrajectoryAccumulator will ignore terminal observations
    '''

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

def policy_list_to_callable(
    policy_list, venv, train_env_list, vecnormalize_list, deterministic_policy=False
):
    """For N environments and M policies, each policy predicts an action for M/N environments.
    We also need to convert the actions to the format expected by the environment."""
    num_envs = venv.num_envs
    envs_per_policy = num_envs // len(policy_list)

    def get_actions(obs, obs_dicts, states, episode_starts):
        actions_list = []
        states_list = []
        for i, policy in enumerate(policy_list):
            train_env_i = train_env_list[i]
            vecnormalize_i = vecnormalize_list[i]
            if train_env_i is None:
                obs_i = {
                    key: obs[key][i * envs_per_policy : (i + 1) * envs_per_policy]
                    for key in obs
                }
                obs_i_normalized = vecnormalize_i.normalize_single_obs_dict(obs_i, env_idx=i)

            else:
                obs_i_list = []
                for d in obs_dicts[i * envs_per_policy : (i + 1) * envs_per_policy]:
                    obs_train_style = np.zeros(0)
                    for key in train_env_i.obs_keys:
                        obs_train_style = np.concatenate(
                            [obs_train_style, d[key].ravel()]
                        )
                    obs_i_list.append(obs_train_style)
                obs_i = np.stack(obs_i_list)
                obs_i_normalized = vecnormalize_i.normalize_obs(obs_i)
            
            if states is None :
                states_i = None
            else :
                states_i = states[i]
            episode_starts_i = episode_starts[
                i * envs_per_policy : (i + 1) * envs_per_policy
            ]
            # import ipdb
            # ipdb.set_trace()
            action, states_i_new = policy.predict(
                obs_i_normalized,
                state=states_i,
                episode_start=episode_starts_i,
                deterministic=deterministic_policy,
            )
            actions_list.append(action)
            states_list.append(states_i_new)
        
        return np.concatenate(actions_list), states_list

    return get_actions

def beta_mix_action(expert_actions, policy_actions, beta) :

    dist = np.random.uniform(low=0.0, high=1.0, size=(expert_actions.shape[0],))
    sel = dist < beta
    sel = sel[:, None]
    return expert_actions * sel + policy_actions * (1-sel)

def generate_trajectoreis(
    expert_policy_list,
    dagger_policy,
    venv,
    train_env_list,
    sample_until,
    estimated_sample_nums,
    vecnormalize_list,
    beta = 0,
):

    get_expert_action = policy_list_to_callable(
        expert_policy_list, venv, train_env_list, vecnormalize_list
    )
    get_dagger_action = rollout.policy_to_callable(dagger_policy, venv)

    trajectories = []
    traj_accum = MyTrajectoryAccumulator()  # Use a customized accumulator to prevent "terminal_obs" errors
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

    with tqdm.tqdm(total=estimated_sample_nums, desc="Generating new trajectories") as pbar :
        while np.any(active):

            step += 1
            obs_dicts = venv.get_attr("obs_dict")
            expert_actions, expert_states = get_expert_action(
                obs, obs_dicts, expert_states, dones
            )

            dagger_action, dagger_state = get_dagger_action(obs, dagger_state, dones)

            mixed_action = beta_mix_action(expert_actions, dagger_action, beta)
            obs, rews, dones, _ = venv.step(mixed_action)

            assert isinstance(obs, (np.ndarray, dict))
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            dones &= active

            new_trajectories = traj_accum.add_steps_and_auto_finish(
                expert_actions, wrapped_obs, rews, dones, [{} for _ in range(venv.num_envs)]
            )
            import ipdb
            ipdb.set_trace()
            trajectories.extend(new_trajectories)

            if sample_until(trajectories):
                active &= ~dones
            
            pbar.update(sum(len(t.obs) - 1 for t in new_trajectories))

    return trajectories


class MultiEnvDAggerTrainer(base.BaseImitationAlgorithm):

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        train_env_list: Env,
        scratch_dir: types.AnyPath,
        rng: np.random.Generator,
        beta_schedule: Optional[Callable[[int], float]] = None,
        omit_history_rollout: bool = False,
        expert_policy_list: policies.BasePolicy,
        bc_trainer: bc.BC,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        vec_normalize_list: Optional[vec_env.VecNormalize] = None,
    ):

        super().__init__(custom_logger=custom_logger)

        if beta_schedule is None:
            beta_schedule = 15
        self.beta_schedule = LinearBetaSchedule(beta_schedule)
        self.scratch_dir = util.parse_path(scratch_dir)
        self.venv = venv
        self.round_num = 0
        self.omit_history_rollout = omit_history_rollout
        self._last_loaded_round = -1
        self._all_demos = []
        self.rng = rng

        self.bc_trainer = bc_trainer
        self.bc_trainer.logger = self.logger

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

        self.expert_trajs = expert_trajs
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
            sample_until = rollout.make_sample_until(
                min_timesteps=max(rollout_round_min_timesteps, self.batch_size),
                min_episodes=rollout_round_min_episodes,
            )

            trajectories = generate_trajectoreis(
                expert_policy_list=self.expert_policy_list,
                dagger_policy=self.bc_trainer.policy,
                venv=self.venv,
                train_env_list=self.train_env_list,
                sample_until=sample_until,
                estimated_sample_nums=max(rollout_round_min_timesteps, self.batch_size),
                vecnormalize_list=self.vec_normalize_list,
                beta=self.beta_schedule(round_num)
            )

            total_reward = 0
            num_trajs = 0
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
            self.extend_and_update(trajectories, max_rollout_storage, bc_train_kwargs)
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
        
        if isinstance(trajectories, DataLoader):
            new_flattened_trajectories = trajectories.dataset
        elif isinstance(trajectories, list) and isinstance(trajectories[0], types.Trajectory):
            new_flattened_trajectories = rollout.flatten_trajectories(trajectories)
        else :
            assert(isinstance(trajectories, types.Transitions))
            new_flattened_trajectories = trajectories

        if self.omit_history_rollout :
            self.expert_trajs = types.Transitions(
                obs=new_flattened_trajectories.obs,
                acts=new_flattened_trajectories.acts,
                infos=new_flattened_trajectories.infos,
                next_obs=new_flattened_trajectories.next_obs,
                dones=new_flattened_trajectories.dones,
            )
        else :
            if self.expert_trajs == None:
                self.expert_trajs = new_flattened_trajectories
            else:
                if isinstance(self.expert_trajs, DataLoader):
                    old_flattened_trajectories = self.expert_trajs.dataset
                elif isinstance(self.expert_trajs, list) and isinstance(self.expert_trajs[0], types.Trajectory):
                    old_flattened_trajectories = rollout.flatten_trajectories(self.expert_trajs)
                else :
                    assert(isinstance(self.expert_trajs, types.Transitions))
                    old_flattened_trajectories = self.expert_trajs
            
                obs_list = [old_flattened_trajectories.obs, new_flattened_trajectories.obs]
                next_obs_list = [
                    old_flattened_trajectories.next_obs,
                    new_flattened_trajectories.next_obs,
                ]
                acts_list = [old_flattened_trajectories.acts, new_flattened_trajectories.acts]
                dones_list = [old_flattened_trajectories.dones, new_flattened_trajectories.dones]
                infos_list = [old_flattened_trajectories.infos, new_flattened_trajectories.infos]
                
                total_length = len(old_flattened_trajectories.acts) + len(new_flattened_trajectories.acts)
                if total_length > max_rollout_storage :
                    raw_index = np.arange(total_length, dtype=np.int32)
                    sel_index = np.random.choice(raw_index, max_rollout_storage, replace=False)
                    self.expert_trajs = types.Transitions(
                        obs=DictObs.concatenate(obs_list)[sel_index],
                        acts=np.concatenate(acts_list)[sel_index],
                        infos=np.concatenate(infos_list)[sel_index],
                        next_obs=DictObs.concatenate(next_obs_list)[sel_index],
                        dones=np.concatenate(dones_list)[sel_index],
                    )
                else :
                    self.expert_trajs = types.Transitions(
                        obs=DictObs.concatenate(obs_list),
                        acts=np.concatenate(acts_list),
                        infos=np.concatenate(infos_list),
                        next_obs=DictObs.concatenate(next_obs_list),
                        dones=np.concatenate(dones_list),
                    )
            logging.info(f"Training at round {self.round_num}")
            self.bc_trainer.set_demonstrations(self.expert_trajs)
            self.bc_trainer.train(**bc_train_kwargs)
            self.round_num += 1
            logging.info(f"New round number is {self.round_num}")
        return self.round_num
