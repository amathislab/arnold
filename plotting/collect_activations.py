import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed

from algos.bc_ppo import MultiTaskBCPPO
from analysis.signals import initialize_file, write_episode
from definitions import ENV_CONFIG_PATH
from envs.environment_factory import EnvironmentFactory
from envs.expert_wrapper import ExpertWrapper
from envs.utilities import create_vec_env
from evaluation import find_vecnormalize, get_episode_horizon
from models.ppo.policies import MuscleTransformerPolicy, to_tensor_dict
from vocabulary import set_vocabulary_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Collect policy rollout signals or transformer activations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--load", type=Path)
    source.add_argument("--expert", action="store_true")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_success", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--arnold", action="store_true")
    parser.add_argument("--custom_experts", default=None)
    parser.add_argument("--signals_only", action="store_true")
    parser.add_argument("--physical_signals", action="store_true")
    parser.add_argument("--policy_id", default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("data/activations"))
    return parser.parse_args()


def transformer_distribution(policy, observation):
    tensors = to_tensor_dict(observation, device=policy.device)
    features = policy.extract_features(tensors)
    encodings, mask, action_target, action_mask, _ = features
    encoded = policy.mlp_extractor.encoder(encodings, src_key_padding_mask=mask)
    decoded = policy.mlp_extractor.action_decoder(
        action_target, encoded, tgt_key_padding_mask=action_mask, memory_key_padding_mask=mask)
    return policy._get_action_dist_from_latent(decoded), encoded, decoded


def policy_action(policy, observation, states, episode_start, deterministic, keep_hidden=False):
    hidden = {}
    if isinstance(policy, MuscleTransformerPolicy):
        distribution, encoded, decoded = transformer_distribution(policy, observation)
        if keep_hidden:
            hidden = {"encoder_output": encoded.cpu().numpy(), "decoder_output": decoded.cpu().numpy()}
    else:
        tensors, _ = policy.obs_to_tensor(observation)
        if isinstance(policy, RecurrentActorCriticPolicy):
            if states is None:
                states = tuple(torch.zeros(policy.lstm_hidden_state_shape, device=policy.device) for _ in range(2))
            starts = torch.tensor([episode_start], dtype=torch.float32, device=policy.device)
            distribution, states = policy.get_distribution(tensors, states, starts)
        else:
            distribution = policy.get_distribution(tensors)
    action = distribution.get_actions(deterministic=deterministic).cpu().numpy().reshape(-1)
    mean = distribution.distribution.mean.cpu().numpy().reshape(-1)
    return action, mean, states, hidden


def normalize_observation(obs, normalizer, arnold, task_index=None):
    if arnold:
        obs = {key: value[None, ...] for key, value in obs.items()}
        if normalizer is not None:
            obs = normalizer.normalize_single_obs_dict(obs, env_idx=0)
        if task_index is not None:
            obs["env_id"] = np.array([[[task_index]]], dtype=np.int32)
    elif normalizer is not None:
        obs = normalizer.normalize_obs(obs)
    return obs


def simulator(env):
    base = env.unwrapped
    return (base.mj_model, base.mj_data) if hasattr(base, "mj_data") else (base.sim.model, base.sim.data)


def collect_episode_data(env, policy, vecnormalize=None, device="cuda", *,
                         seed=None, deterministic=False, signals_only=False,
                         physical_signals=False, expert=False, task_index=None, num_steps=None):
    obs, _ = env.reset(seed=seed)
    model, data = simulator(env)
    joint_indices = [model.jnt_qposadr[model.joint_name2id(name) if hasattr(model, "joint_name2id") else model.joint(name).id] for name in env.joint_names]
    episode = {key: [] for key in ("actions", "action_means", "joint_positions", "rewards", "solved")}
    if physical_signals:
        episode.update(muscle_controls=[], muscle_activations=[])
    if not signals_only:
        episode["observations"] = []
    states = None
    done = False
    while not done:
        episode["joint_positions"].append(data.qpos[joint_indices].copy())
        if expert:
            wrapper = env
            _, obs_vec = wrapper.expert_env.unwrapped.obsdict2obsvec(wrapper.unwrapped.obs_dict, wrapper.expert_env.unwrapped.obs_keys)
            obs_vec = obs_vec.astype(np.float32)
            if wrapper.arnold_policy:
                history = wrapper.expert_env.create_history_obs(obs_vec)
                observation = wrapper.expert_vecnormalize.normalize_single_obs_dict(history, wrapper.env_idx)
            else:
                observation = wrapper.expert_vecnormalize.normalize_obs(obs_vec)
        else:
            observation = normalize_observation(obs, vecnormalize, isinstance(policy, MuscleTransformerPolicy), task_index)
        with torch.no_grad():
            action, mean, states, hidden = policy_action(policy, observation, states, not episode["actions"], deterministic, keep_hidden=not signals_only)
        if expert and not isinstance(policy, MuscleTransformerPolicy):
            action = np.clip(action, policy.action_space.low, policy.action_space.high)
        if not signals_only:
            episode["observations"].append(obs)
            for name, value in hidden.items():
                episode.setdefault(name, []).append(value)
        obs, reward, terminated, truncated, info = env.step(action)
        stored_mean = mean[None, :] if not signals_only and isinstance(policy, MuscleTransformerPolicy) else mean
        for name, value in (("actions", action), ("action_means", stored_mean), ("rewards", reward),
                            ("solved", float(info["rwd_dict"]["solved"]))):
            episode[name].append(value)
        if physical_signals:
            episode["muscle_controls"].append(data.ctrl.copy())
            episode["muscle_activations"].append(data.act.copy())
        done = terminated or truncated or (num_steps is not None and len(episode["actions"]) >= num_steps)
    return {key: np.asarray(value) for key, value in episode.items()}


def save_episode_data(episode_data, filepath):
    with h5py.File(filepath, "x") as file:
        file.attrs.update(episode_length=len(episode_data["rewards"]),
                          total_reward=np.sum(episode_data["rewards"]),
                          solved_steps=np.sum(episode_data["solved"]))
        for name, values in episode_data.items():
            if name == "observations" and isinstance(values[0], dict):
                group = file.create_group(name)
                for key in values[0]:
                    group.create_dataset(key, data=np.asarray([obs[key] for obs in values]), compression="gzip", shuffle=True)
            else:
                file.create_dataset(name, data=values, compression="gzip", shuffle=True)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    set_vocabulary_mode("compositional")
    training_args, vocabulary = {}, None
    if args.load:
        args_path = args.load.parent / "args.json"
        if args_path.is_file():
            training_args = json.loads(args_path.read_text())
        vocabulary_path = args.load.parent / "vocabulary.json"
        if args.arnold and vocabulary_path.is_file():
            vocabulary = json.loads(vocabulary_path.read_text())
            set_vocabulary_mode("atomic" if any("/" in key for key in vocabulary) else "compositional", str(vocabulary_path))
    prefix = "arnold_" if args.arnold or args.expert else ""
    if args.arnold and training_args.get("dense_reward", False):
        prefix = "dense_" + prefix
    config_path = Path(ENV_CONFIG_PATH) / f"{prefix}{args.task}_config.json"
    config = json.loads(config_path.read_text())
    if "num_memory_steps" in training_args:
        config["num_memory_steps"] = training_args["num_memory_steps"]
    normalizer = None
    env = EnvironmentFactory.create(**config)
    if args.expert:
        env = ExpertWrapper(env, args.task, device=args.device, custom_expert_config_path=args.custom_experts)
        policy = env.expert_policy
        policy_id = args.policy_id or "expert"
    else:
        policy_class = MuscleTransformerPolicy if args.arnold else ActorCriticPolicy
        try:
            policy = policy_class.load(str(args.load), device=args.device)
        except Exception:
            policy = MultiTaskBCPPO.load(str(args.load), device=args.device, custom_objects={"vocabulary": vocabulary}).policy
        policy.observation_space = env.observation_space
        policy_id = args.policy_id or f"{args.load.parent.parent.name}/{args.load.parent.name}/{args.load.stem}"
        if args.normalize:
            normalizer = create_vec_env(env_config_list=[config], load_env_path=find_vecnormalize(str(args.load)),
                                        multi_env=args.arnold, old_vocabulary=vocabulary, seed=args.seed)
            normalizer.training = False
            normalizer.norm_reward = False
    policy.to(args.device)
    policy.set_training_mode(False)
    task_index = None
    extractor = getattr(policy, "features_extractor", None)
    if getattr(extractor, "position_embedding", None) == "task_specific":
        task_index = extractor.task_names.index(args.task)
    axes = {"joint_names": list(env.joint_names), "muscle_names": list(env.muscle_names)}
    if args.physical_signals:
        model, _ = simulator(env)
        axes["activation_names"] = [name for i, name in enumerate(axes["muscle_names"]) if model.actuator_actadr[i] >= 0]
    metadata = {"policy_id": policy_id, "task": args.task, "seed": args.seed,
                "deterministic": args.deterministic, "horizon": get_episode_horizon(env), "dt": env.dt,
                "normalize_obs": bool(getattr(env.expert_vecnormalize if args.expert else normalizer, "norm_obs", False)),
                "actions_clipped": args.expert and not isinstance(policy, MuscleTransformerPolicy), "means_clipped": False}
    out_dir = args.out_dir / policy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.num_success if args.num_success is not None else args.num_episodes
    attempts = args.max_attempts or (10 * target if args.num_success is not None else target)
    file = None
    saved = 0
    try:
        if args.signals_only:
            file = h5py.File(out_dir / f"{args.task}.h5", "x")
            initialize_file(file, metadata, axes)
            file.attrs["success_only"] = args.num_success is not None
        with tqdm.tqdm(total=target, desc=args.task) as progress:
            for episode in range(attempts):
                set_random_seed(args.seed + episode)
                signals = collect_episode_data(env, policy, normalizer, args.device, seed=args.seed + episode,
                    deterministic=args.deterministic, signals_only=args.signals_only,
                    physical_signals=args.physical_signals, expert=args.expert,
                    task_index=task_index, num_steps=args.num_steps)
                if args.num_success is not None and not np.any(signals["solved"]):
                    continue
                if file is not None:
                    write_episode(file, episode, signals)
                else:
                    save_episode_data(signals, out_dir / f"{args.task}_episode_{episode}.h5")
                saved += 1
                progress.update(1)
                if saved == target:
                    break
        if saved < target:
            raise RuntimeError(f"Collected {saved}/{target} episodes in {attempts} attempts")
    finally:
        if file is not None:
            file.close()
        env.close()
        if normalizer is not None:
            normalizer.close()



if __name__ == "__main__":
    main()
