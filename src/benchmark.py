import numpy as np
import tqdm
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from models.ppo.helpers import call_muscle_transformer_policy
from algos.bc_ppo import MultiTaskBCPPO
from models.ppo.policies import (
    MuscleTransformerPolicy,
    PredictiveMuscleTransformerPolicy,
    BilateralMuscleTransformerPolicy,
)
from definitions import ROOT_DIR, ENV_CONFIG_PATH
from stable_baselines3 import SAC, PPO
from models.ppo.helpers import call_sb3_policy
from algos.dagger_bilateral import bilateral_policy_to_callable, add_timestep_to_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from envs.utilities import create_vec_env
import torch
import argparse
import os
from envs.environment_factory import EnvironmentFactory
import json
from envs.expert_wrapper import ExpertWrapper
from envs.loaders import load_expert_policy_and_env
from models.csi_model import CSIActionNet
from evaluation import find_vecnormalize, get_episode_horizon, summarize_episodes
from stable_baselines3.common.utils import set_random_seed
from vocabulary import set_vocabulary_mode

def get_training_args(policy_path):
    args_path = os.path.join(os.path.dirname(policy_path), "args.json")
    if not os.path.isfile(args_path):
        return {}
    with open(args_path) as stream:
        return json.load(stream)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark",
        description="Benchmarking the performance of a single policy on multiple environments",
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        type=str,
        default="transformer",
        help="Policy network to use, transformer or predictive_transformer",
        choices=[
            "transformer",
            "predictive_transformer",
            "bilateral_transformer",
            "recurrent",
            "mlp",
            "csi",
            "sac",
        ],
    )
    parser.add_argument(
        "--task",
        nargs="+",
        help="Tasks to benchmark. If not provided, will read from args.json for multi-task policies",
        default=None,
        choices=[
            "baoding_p1_ccw",
            "baoding_p1_cw",
            "baoding_p2",
            "baoding_p2_overlap",
            "hand_thumb_reach",
            "hand_index_reach",
            "hand_middle_reach",
            "hand_ring_reach",
            "hand_little_reach",
            "hand_pose",
            "hand_reach",
            "pen",
            "relocate",
            "reorient",
            "elbow_joint_pose",
            "elbow_pose",
            "finger_pose",
            "kinesis",
        ],
    )
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Number of steps per episode"
    )
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.5,
        help="Masking observations and actions with a specific rate for bilateral transformer",
    )
    parser.add_argument(
        "--time_skip",
        type=int,
        default=1,
        help="Time to skip observations for bilateral transformer",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_failed_video", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--expert_stochastic", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--arnold", action="store_true")
    parser.add_argument("--csi_components", type=str, default=None)
    parser.add_argument("--csi_subspace", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save benchmark results to JSON file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/benchmarks/student_policies",
        help="Directory to save benchmark results",
    )
    parser.add_argument("--out_file", default=None, help="Result JSON filename")
    parser.add_argument(
        "--custom_experts",
        type=str,
        default=None,
        help="Path to custom expert config file",
    )
    return parser.parse_args()


def save_results(scores, args):
    if not args.save_results:
        return

    out_dir = os.path.join(ROOT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.load:
        policy_dir = os.path.dirname(args.load)
        policy_name = os.path.basename(policy_dir)
        checkpoint_name = os.path.basename(args.load).replace(".zip", "")
        run_name = f"{policy_name}_{checkpoint_name}"
        if args.csi_subspace is not None:
            run_name += f"_csi_{args.csi_subspace}"
    else:
        run_name = f"expert_{'_'.join(args.task)}"

    results_file = os.path.join(out_dir, args.out_file or f"{run_name}_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def save_video(frames, out_dir, task_name):
    out_dir = os.path.join(ROOT_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print("Saving video to", os.path.join(out_dir, f"{task_name}.mp4"))
    import cv2

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(out_dir, f"{task_name}.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"),
        30,
        (width, height),
    )
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()


if __name__ == "__main__":
    args = parse_args()

    training_args = get_training_args(args.load) if args.load is not None else {}
    num_memory_steps = training_args.get("num_memory_steps")
    if args.task is None:
        args.task = list(dict.fromkeys(training_args.get("tasks", [])))
    if not args.task:
        raise ValueError("Specify --task or load a checkpoint with tasks in args.json")

    vocabulary = None
    if args.arnold and args.load is not None:
        vocabulary_path = os.path.join(os.path.dirname(args.load), "vocabulary.json")
        if os.path.isfile(vocabulary_path):
            with open(vocabulary_path) as stream:
                vocabulary = json.load(stream)
            mode = "atomic" if any("/" in key for key in vocabulary) else "compositional"
            set_vocabulary_mode(mode, vocabulary_path)

    scores = {}
    arnold_envs = args.arnold

    for task_name in args.task:
        vecnormalize = None
        set_random_seed(args.seed)
        if args.expert:
            if "kinesis" in task_name:
                eval_env_config_path = os.path.join(
                    ENV_CONFIG_PATH, f"{task_name}_config.json"
                )
                with open(eval_env_config_path, "r") as f:
                    eval_env_config = json.load(f)
                env = EnvironmentFactory.create(
                    eval_env_config["env_name"], headless=not args.render
                )
            else:
                # Use the simplified expert loading approach
                if args.custom_experts is not None:
                    with open(args.custom_experts, "r") as f:
                        custom_expert_config = json.load(f)
                else:
                    custom_expert_config = None
                policy, env, vecnormalize, _ = load_expert_policy_and_env(
                    task_name, device=args.device, custom_expert_config_dict=custom_expert_config
                )
            env = ExpertWrapper(env, task_name, device=args.device, custom_expert_config_path=args.custom_experts)
            vecnormalize = None
        else:
            # Load student policy
            if args.policy == "transformer":
                policy_class = MuscleTransformerPolicy
            elif args.policy == "predictive_transformer":
                policy_class = PredictiveMuscleTransformerPolicy
            elif args.policy == "bilateral_transformer":
                policy_class = BilateralMuscleTransformerPolicy
            elif args.policy == "recurrent":
                policy_class = RecurrentActorCriticPolicy
            elif args.policy == "mlp":
                policy_class = ActorCriticPolicy
            elif args.policy == "csi":
                policy_class = CSIActionNet
            elif args.policy == "sac":
                policy_class = None  # loaded directly below
            else:
                raise NotImplementedError(f"Policy type {args.policy} is not supported")

            custom_objects = {"vocabulary": vocabulary}

            if args.policy == "sac":
                policy = SAC.load(args.load, device=args.device, buffer_size=1).policy
                policy.eval()
            else:
                try:
                    policy = policy_class.load(args.load, device=args.device)
                except Exception:
                    if args.policy == "csi":
                        algo_class = PPO
                    else:
                        algo_class = MultiTaskBCPPO

                    policy = algo_class.load(
                        args.load, device=args.device, custom_objects=custom_objects
                    ).policy
                policy.eval()
            if args.csi_subspace is not None:
                assert args.policy == "csi"
                policy.change_projection(subspace=args.csi_subspace, trainable=False)
            if args.csi_components is not None:
                assert args.policy == "csi"
                csi_projection = np.load(args.csi_components)
                csi_mean = np.load(
                    args.csi_components.replace("subspace.npy", "mean.npy")
                )
                csi_projection = torch.from_numpy(csi_projection)
                csi_mean = torch.from_numpy(csi_mean)
                policy.change_projection(csi_projection, csi_mean, trainable=False)
            policy.to(args.device)

            # Load environment for student policy
            prefix = "arnold_" if arnold_envs else "dense_"
            if arnold_envs and training_args.get("dense_reward", False):
                prefix = "dense_" + prefix
            eval_env_config_path = os.path.join(
                ENV_CONFIG_PATH, f"{prefix}{task_name}_config.json"
            )
            with open(eval_env_config_path, "r") as f:
                eval_env_config = json.load(f)
            if num_memory_steps is not None:
                eval_env_config["num_memory_steps"] = num_memory_steps
            if task_name == "kinesis":
                eval_env_config["headless"] = not args.render

            if args.normalize:
                vecnormalize_path = find_vecnormalize(args.load)
                print("Loading vecnormalize from", vecnormalize_path)
                vecnormalize = create_vec_env(
                    env_config_list=[eval_env_config],
                    load_env_path=vecnormalize_path,
                    multi_env=args.arnold,
                    old_vocabulary=vocabulary,
                    seed=args.seed,
                )
                vecnormalize.training = False
                vecnormalize.norm_reward = False
            else:
                vecnormalize = None

            env = EnvironmentFactory.create(**eval_env_config)
            if args.policy != "sac":
                policy.observation_space = env.observation_space

            if task_name == "kinesis":
                env.env.env.gym_env.env.render_mode = "rgb_array" if args.save_video else "human"

        if args.render:
            env.mujoco_render_frames = True
        if args.save_video:
            env.mujoco_render_frames = False
            frames = []

        task_index = None
        if not args.expert:
            extractor = getattr(policy, "features_extractor", None)
            if getattr(extractor, "position_embedding", None) == "task_specific":
                task_index = extractor.task_names.index(task_name)
        set_random_seed(args.seed)
        env.action_space.seed(args.seed)

        # Lists to store per-episode metrics for std calculation
        episode_cum_rewards = []
        episode_solved_steps = []  # Will store raw count of solved steps per episode
        episode_steps = []
        frames = []

        max_episode_steps = get_episode_horizon(env)


        with tqdm.tqdm(total=args.num_episodes, desc=f"Evaluating {task_name}") as pbar:
            for i in range(args.num_episodes):
                lstm_states = None
                cum_rew = 0
                step = 0
                obs = env.reset(seed=args.seed) if i == 0 else env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                episode_starts = np.ones((1,), dtype=bool)
                done = False
                solved_count = 0
                episode_frames = []

                while not done:
                    if args.render:
                        if task_name != "kinesis":
                            env.sim.renderer.render_to_window()
                        else:
                            env.render()
                    if args.save_video:
                        if task_name != "kinesis":
                            curr_frame = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1, device_id=0)
                        else:
                            curr_frame = env.render()
                        episode_frames.append(curr_frame)

                    # Get action based on policy type
                    if args.expert:
                        action = env.get_expert_action(deterministic=not args.expert_stochastic)
                    else:
                        if arnold_envs:
                            obs_i = {key: obs[key][None, ...] for key in obs}
                            if args.normalize:
                                obs_i_normalized = (
                                    vecnormalize.normalize_single_obs_dict(
                                        obs_i, env_idx=0
                                    )
                                )
                            else:
                                obs_i_normalized = obs_i
                        else:
                            if vecnormalize:
                                obs_i_normalized = vecnormalize.normalize_obs(obs)
                            else:
                                obs_i_normalized = obs

                        if task_index is not None:
                            obs_i_normalized["env_id"] = np.array([[[task_index]]], dtype=np.int32)

                        # Get action based on policy type
                        if isinstance(policy, BilateralMuscleTransformerPolicy):
                            obs_i_normalized = add_timestep_to_obs(
                                obs_i_normalized, np.ones((1,)) * step
                            )
                            action, lstm_states = bilateral_policy_to_callable(
                                policy,
                                env,
                                masking_ratio=args.mask_rate,
                                time_skip=args.time_skip,
                                deterministic_policy=args.deterministic,
                            )(obs_i_normalized, lstm_states, episode_starts)
                        elif isinstance(
                            policy,
                            (
                                MuscleTransformerPolicy,
                                PredictiveMuscleTransformerPolicy,
                            ),
                        ):
                            action, value = call_muscle_transformer_policy(
                                policy, obs_i_normalized, args.deterministic
                            )
                        elif isinstance(policy, RecurrentActorCriticPolicy):
                            action, value, lstm_states = call_sb3_policy(
                                policy,
                                obs_i_normalized,
                                lstm_states,
                                episode_starts,
                                args.deterministic,
                            )
                        elif args.policy == "sac":
                            # Pad each obs key to the training obs space shape so
                            # policy.predict()'s shape validation doesn't reject
                            # single-task obs (fewer tokens than the multi-task max).
                            obs_sac = {}
                            for key, obs_arr in obs_i_normalized.items():
                                target_shape = policy.observation_space[key].shape
                                if obs_arr.shape[1:] != target_shape:
                                    padded = np.zeros(
                                        (obs_arr.shape[0], *target_shape),
                                        dtype=obs_arr.dtype,
                                    )
                                    src = tuple(slice(0, s) for s in obs_arr.shape[1:])
                                    padded[(slice(None),) + src] = obs_arr
                                    obs_sac[key] = padded
                                else:
                                    obs_sac[key] = obs_arr
                            action, value = policy.predict(
                                obs_sac, deterministic=args.deterministic
                            )
                            # SAC is trained on the merged multi-task action space;
                            # trim to the eval env's action dim (same logic as
                            # PaddedActionWrapper used during training).
                            action = action[..., : env.action_space.shape[0]]
                        else:
                            action, value = policy.predict(
                                obs_i_normalized, deterministic=args.deterministic
                            )

                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    action = np.squeeze(action)
                    next_obs, rewards, term, trunc, info = env.step(action)

                    done = term or trunc or (args.num_steps is not None and step + 1 >= args.num_steps)

                    obs = next_obs
                    episode_starts = np.array([done])
                    cum_rew += rewards
                    step += 1
                    solved = 1.0 * info["rwd_dict"]["solved"]
                    solved_count += solved

                pbar.update(1)

                # Store per-episode metrics
                episode_cum_rewards.append(cum_rew)
                episode_solved_steps.append(solved_count)  # Store raw count
                episode_steps.append(step)

                if args.save_failed_video :
                    if solved_count == 0:
                        frames.append(episode_frames)
                else :
                    frames += episode_frames

        scores[task_name] = summarize_episodes({
            "cum_rewards": episode_cum_rewards,
            "steps": episode_steps,
            "solved_counts": episode_solved_steps,
        }, max_episode_steps)
        scores[task_name].update(
            seed=args.seed,
            deterministic=not args.expert_stochastic if args.expert else args.deterministic,
            normalize_obs=bool(getattr(env.expert_vecnormalize, "norm_obs", False))
                if args.expert and not env.using_kinesis_default_expert
                else (True if args.expert else bool(getattr(vecnormalize, "norm_obs", False))),
            num_steps=args.num_steps,
        )
        env.close()
        if vecnormalize is not None:
            vecnormalize.close()
        if args.save_video:
            if args.save_failed_video :
                for i, episode_frames in enumerate(frames) :
                    save_video(episode_frames, args.out_dir, task_name+"_"+str(i))
            else :
                save_video(frames, args.out_dir, task_name)
    print(json.dumps(scores, indent=2))

    # Save results if requested
    save_results(scores, args)
