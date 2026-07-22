import numpy as np
import tqdm
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from train.algo_factory import AlgoFactory
from models.ppo.helpers import call_muscle_transformer_policy
from algos.bc_ppo import MultiTaskBCPPO
from models.ppo.policies import (
    MuscleTransformerPolicy,
    PredictiveMuscleTransformerPolicy,
    BilateralMuscleTransformerPolicy,
)
from definitions import ROOT_DIR, ENV_CONFIG_PATH
from stable_baselines3 import SAC, TD3, PPO
from models.ppo.helpers import call_sb3_policy
from algos.dagger_bilateral import bilateral_policy_to_callable, add_timestep_to_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from envs.utilities import create_vec_env
import torch
import argparse
import os
from envs.environment_factory import EnvironmentFactory
import time
import json
import torch
from definitions import (
    ENV_CONFIG_PATH,
)
from envs.expert_wrapper import ExpertWrapper
from envs.loaders import load_expert_policy_and_env
from models.csi_model import CSIActionNet

def get_tasks_from_args_json(policy_path):
    """Get list of tasks from args.json in the policy directory"""
    policy_dir = os.path.dirname(policy_path)
    args_path = os.path.join(policy_dir, "args.json")

    if not os.path.exists(args_path):
        return None, None

    with open(args_path, "r") as f:
        args = json.load(f)
        return list(set(args.get("tasks", None))), args.get("num_memory_steps", None)


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
    parser.add_argument("--num_episodes", type=int, default=20)
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

    # Add num_episodes to scores dict
    for task in scores:
        scores[task]["num_episodes"] = args.num_episodes

    results_file = os.path.join(out_dir, f"{run_name}_results.json")
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

    # For multi-task policies, get tasks from args.json if not specified
    if args.task is None and args.load is not None:
        args.task, num_memory_steps = get_tasks_from_args_json(args.load)
        if args.task is None:
            raise ValueError(
                "No tasks specified and couldn't find args.json with tasks"
            )
    elif args.task is None:
        raise ValueError("Must specify --task when not loading a policy")
    else:
        num_memory_steps = None

    scores = {}
    arnold_envs = args.arnold

    for task_name in args.task:
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
            env = ExpertWrapper(env, task_name, custom_expert_config_path=args.custom_experts)
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

            if args.arnold:
                experiment_path = os.path.dirname(args.load)
                vocabulary_path = os.path.join(experiment_path, "vocabulary.json")
                if os.path.exists(vocabulary_path):
                    with open(vocabulary_path, "r") as f:
                        vocabulary = json.load(f)
                else:
                    vocabulary = None
                custom_objects = {"vocabulary": vocabulary}
            else:
                custom_objects = {}
                vocabulary = None

            if args.policy == "sac":
                policy = SAC.load(args.load, device=args.device).policy
                policy.eval()
            else:
                try:
                    policy = policy_class.load(args.load, device=args.device)
                except:
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
                vecnormalize_path = args.load.replace(
                    "model", "model_vecnormalize"
                ).replace(".zip", ".pkl")
                if not os.path.exists(vecnormalize_path):
                    vecnormalize_path = args.load.replace(
                        "model", "env"
                    ).replace(".zip", ".pkl")
                print("Loading vecnormalize from", vecnormalize_path)
                vecnormalize = create_vec_env(
                    env_config_list=[eval_env_config],
                    load_env_path=vecnormalize_path,
                    multi_env=args.arnold,
                    old_vocabulary=vocabulary,
                )
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

        total_steps = 0
        total_solved = 0
        total_solved_steps = 0
        total_cum_reward = 0

        # Lists to store per-episode metrics for std calculation
        episode_cum_rewards = []
        episode_step_rewards = []
        episode_solved = []
        episode_solved_steps = []  # Will store raw count of solved steps per episode
        episode_solved_fracs = []  # Will store solved fraction per episode
        episode_steps = []
        frames = []

        max_episode_steps = env.unwrapped.env.spec.max_episode_steps


        with tqdm.tqdm(total=args.num_episodes, desc=f"Evaluating {task_name}") as pbar:
            for i in range(args.num_episodes):
                lstm_states = None
                cum_rew = 0
                step = 0
                obs = env.reset()
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
                        # time.sleep(0.001)
                    if args.save_video:
                        if task_name != "kinesis":
                            curr_frame = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1, device_id=0)
                        else:
                            curr_frame = env.render()
                        episode_frames.append(curr_frame)

                    # Get action based on policy type
                    if args.expert:
                        action = env.get_expert_action()
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
                                deterministic_policy=True,
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

                    if args.num_steps is not None:
                        done = step >= args.num_steps - 1
                    else:
                        done = term or trunc

                    obs = next_obs
                    episode_starts = np.array([done])
                    cum_rew += rewards
                    step += 1
                    solved = 1.0 * info["rwd_dict"]["solved"]
                    solved_count += solved

                pbar.update(1)
                total_steps += step
                total_solved += solved_count > 0
                total_solved_steps += solved_count
                total_cum_reward += cum_rew

                # Store per-episode metrics
                episode_cum_rewards.append(cum_rew)
                episode_step_rewards.append(cum_rew / step)
                episode_solved.append(1.0 if solved_count > 0 else 0.0)
                episode_solved_steps.append(solved_count)  # Store raw count
                episode_solved_fracs.append(
                    solved_count / max_episode_steps if max_episode_steps else 0
                )  # Store fraction
                episode_steps.append(step)

                if args.save_failed_video :
                    if solved_count == 0:
                        frames.append(episode_frames)
                else :
                    frames += episode_frames

        scores[task_name] = {
            "avg_cum_reward": float(np.mean(episode_cum_rewards)),
            "std_cum_reward": float(np.std(episode_cum_rewards)),
            "avg_step_reward": float(np.mean(episode_step_rewards)),
            "std_step_reward": float(np.std(episode_step_rewards)),
            "avg_solved": float(np.mean(episode_solved)),
            "std_solved": float(np.std(episode_solved)),
            "avg_solved_steps": float(
                np.mean(episode_solved_steps)
            ),  # Average of raw counts
            "avg_solved_step_frac": float(
                np.mean(episode_solved_fracs)
            ),  # Average of fractions
            "std_solved_steps": float(np.std(episode_solved_steps)),
            "avg_steps": float(np.mean(episode_steps)),
            "std_steps": float(np.std(episode_steps)),
            "max_episode_steps": int(max_episode_steps),
            "episode_cum_rewards": [float(reward) for reward in episode_cum_rewards],
            "episode_solve_step_fracs": [float(solved_frac) for solved_frac in episode_solved_fracs]
        }
        if args.save_video:
            if args.save_failed_video :
                for i, episode_frames in enumerate(frames) :
                    save_video(episode_frames, args.out_dir, task_name+"_"+str(i))
            else :
                save_video(frames, args.out_dir, task_name)
    print(json.dumps(scores, indent=2))

    # Save results if requested
    save_results(scores, args)


"""
mjpython src/benchmark.py --task elbow_pose --num_episodes 10 --render --expert

mjpython src/benchmark.py --task hand_thumb_reach --num_episodes 10 --render --expert --device cpu

mjpython src/benchmark.py \
    --task pen \
    --arnold \
    --policy transformer \
    --load data/student_policies/105_arnold_pen_bc_ppo_seed_0/rl_model_6400000_steps.zip \
    --device cpu \
    --num_episodes 200 \
    --normalize \
    --render
    
python src/benchmark.py\
    --task elbow_pose\
    --policy mlp\
    --load saves/server/new_single_task_mlp/elbow_pose/rl_model_500000_steps.zip\
    --device cpu\
    --num_episodes 200\
    --normalize
    
mjpython src/benchmark.py --task hand_index_reach --num_episodes 10 --render --expert --device cpu


mjpython src/benchmark.py \
    --task hand_middle_reach \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_single_task/hand_middle_reach/rl_model_5100000_steps.zip \
    --device cpu \
    --num_episodes 3 \
    --normalize \
    --render
    
mjpython src/benchmark.py \
    --task hand_ring_reach \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/119_arnold_htr_hir_hmr_hrr_hlr_bc_ppo_seed_0/rl_model_2199780_steps.zip \
    --device cpu \
    --num_episodes 100 \
    --normalize \
    --render
    
mjpython src/benchmark.py \
    --task baoding_p2 \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/120_arnold_bpc_bpc_bpo_bp_bc_ppo_seed_0/rl_model_20900000_steps.zip \
    --device cpu \
    --num_episodes 3 \
    --normalize \
    --render

mjpython src/benchmark.py \
    --task pen \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/122_arnold_hand_index_reach_pen_bc_ppo_seed_0/rl_model_15400000_steps.zip \
    --device cpu \
    --num_episodes 3 \
    --normalize \
    --render

mjpython src/benchmark.py \
    --task elbow_pose \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/127_arnold_hand_index_reach_elbow_pose_bc_ppo_seed_0/rl_model_3700000_steps.zip \
    --device cpu \
    --num_episodes 100 \
    --normalize \
    --render
    
mjpython src/benchmark.py \
    --task baoding_p1_cw \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0/rl_model_20593408_steps.zip \
    --device cpu \
    --num_episodes 10 \
    --normalize \
    --deterministic \
    --render

mjpython src/benchmark.py \
    --task pen \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0/rl_model_26691456_steps.zip \
    --device cpu \
    --num_episodes 20 \
    --normalize \
    --deterministic \
    --render

mjpython src/benchmark.py \
    --task reorient \
    --arnold \
    --policy transformer \
    --load data/student_policies/arnold_multi_task/139_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_bc_ppo_seed_0/rl_model_25898964_steps.zip \
    --device cpu \
    --num_episodes 500 \
    --normalize \
    --render

python src/benchmark.py \
    --load data/student_policies/arnold_multi_task/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0/rl_model_20693376_steps.zip \
    --task relocate \
    --arnold \
    --normalize \
    --num_episodes 200 \
    --deterministic \
    --out_dir data/benchmarks/student_policies/arnold_multi_task \
    --device cpu 

python src/benchmark.py \
    --load data/student_policies/arnold_multi_task/138_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_bc_ppo_seed_0/rl_model_26591488_steps.zip \
    --arnold \
    --normalize \
    --num_episodes 200\
    --deterministic \
    --save_results \
    --out_dir data/benchmarks/student_policies/arnold_multi_task \
    --device cpu

mjpython src/benchmark.py \
    --load data/student_policies/arnold_multi_task/148_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_bc_ppo_seed_2/rl_model_49898004_steps.zip \
    --task relocate \
    --arnold \
    --normalize \
    --num_episodes 10\
    --deterministic \
    --device cpu \
    --render
    
mjpython src/benchmark.py \
    --load data/student_policies/arnold_multi_task/162_arnold_kinesis_bc_ppo_seed_0/rl_model_8000000_steps.zip \
    --task kinesis \
    --arnold \
    --normalize \
    --num_episodes 10\
    --deterministic \
    --device cpu \
    --render

mjpython src/benchmark.py \
    --load data/student_policies/arnold_multi_task/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0/rl_model_45279162_steps.zip \
    --arnold \
    --normalize \
    --num_episodes 200 \
    --deterministic \
    --save_results \
    --out_dir data/benchmarks/student_policies/arnold_multi_task \
    --device cpu

mjpython src/benchmark.py \
    --load data/student_policies/arnold_multi_task/170_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_0/rl_model_45279162_steps.zip \
    --task elbow_pose \
    --arnold \
    --normalize \
    --num_episodes 200 \
    --deterministic \
    --device cpu \
    --render

mjpython src/benchmark.py \
    --load data/student_policies/arnold_multi_task/285_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_1/rl_model_64670238_steps.zip \
    --task kinesis \
    --arnold \
    --normalize \
    --num_episodes 200 \
    --deterministic \
    --device cpu \
    --render

mjpython src/benchmark.py \
    --load data/student_policies/arnold_single_task/baoding_p1_ccw/rl_model_100000_steps.zip \
    --task relocate \
    --arnold \
    --normalize \
    --num_episodes 200 \
    --deterministic \
    --device cpu \
    --render
    
mjpython src/benchmark.py \
    --task elbow_pose \
    --expert \
    --num_episodes 200\
    --deterministic \
    --device cpu \
    --render
    
mjpython src/benchmark.py --task relocate --num_episodes 10 --render --expert

"""
