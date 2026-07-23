import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import numpy as np
import tqdm
from models.ppo.policies import MuscleTransformerPolicy
from envs.utilities import create_vec_env
import torch
import argparse
import os
import json
from definitions import ENV_CONFIG_PATH
from envs.environment_factory import EnvironmentFactory
import h5py
from algos.bc_ppo import BCPPO
from models.ppo.policies import to_tensor_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect activations from MuscleTransformerPolicy rollouts"
    )
    parser.add_argument(
        "--load", type=str, required=True, help="Path to policy checkpoint"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task to collect activations for"
    )
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--arnold", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/activations",
        help="Directory to save activation data",
    )
    return parser.parse_args()


def collect_episode_data(env, policy, vecnormalize=None, device="cuda"):
    """Collect a single episode of data including intermediate layer activations"""
    obs, _ = env.reset()
    done = False
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "solved": [],
        "encoder_output": [],
        "decoder_output": [],
        "action_means": [],
    }

    while not done:
        # Normalize observation if needed
        if vecnormalize:
            if isinstance(obs, dict):
                obs_normalized = {key: obs[key][None, ...] for key in obs}
                obs_normalized = vecnormalize.normalize_single_obs_dict(
                    obs_normalized, env_idx=0
                )
            else:
                obs_normalized = vecnormalize.normalize_obs(obs)[None, ...]
        else:
            obs_normalized = obs

        # Convert observation to tensor
        obs_tensor = to_tensor_dict(obs_normalized, device=device)

        # Forward pass through policy collecting intermediate activations
        with torch.no_grad():
            # Get features
            features = policy.extract_features(obs_tensor)

            # Run encoder and collect output
            encodings, encodings_mask, action_target, action_mask, value_target = (
                features
            )
            encoded = policy.mlp_extractor.encoder(
                encodings, src_key_padding_mask=encodings_mask
            )

            # Run action decoder and collect output
            action_decoded = policy.mlp_extractor.action_decoder(
                action_target,
                encoded,
                tgt_key_padding_mask=action_mask,
                memory_key_padding_mask=encodings_mask,
            )

            # # Run value decoder and collect output
            # value_decoded = policy.mlp_extractor.value_decoder(
            #     value_target,
            #     encoded,
            #     memory_key_padding_mask=encodings_mask
            # )

            # Get action distribution mean
            # latent_pi, latent_vf = policy.mlp_extractor((encoded, action_decoded, value_decoded))
            action_dist = policy._get_action_dist_from_latent(action_decoded)
            action_mean = action_dist.distribution.mean.cpu().numpy()

        # Get action using policy.predict() for environment stepping
        action = action_dist.sample().cpu().numpy().squeeze()

        # Step environment
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        # Store data
        episode_data["observations"].append(obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["solved"].append(float(info["rwd_dict"]["solved"]))
        episode_data["encoder_output"].append(encoded.cpu().numpy())
        episode_data["decoder_output"].append(action_decoded.cpu().numpy())
        episode_data["action_means"].append(action_mean)

        obs = next_obs

    # Convert lists to numpy arrays
    for key in episode_data:
        episode_data[key] = np.array(episode_data[key])

    return episode_data


def save_episode_data(episode_data, filepath):
    """Save episode data to HDF5 file"""
    with h5py.File(filepath, "w") as f:
        # Store episode statistics
        f.attrs["episode_length"] = len(episode_data["rewards"])
        f.attrs["total_reward"] = np.sum(episode_data["rewards"])
        f.attrs["solved_steps"] = np.sum(episode_data["solved"])

        # Store all data
        for key, value in episode_data.items():
            if key == "observations" and isinstance(value[0], dict):
                obs_group = f.create_group("observations")
                for obs_key in value[0].keys():
                    obs_group.create_dataset(
                        obs_key, data=np.array([obs[obs_key] for obs in value])
                    )
            else:
                f.create_dataset(key, data=value)


def main():
    args = parse_args()

    # Simplify policy ID and checkpoint extraction
    policy_id = os.path.basename(os.path.dirname(args.load)).split("_")[
        0
    ]  # Get just the number
    checkpoint_num = os.path.basename(args.load).split("_")[2]

    # Simplified output directory
    out_dir = os.path.join(args.out_dir, f"{policy_id}_{checkpoint_num}")
    os.makedirs(out_dir, exist_ok=True)

    # Load student policy
    try:
        policy = MuscleTransformerPolicy.load(args.load, device=args.device)
    except:
        policy = BCPPO.load(args.load, device=args.device).policy
    policy.to(args.device)

    # Load environment
    prefix = "arnold_" if args.arnold else ""
    env_config_path = os.path.join(ENV_CONFIG_PATH, f"{prefix}{args.task}_config.json")
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    if args.normalize:
        vecnormalize_path = args.load.replace("model", "model_vecnormalize").replace(
            ".zip", ".pkl"
        )
        vecnormalize = create_vec_env(
            env_config_list=[env_config],
            load_env_path=vecnormalize_path,
            multi_env=args.arnold,
            old_vocabulary=None,
        )
    else:
        vecnormalize = None

    env = EnvironmentFactory.create(**env_config)
    policy.observation_space = env.observation_space

    # Initialize solved count
    total_solved_fractions = []

    # Collect episodes
    for episode in tqdm.tqdm(
        range(args.num_episodes), desc=f"Collecting episodes for {args.task}"
    ):
        episode_data = collect_episode_data(env, policy, vecnormalize, args.device)

        # Calculate solved fraction for this episode
        solved_fraction = np.mean(episode_data["solved"])
        total_solved_fractions.append(solved_fraction)

        # Save episode data with new naming convention
        filepath = os.path.join(out_dir, f"{args.task}_episode_{episode}.h5")
        print(f"\nSaving episode data to: {filepath}")
        save_episode_data(episode_data, filepath)

        # Print shapes for the first episode only
        if episode == 0:
            print("\nActivation shapes for first episode:")
            print("-" * 50)
            print(f"Episode length: {len(episode_data['rewards'])} timesteps\n")

            print("Encoded:")
            print(f"  Encoder output shape: {episode_data['encoder_output'][0].shape}")

            print("Decoded:")
            print(f"  Decoder output shape: {episode_data['decoder_output'][0].shape}")

            print("-" * 50)

    # Print average solved fraction
    avg_solved = np.mean(total_solved_fractions)
    print(
        f"\nAverage solved fraction across {args.num_episodes} episodes: {avg_solved:.3f}"
    )


if __name__ == "__main__":
    main()

"""
for task in hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap; do
    python src/collect_activations.py \
        --load data/student_policies/arnold_multi_task/285_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_1/rl_model_64670238_steps.zip \
        --task $task \
        --num_episodes 100 \
        --arnold \
        --normalize \
        --device cpu \
        --out_dir data/activations
done
"""
