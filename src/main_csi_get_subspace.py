import argparse
from envs.utilities import create_vec_env
from stable_baselines3 import PPO
from models.csi_model import CSIActionNet
from definitions import ENV_CONFIG_PATH
import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import tqdm
import json
from definitions import ROOT_DIR, ENV_CONFIG_PATH

parser = argparse.ArgumentParser(description="Main script to get the CSI subspace")

parser.add_argument("--policy_path", type=str, help="Path to the policy to load")
# parser.add_argument("--vecnorm_path", type=str, help="Path to the vecnorm to load")
parser.add_argument("--task", type=str, help="Name of the task", choices=[
    "elbow_pose",
    "hand_index_reach",
    "hand_little_reach",
    "hand_middle_reach",
    "hand_ring_reach",
    "hand_thumb_reach",
    "kinesis",
    "pen",
    "relocate",
    "reorient",
    "baoding_p1_ccw",
    "baoding_p1_cw",
    "baoding_p2",
    "baoding_p2_overlap"
])
parser.add_argument("--save_path", type=str, help="Path to save the subspace")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to collect data")
args = parser.parse_args()

if __name__ == "__main__":

    env_config_path = os.path.join(ENV_CONFIG_PATH, f"{args.task}_config.json")
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
        
    os.makedirs(os.path.join(ROOT_DIR, args.save_path), exist_ok=True)
    print(f"Saving subspace to {os.path.join(ROOT_DIR, args.save_path)}")
    
    envs = create_vec_env(
        env_config_list=[env_config],
        num_envs_per_config=args.num_envs,
        load_env_path=args.policy_path.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"),
        multi_env=False,
        old_vocabulary=None,
    )
    policy = PPO.load(
        args.policy_path,
        envs,
        custom_objects={"policy_class": CSIActionNet},
        device="cpu"
    )
    
    actions = []
    
    obs = envs.reset()
    with tqdm.tqdm(total=args.num_steps) as pbar:
        for i in range(args.num_steps):
            action, _ = policy.predict(obs)
            obs, rew, done, info = envs.step(action)
            actions.append(action)
            pbar.update(1)
    
    actions = np.concatenate(actions, axis=0)
    # get PCA of actions
    pca = PCA()
    pca.fit(actions)
    subspace = pca.components_
    explained_variance = pca.explained_variance_
    mean = pca.mean_
    np.save(os.path.join(ROOT_DIR, args.save_path, "subspace.npy"), subspace)
    np.save(os.path.join(ROOT_DIR, args.save_path, "explained_variance.npy"), explained_variance)
    np.save(os.path.join(ROOT_DIR, args.save_path, "mean.npy"), mean)
    print(f"Explained variance: {explained_variance}")
    print(f"Subspace saved to {args.save_path}")
