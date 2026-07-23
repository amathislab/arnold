import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import numpy as np
import h5py
import os
import glob
from sklearn.decomposition import PCA
from envs.utilities import create_vec_env
from envs.environment_factory import EnvironmentFactory
from models.ppo.policies import MuscleTransformerPolicy
import torch
import json
import tqdm
from algos.bc_ppo import BCPPO
from definitions import ENV_CONFIG_PATH, ROOT_DIR
import time

def load_actions_from_h5(h5_path):
    """Load actions from an H5 file"""
    with h5py.File(h5_path, 'r') as f:
        actions = np.array(f['action_means'])
        solved = np.array(f['solved'])
        
    return actions.squeeze(axis=1), np.mean(solved)

def load_all_episode_actions(base_dir, task):
    """Load actions from all episodes for a given task"""
    pattern = os.path.join(base_dir, f"{task}_episode_*.h5")
    files = glob.glob(pattern)
    all_actions = []
    solved_ratios = []
    for f in files:
        actions, solved_ratio = load_actions_from_h5(f)
        all_actions.append(actions)
        solved_ratios.append(solved_ratio)
    
    avg_solved = np.mean(solved_ratios)
    print(f"Task {task} average solved ratio: {avg_solved:.3f}")
    return np.vstack(all_actions)

def evaluate_with_pca_inactivation(env, vecnormalize, policy, pca, num_episodes=10, device="cuda"):
    """Evaluate policy performance while inactivating PCs one by one"""
    n_comp = pca.n_components_
    performance = []
    
    for k in range(n_comp):
        print(f"Testing with {n_comp-k} components")
        components = pca.components_[:n_comp-k]
        
        performance_ep = []
        solved_steps_ep = []
        step_count_ep = []
        is_solved_ep = []
        for n in range(num_episodes):
            cum_reward = 0
            solved_steps = 0
            total_steps = 0
            obs, _ = env.reset()
            done = False
            
            while not done:
                # Normalize observation if needed
                if vecnormalize:
                    if isinstance(obs, dict):
                        obs_normalized = {key: obs[key][None, ...] for key in obs}
                        obs_normalized = vecnormalize.normalize_single_obs_dict(obs_normalized, env_idx=0)
                    else:
                        obs_normalized = vecnormalize.normalize_obs(obs)[None, ...]
                else:
                    obs_normalized = obs

                # Get action from policy
                with torch.no_grad():
                    action, _ = policy.predict(obs_normalized, deterministic=False)

                # Project action through reduced PCA space
                action_proj = np.dot(action.reshape(1,-1)-pca.mean_, components.T)
                action_backproj = np.dot(action_proj, components)+pca.mean_
                
                # Step environment
                next_obs, reward, term, trunc, info = env.step(action_backproj.squeeze())
                done = term or trunc
                obs = next_obs
                cum_reward += reward
                solved_steps += float(info["rwd_dict"]["solved"])
                total_steps += 1
                
            performance_ep.append(cum_reward)
            solved_steps_ep.append(solved_steps)
            step_count_ep.append(total_steps)
            is_solved_ep.append(solved_steps > 0)
            print(f"Episode {n}, reward: {cum_reward:.3f}, solved count: {solved_steps}, total steps: {total_steps}, solved: {solved_steps > 0}")
            
        perf_array = np.array(performance_ep)
        solved_array = np.array(solved_steps_ep)
        
        data_point = {
            'components': components,
            'reward_mean': float(np.mean(perf_array)),
            'reward_sem': float(np.std(perf_array) / np.sqrt(len(perf_array))),
            'solved_count_mean': float(np.mean(solved_array)),
            'solved_count_sem': float(np.std(solved_array) / np.sqrt(len(solved_array))),
            'solved_mean': float(np.mean(is_solved_ep)),
            'solved_sem': float(np.std(is_solved_ep) / np.sqrt(len(is_solved_ep))),
            'ep_len_mean': float(np.mean(step_count_ep)),
            'ep_len_sem': float(np.std(step_count_ep) / np.sqrt(len(step_count_ep))),
            'n_episodes': num_episodes,
        }
        performance.append(data_point)
        
        print(f"\nComponents: {n_comp-k}")
        print(f"Mean reward: {data_point['reward_mean']:.3f} ± {data_point['reward_sem']:.3f}")
        print(f"Mean solved: {data_point['solved_mean']:.3f} ± {data_point['solved_sem']:.3f}")
        print(f"Mean solved count: {data_point['solved_count_mean']:.3f} ± {data_point['solved_count_sem']:.3f}")
        print(f"Mean episode length: {data_point['ep_len_mean']:.3f} ± {data_point['ep_len_sem']:.3f}")
        
    return performance

def main():
    # Configuration
    policy_dir = os.path.join(ROOT_DIR, "data/student_policies/arnold_multi_task/285_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_1")
    model_path = os.path.join(policy_dir, "rl_model_64670238_steps.zip")
    vecnorm_path = os.path.join(policy_dir, "rl_model_vecnormalize_64670238_steps.pkl")
    activations_dir = os.path.join(ROOT_DIR, "data/activations/285_64670238")
    out_dir = os.path.join(ROOT_DIR, "data/pca_analysis/285_64670238")
    
    # Define policy name from path
    policy_name = os.path.basename(os.path.dirname(model_path))
    print(f"\nTesting policy: {policy_name}")
    
    tasks = ["hand_thumb_reach", "hand_index_reach", "hand_middle_reach", 
             "hand_ring_reach", "hand_little_reach", "reorient", "pen",
             "baoding_p1_ccw", "baoding_p1_cw", "baoding_p2", "baoding_p2_overlap"]
    num_episodes = 100
    
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load policy
    try:
        policy = MuscleTransformerPolicy.load(model_path, device=device)
    except:
        policy = BCPPO.load(model_path, device=device).policy
    policy.to(device)

    # First analyze each task separately
    for task in tasks:
        print(f"\nAnalyzing task: {task}")
        
        # Load actions and compute PCA
        actions = load_all_episode_actions(activations_dir, task)
        pca = PCA(n_components=39)
        pca.fit(actions)
        
        print(f"Task PCA computed. Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
        
        # Create environment
        prefix = "arnold_"
        env_config_path = os.path.join(ENV_CONFIG_PATH, f"{prefix}{task}_config.json")
        with open(env_config_path, "r") as f:
            env_config = json.load(f)
            
        env = EnvironmentFactory.create(**env_config)
        vecnormalize = create_vec_env(env_config_list=[env_config], load_env_path=vecnorm_path, multi_env=True)
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        
        # Evaluate with PCA inactivation
        print(f"Evaluating task {task} with task-specific PCA")
        performance = evaluate_with_pca_inactivation(
            env, vecnormalize, policy, pca, num_episodes=num_episodes, device=device
        )
        
        # Save results
        results = {
            "pca": pca,
            "performance": performance,
            "policy": policy_name,
            "task": task,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        }
        
        out_path = os.path.join(out_dir, f"pca_inactivation_{task}_{policy_name}.pkl")
        with open(out_path, "wb") as f:
            np.save(f, results)
            
    # Now analyze all tasks together
    print("\nAnalyzing all tasks together")
    
    # Combine actions from all tasks
    all_actions = []
    for task in tasks:
        actions = load_all_episode_actions(activations_dir, task)
        all_actions.append(actions)
    all_actions = np.vstack(all_actions)
    
    # Compute PCA on combined actions
    pca_all = PCA(n_components=39)
    pca_all.fit(all_actions)
    print(f"Combined PCA computed. Explained variance ratio: {pca_all.explained_variance_ratio_[:5]}")
    
    # Evaluate on each task using the combined PCA
    for task in tasks:
        print(f"\nEvaluating task {task} with combined PCA")
        
        # Create environment
        prefix = "arnold_"
        env_config_path = os.path.join(ENV_CONFIG_PATH, f"{prefix}{task}_config.json")
        with open(env_config_path, "r") as f:
            env_config = json.load(f)
            
        env = EnvironmentFactory.create(**env_config)
        vecnormalize = create_vec_env([env_config], load_env_path=vecnorm_path, multi_env=True)
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        
        # Evaluate with PCA inactivation
        performance = evaluate_with_pca_inactivation(
            env, vecnormalize, policy, pca_all, num_episodes=num_episodes, device=device
        )
        
        # Save results
        results = {
            "pca": pca_all,
            "performance": performance,
            "policy": policy_name,
            "task": task,
            "pca_type": "combined",
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        }
        
        out_path = os.path.join(out_dir, f"combined_pca_inactivation_{task}_{policy_name}.pkl")
        with open(out_path, "wb") as f:
            np.save(f, results)

if __name__ == "__main__":
    main()

"""
python src/analyze_pca_inactivation.py
"""