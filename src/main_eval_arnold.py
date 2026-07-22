import argparse
import os
import json
import numpy as np
import pandas as pd
import glob
import skvideo
import platform
import time
from definitions import ROOT_DIR, MODEL_PATTERN, ENV_FILE_NAME
from envs.environment_factory import EnvironmentFactory
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.loaders import load_vocabulary
from train.algo_factory import AlgoFactory
from utilities import (
    get_number,
    load_model,
    get_best_checkpoint,
    get_experiment_data,
)
from envs.utilities import create_vec_env, load_vecnormalize


TB_DIR_NAME = "PPO_1"  # "RecurrentPPO_1", "SAC_1"
CKPT_CHOICE_CRITERION = "rollout/ep_rew_mean"  # "rollout/ep_rew_mean", "rollout/solved"
VIDEO_DIR = os.path.join(ROOT_DIR, "data", "videos")
HOST = "chiappa@sv-rcp-gateway.intranet.epfl.ch"
HOST_PROJECT_ROOT = "/storage-rcp-pure/upamathis_scratch/alberto/arnold"


def main(args):
    if args.experiment_path is None:
        eval_env = EnvironmentFactory.create(args.env_name)
        eval_env.seed(args.seed)
        model = AlgoFactory.get_algo_class(args.algo)(policy="MlpPolicy", env=eval_env)
        venv = DummyVecEnv([lambda: eval_env])
        vecnormalize = VecNormalize(venv, norm_obs=False)
    else:
        eval_env_config = json.load(open(args.eval_env_config, "r"))
        eval_env = EnvironmentFactory.create(**eval_env_config)
        eval_env.seed(args.seed)
        if args.train_env is not None:
            train_env = EnvironmentFactory.create(args.train_env)
        else:
            train_env = eval_env

        if args.checkpoint is None:
            # First get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(args.experiment_path, MODEL_PATTERN)),
                key=get_number,
            )
            checkpoints = [
                get_number(el)
                for el in model_list
                if get_number(el) < args.max_checkpoint
            ]
            if len(checkpoints) > 1:
                # Get the training data from the tensorboard log
                tb_dir_path = os.path.join(args.experiment_path, TB_DIR_NAME)
                experiment_data = get_experiment_data(
                    tb_dir_path, CKPT_CHOICE_CRITERION
                )
                steps = experiment_data[CKPT_CHOICE_CRITERION]["x"][0]
                rewards = experiment_data[CKPT_CHOICE_CRITERION]["y"][0]

                # Select the checkpoint corresponding to the best reward
                checkpoint = get_best_checkpoint(steps, rewards, checkpoints)
            elif len(checkpoints) == 1:
                checkpoint = checkpoints[0]
            else:
                checkpoint = None
        else:
            checkpoint = args.checkpoint

        print("Single env:", args.single_env)
        if not args.single_env:
            vocabulary = load_vocabulary(args.experiment_path)
        else:
            vocabulary = None

        custom_objects = {
            "observation_space": train_env.observation_space,
            "action_space": eval_env.action_space,
            "vocabulary": vocabulary,
        }
        model = load_model(
            algo=args.algo,
            experiment_path=args.experiment_path,
            checkpoint_number=checkpoint,
            custom_objects=custom_objects,
            custom_config=args.custom_model_config,
            custom_model_name=args.custom_model_name,
        )

        if args.no_clip_actions:
            action_low = model.action_space.low
            action_high = model.action_space.high
            model.action_space.low = -np.inf
            model.action_space.high = np.inf

        vecnormalize_path = os.path.join(args.experiment_path, ENV_FILE_NAME)
        vecnormalize = create_vec_env(
            env_config_list=[eval_env_config],
            load_env_path=vecnormalize_path,
            multi_env=not args.single_env,
            old_vocabulary=vocabulary,
        )
        # vecnormalize = load_vecnormalize(
        #     args.experiment_path,
        #     checkpoint,
        #     [eval_env_config],
        #     vocabulary,
        #     single_env=args.single_env,
        #     host=HOST,
        #     host_project_root=HOST_PROJECT_ROOT,
        # )

    # Collect rollouts and store them
    vecnormalize.training = False
    episode_data = []
    if args.render:
        eval_env.mujoco_render_frames = True
    if args.save_video:
        eval_env.mujoco_render_frames = False
        frames = []
    solved_avg = 0
    for i in range(args.num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        obs, _ = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        solved_count = 0
        while not done:
            if args.render:
                eval_env.sim.renderer.render_to_window()
                time.sleep(0.02)
            if args.save_video:
                curr_frame = eval_env.sim.renderer.render_offscreen(
                    # cameras=[None],
                    width=640,
                    height=480,
                    camera_id=1,
                    device_id=0,
                )
                frames.append(curr_frame)
            if args.train_env is not None:
                # We want to store the observation in arnold format, but if the agent was
                # trained in a different environment, we need to convert it
                obs_train_style = np.zeros(0)
                for key in train_env.obs_keys:
                    obs_train_style = np.concatenate(
                        [obs_train_style, eval_env.obs_dict[key].ravel()]
                    )  # ravel helps with images
            else:
                obs_train_style = obs

            normalized_obs = vecnormalize.normalize_obs(obs_train_style)
            action_store, lstm_states = model.predict(
                normalized_obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            action_store = np.squeeze(action_store)
            if args.no_clip_actions:
                action = np.clip(action_store, action_low, action_high)
            else:
                action = action_store
            next_obs, rewards, term, trunc, _ = eval_env.step(action)
            if args.num_steps_per_episode is not None:
                done = step >= args.num_steps_per_episode - 1
            else:
                done = term or trunc
            episode_data.append(
                [
                    i,
                    step,
                    obs,
                    action_store,
                    rewards,
                    next_obs,
                    eval_env.last_ctrl,
                    eval_env.rwd_dict,
                ]
            )
            obs = next_obs
            episode_starts = done
            cum_rew += rewards
            step += 1
            solved = 1.0 * eval_env.rwd_dict["solved"]
            solved_count += solved
        solved_avg = (solved_avg * i + solved_count / step) / (i + 1)
        print(
            "Episode",
            i,
            "Solved (ep):",
            solved_count / step,
            "Ep len:",
            step,
            "Solved (avg):",
            solved_avg,
        )
    eval_env.close()

    if args.save_video:
        if not os.path.exists(VIDEO_DIR):
            os.mkdir(VIDEO_DIR)
        file_name = os.path.join(VIDEO_DIR, "video.mp4")
        # check if the platform is OS -- make it compatible with quicktime
        if platform == "darwin":
            skvideo.io.vwrite(
                file_name, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"}
            )
        else:
            skvideo.io.vwrite(file_name, np.asarray(frames))
        print("saved", file_name)
    if not args.no_save_df:
        df = pd.DataFrame(
            episode_data,
            columns=[
                "episode",
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "muscle_act",
                "rew_dict",
            ],
        )
        out_name = (
            args.out_name if args.out_name is not None else eval_env_config["env_name"]
        )
        out_dir = os.path.join(ROOT_DIR, "data", "datasets", out_name)
        os.makedirs(out_dir, exist_ok=True)

        # Save the args as json
        args_dict = vars(args)
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            json.dump(args_dict, f)

        if args.deterministic:
            suffix = "_deterministic.pkl"
        else:
            suffix = "_stochastic.pkl"
        out_path = os.path.join(out_dir, "df" + suffix)
        df.to_pickle(out_path)
        print("Saved to ", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script to create a dataset of episodes with a trained agent"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Algorithm used to train the agent",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=None,
        help="Path to the folder where the experiment results are stored",
    )
    parser.add_argument(
        "--custom_model_name",
        type=str,
        default=None,
        help="The model name inside the experiment path",
    )
    parser.add_argument(
        "--custom_model_config",
        type=str,
        default=None,
        help="Path to the custom model configuration file (within the experiment folder)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Number of the checkpoint to select. Otherwise the checkpoint corresponding to the highest reward is selected.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Name of the environment where to test the agent",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Name of the environment where to test the agent",
    )
    parser.add_argument(
        "--train_env",
        type=str,
        default=None,
        help="Name of the environment where the agent was trained",
    )
    parser.add_argument(
        "--eval_env_config",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "env_configs", "eval_config.json"),
        help="Path to the environment configuration file",
    )
    parser.add_argument(
        "--single_env",
        action="store_true",
        default=False,
        help="Flag to add if the model was trained without arnold multi-env",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--num_steps_per_episode",
        type=int,
        default=None,
        help="Number of steps per episode",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Flag to use the deterministic policy",
    )
    parser.add_argument(
        "--no_clip_actions",
        action="store_true",
        default=False,
        help="Flag to not clip the actions within the action space range",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed of the environment.",
    )
    parser.add_argument(
        "--max_checkpoint",
        type=float,
        default=float("inf"),
        help="Do not consider checkpoints past this number (to be fair across trainings)",
    )
    parser.add_argument(
        "--no_save_df",
        action="store_true",
        default=False,
        help="Flag to not save the dataframe",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Flag to render at the screen",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Flag to save a video",
    )
    args = parser.parse_args()
    main(args)

    """
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
    --num_episodes=1000 --no_save_df --checkpoint=25200000 --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MuscleHandReachRandom_ppo_seed_1_nl_2_nh_1_es_128_df_256_033 \
    --num_episodes=1000 --no_save_df --checkpoint=39360000 --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
    --num_episodes=1000 --no_save_df --checkpoint=26280000 --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MuscleBaodingP0_ppo_seed_0_nl_2_nh_1_es_128_df_256_act_reg_037 \
    --num_episodes=1000 --no_save_df --checkpoint=40320000 --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MBP_MBC_MBC_MBC_MBC_MBC_MBC_MBC_MBC_MBC_MBP_ppo_seed_0_nl_2_nh_1_es_128_df_256_act_reg_044 \
    --num_episodes=1000 --no_save_df --checkpoint=49915008 --deterministic --render

    # Render
    mjpython src/main_eval_arnold.py --train_env=CleanBaodingBalls --algo=recurrent_ppo --experiment_path=data/expert_policies/baoding_phase_1 \
        --eval_env_config=data/env_configs/baoding_p1_config.json --single_env --num_episodes=10 --no_save_df --deterministic --render
    
    # Dataset baoding p1 ccw
    python src/main_eval_arnold.py --out_name=baoding_p1_ccw_no_clip_100k --train_env=CleanBaodingBalls --algo=recurrent_ppo --experiment_path=data/expert_policies/baoding_phase_1 \
        --eval_env_config=data/env_configs/baoding_p1_ccw_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions
        
    # Dataset baoding p1 cw
    python src/main_eval_arnold.py --out_name=baoding_p1_cw_no_clip_100k --train_env=CleanBaodingBalls --algo=recurrent_ppo --experiment_path=data/expert_policies/baoding_phase_1 \
        --eval_env_config=data/env_configs/baoding_p1_cw_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions
        
    # Dataset baoding p2 overlap
    python src/main_eval_arnold.py --out_name=baoding_p2_overlap_no_clip_100k --train_env=CleanBaodingBalls --algo=recurrent_ppo --experiment_path=data/expert_policies/baoding_phase_2 \
        --eval_env_config=data/env_configs/baoding_p2_overlap_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions
        
    # Dataset baoding p2
    python src/main_eval_arnold.py --out_name=baoding_p2_no_clip_100k --train_env=CleanBaodingBalls --algo=recurrent_ppo --experiment_path=data/expert_policies/baoding_phase_2 \
        --eval_env_config=data/env_configs/baoding_p2_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions
        
    # Dataset hand reach
    mjpython src/main_eval_arnold.py --out_name=hand_reach_no_clip_100k --train_env=HandReachRandom --algo=recurrent_ppo --experiment_path=data/expert_policies/hand_reach_lattice \
        --custom_model_config="model_config.json" --eval_env_config=data/env_configs/hand_reach_config.json --single_env --num_episodes=1000 --num_steps_per_episode=100 --no_clip_actions --render

    # Dataset hand pose (WIP: cannot get it to work)
    python src/main_eval_arnold.py --out_name=hand_pose --train_env=CustomMyoHandPoseRandom --algo=recurrent_ppo --experiment_path=data/expert_policies/hand_pose_lattice \
        --custom_model_config="model_config.json" --eval_env_config=data/env_configs/hand_pose_config.json --single_env --num_episodes=10000 --num_steps_per_episode=100

    # Dataset pen
    mjpython src/main_eval_arnold.py --out_name=pen_no_clip_100k --train_env=CustomMyoPenTwirlRandom --algo=recurrent_ppo --experiment_path=data/expert_policies/pen_lattice \
        --custom_model_config="model_config.json" --eval_env_config=data/env_configs/pen_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions --render

    # Dataset reorient
    python src/main_eval_arnold.py --out_name=reorient_no_clip_100k --train_env=CustomMyoReorientP0 --algo=recurrent_ppo --experiment_path=data/expert_policies/reorient_lattice \
        --custom_model_config="model_config.json" --eval_env_config=data/env_configs/reorient_config.json --single_env --num_episodes=500 --num_steps_per_episode=200 --no_clip_actions

    # Dataset relocate
    mjpython src/main_eval_arnold.py --out_name=relocate_no_clip_100k --train_env=CustomRelocate --algo=recurrent_ppo --experiment_path=data/expert_policies/relocate \
        --custom_model_config="model_config.json" --eval_env_config=data/env_configs/relocate_config.json --single_env --num_episodes=667 --num_steps_per_episode=150 --no_clip_actions

    # Dataset hand little reach
    mjpython src/main_eval_arnold.py --out_name=hand_little_reach_no_clip_100k --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_little_reach_config.json --num_episodes=10_00 --num_steps_per_episode=100 --no_clip_actions

    # Dataset hand index reach
    mjpython src/main_eval_arnold.py --out_name=hand_index_reach_no_clip_100k --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_index_reach_config.json --num_episodes=10_00 --num_steps_per_episode=100 --no_clip_actions --render
    
    # Dataset hand middle reach
    mjpython src/main_eval_arnold.py --out_name=hand_middle_reach_no_clip_100k --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_middle_reach_config.json --num_episodes=10_00 --num_steps_per_episode=100 --no_clip_actions

    # Dataset hand ring reach
    mjpython src/main_eval_arnold.py --out_name=hand_ring_reach_no_clip_100k --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_ring_reach_config.json --num_episodes=10_00 --num_steps_per_episode=100 --no_clip_actions

    # Dataset hand thumb reach
    mjpython src/main_eval_arnold.py --out_name=hand_thumb_reach_no_clip_100k --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_thumb_reach_config.json --num_episodes=10_00 --num_steps_per_episode=100 --no_clip_actions

    # Dataset hand index reach - tiny version
    mjpython src/main_eval_arnold.py --out_name=hand_index_reach_tiny_no_clip --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_index_reach_config.json --num_episodes=1 --num_steps_per_episode=100 --no_clip_actions
        

    # Video all hand reach - no act reg
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
        --checkpoint=25200000 --eval_env_config=data/env_configs/hand_little_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
        --checkpoint=25200000 --eval_env_config=data/env_configs/hand_index_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
        --checkpoint=25200000 --eval_env_config=data/env_configs/hand_middle_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render
    
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
        --checkpoint=25200000 --eval_env_config=data/env_configs/hand_ring_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render
        
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256_018 \
        --checkpoint=25200000 --eval_env_config=data/env_configs/hand_thumb_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render
        
    # Video all hand reach - act reg
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_little_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_index_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render

    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_middle_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render
    
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_ring_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render
        
    mjpython src/main_eval_arnold.py --experiment_path=output/training/ongoing/MHTRR_MHIRR_MHMRR_MHRRR_MHLRR_ppo_seed_1_nl_2_nh_1_es_128_df_256act_reg_034 \
        --checkpoint=31920000 --eval_env_config=data/env_configs/hand_thumb_reach_config.json --num_episodes=1000 --no_save_df --deterministic --render

    """
