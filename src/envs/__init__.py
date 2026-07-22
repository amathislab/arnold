import os
import gymnasium as gym
import numpy as np
import myosuite
from definitions import ROOT_DIR  # pylint: disable=import-error
from myosuite.envs.myo.myobase import register_env_with_variants
from itertools import combinations
from gym.envs.registration import register

myosuite_path = os.path.dirname(myosuite.__file__)
local_myosuite_asset_path = os.path.join(myosuite_path, "envs", "myo")

# Kinesis
register(
    id="Kinesis-v0",
    entry_point="envs.kinesis:KinesisEnv",
    max_episode_steps=150,
)

register(
    id="MuscleKinesis-v0",
    entry_point="envs.kinesis:MuscleKinesisEnv",
    max_episode_steps=150,
)

# Non-muscle environments
register_env_with_variants(
    id="CleanBaodingBalls-v1",
    entry_point="envs.baoding:CleanBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
    },
)

register_env_with_variants(
    id="CleanBaodingBallsP1CCW-v1",
    entry_point="envs.baoding:CleanBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "goal_time_period": [5, 5],
        "initial_phase": 1.5707963267948966,
        "limit_sds_angle": 0,
        "limit_init_angle": 0,
        "task_choice": "ccw",
    },
)

register_env_with_variants(
    id="CleanBaodingBallsP1CW-v1",
    entry_point="envs.baoding:CleanBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "goal_time_period": [5, 5],
        "initial_phase": 1.5707963267948966,
        "limit_sds_angle": 0,
        "limit_init_angle": 0,
        "task_choice": "cw",
    },
)

register_env_with_variants(
    id="CleanBaodingBallsP2Overlap-v1",
    entry_point="envs.baoding:CleanBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
        "limit_init_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="CleanBaodingBallsP2-v1",
    entry_point="envs.baoding:CleanBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
    },
)


# Elbow posing ==============================
register_env_with_variants(
    id="MuscleElbowPoseFixed-v0",
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/assets/elbow/myoelbow_1dof6muscles.xml",
        "target_jnt_range": {
            "r_elbow_flex": (2, 2),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)

register_env_with_variants(
    id="MuscleElbowPoseRandom-v0",
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/assets/elbow/myoelbow_1dof6muscles.xml",
        "target_jnt_range": {
            "r_elbow_flex": (0, 2.27),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
    },
)


register_env_with_variants(
    id="CleanElbowPose-v0",
    entry_point="envs.pose:CleanPoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/assets/elbow/myoelbow_1dof6muscles.xml",
        "target_jnt_range": {
            "r_elbow_flex": (0, 2.27),
        },
        "viz_site_targets": ("wrist",),
        "normalize_act": True,
        "pose_thd": 0.175,
        "reset_type": "random",
        "weighted_reward_keys": {
            "pose": 0,
            "bonus": 0,
            "penalty": 0,
            "act_reg": 0,
            "solved": 1,
            "done": 0,
            "sparse": 0,
        },
    },
)
# Finger-Joint posing ==============================
register_env_with_variants(
    id="CleanFingerPose-v0",
    entry_point="envs.pose:CleanPoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/finger/motorfinger_v0.xml",
        "target_jnt_range": {
            "IFadb": (-0.2, 0.2),
            "IFmcp": (-0.4, 1),
            "IFpip": (0.1, 1),
            "IFdip": (0.1, 1),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
        "weighted_reward_keys": {
            "pose": 0,
            "bonus": 0,
            "penalty": 0,
            "act_reg": 0,
            "solved": 1,
            "done": 0,
            "sparse": 0,
        },
    },
)

register_env_with_variants(
    id="MuscleFingerPoseFixed-v0",
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/finger/motorfinger_v0.xml",
        "target_jnt_range": {
            "IFadb": (0, 0),
            "IFmcp": (0, 0),
            "IFpip": (0.75, 0.75),
            "IFdip": (0.75, 0.75),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
    },
)

register_env_with_variants(
    id="MuscleFingerPoseRandom-v0",
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/finger/motorfinger_v0.xml",
        "target_jnt_range": {
            "IFadb": (-0.2, 0.2),
            "IFmcp": (-0.4, 1),
            "IFpip": (0.1, 1),
            "IFdip": (0.1, 1),
        },
        "viz_site_targets": ("IFtip",),
        "normalize_act": True,
    },
)
# Hand-Joint posing ==============================
# Create ASL envs ==============================
jnt_namesHand = [
    "pro_sup",
    "deviation",
    "flexion",
    "cmc_abduction",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
]

ASL_qpos = {}
ASL_qpos[0] = (
    "0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 1.26466 0 1.40604 0.227795 1.07614 -0.020944 1.46103 0.06284 0.83263 -0.14399 1.571 1.38248".split(
        " "
    )
)
ASL_qpos[1] = (
    "0 0 0 0.0248 0.04536 -0.7854 -1.309 0.366605 0.010473 0.269258 0.111722 1.48459 0 1.45318 1.44532 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459".split(
        " "
    )
)
ASL_qpos[2] = (
    "0 0 0 0.0248 0.04536 -0.7854 -1.13447 0.514973 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459".split(
        " "
    )
)
ASL_qpos[3] = (
    "0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.571 -0.036652 1.52387 1.45318 1.40604 -0.068068 1.39033 1.571".split(
        " "
    )
)
ASL_qpos[4] = (
    "0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571".split(
        " "
    )
)
ASL_qpos[5] = (
    "0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571".split(
        " "
    )
)
ASL_qpos[6] = (
    "0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 1.1861 -0.2618 1.35891 1.48459".split(
        " "
    )
)
ASL_qpos[7] = (
    "0 0 0 0.524 0.01569 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.28036 -0.115192 1.52387 1.45318 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)
ASL_qpos[8] = (
    "0 0 0 0.428 0.22338 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.194636 1.39033 0 1.08399 0.573415 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)
ASL_qpos[9] = (
    "0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 0.39275 0 0.18852 0.227795 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245".split(
        " "
    )
)


# ASL Train Env
m = np.array([ASL_qpos[i] for i in range(10)]).astype(float)
Rpos = {}
for i_n, n in enumerate(jnt_namesHand):
    Rpos[n] = (np.min(m[:, i_n]), np.max(m[:, i_n]))

register_env_with_variants(
    id="myoHandPoseRandomHalfRange-v0",
    entry_point="envs.pose:CleanPoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "viz_site_targets": ("THtip", "IFtip", "MFtip", "RFtip", "LFtip"),
        "target_jnt_range": Rpos,
        "normalize_act": True,
        "pose_thd": 0.8,
        "reset_type": "random",  # none, init, random
        "target_type": "generate",  # generate/ fixed
        "target_distance": 0.5,
    },
)


register_env_with_variants(
    id="MuscleHandPoseFixed-v0",  # reconsider
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "viz_site_targets": ("THtip", "IFtip", "MFtip", "RFtip", "LFtip"),
        "target_jnt_value": np.array(
            [
                0,
                0,
                0,
                -0.0904,
                0.0824475,
                -0.681555,
                -0.514888,
                0,
                -0.013964,
                -0.0458132,
                0,
                0.67553,
                -0.020944,
                0.76979,
                0.65982,
                0,
                0,
                0,
                0,
                0.479155,
                -0.099484,
                0.95831,
                0,
            ]
        ),
        "normalize_act": True,
        "pose_thd": 0.7,
        "reset_type": "init",  # none, init, random
        "target_type": "fixed",  # generate/ fixed
    },
)

register_env_with_variants(
    id="MuscleHandPoseRandom-v0",  # reconsider
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "viz_site_targets": ("THtip", "IFtip", "MFtip", "RFtip", "LFtip"),
        "target_jnt_range": Rpos,
        "normalize_act": True,
        "pose_thd": 0.7,
        "reset_type": "random",  # none, init, random
        "target_type": "generate",  # generate/ fixed
    },
)

half_rpos = {k: ((3 * v[0] + v[1]) / 4, (v[0] + 3 * v[1]) / 4) for k, v in Rpos.items()}

register_env_with_variants(
    id="MuscleHandPoseRandomHalfRange-v0",  # reconsider
    entry_point="envs.pose:MusclePoseEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "viz_site_targets": ("THtip", "IFtip", "MFtip", "RFtip", "LFtip"),
        "target_jnt_range": half_rpos,
        "normalize_act": True,
        "pose_thd": 0.7,
        "reset_type": "random",  # none, init, random
        "target_type": "generate",  # generate/ fixed
    },
)

# =========== Reaching environemtns ===============
register_env_with_variants(
    id="MuscleFingerReachFixed-v0",
    entry_point="envs.reach:MuscleReachEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/finger/motorfinger_v0.xml",
        "target_reach_range": {
            "IFtip": ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),
        },
        "normalize_act": True,
    },
)

register_env_with_variants(
    id="MuscleFingerReachRandom-v0",
    entry_point="envs.reach:MuscleReachEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/finger/motorfinger_v0.xml",
        "target_reach_range": {
            "IFtip": ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3)),
        },
        "normalize_act": True,
    },
)


# Hand reach
tip_range_fixed_dict = {
    "THtip": ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
    "IFtip": ((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455)),
    "MFtip": ((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447)),
    "RFtip": ((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445)),
    "LFtip": ((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434)),
}

tip_range_random_dict = {
    "THtip": (
        (-0.165 - 0.020, -0.537 - 0.040, 1.495 - 0.040),
        (-0.165 + 0.040, -0.537 + 0.020, 1.495 + 0.040),
    ),
    "IFtip": (
        (-0.151 - 0.040, -0.547 - 0.020, 1.455 - 0.010),
        (-0.151 + 0.040, -0.547 + 0.020, 1.455 + 0.010),
    ),
    "MFtip": (
        (-0.146 - 0.040, -0.547 - 0.020, 1.447 - 0.010),
        (-0.146 + 0.040, -0.547 + 0.020, 1.447 + 0.010),
    ),
    "RFtip": (
        (-0.148 - 0.040, -0.543 - 0.020, 1.445 - 0.010),
        (-0.148 + 0.040, -0.543 + 0.020, 1.445 + 0.010),
    ),
    "LFtip": (
        (-0.148 - 0.040, -0.528 - 0.020, 1.434 - 0.010),
        (-0.148 + 0.040, -0.528 + 0.020, 1.434 + 0.010),
    ),
}

name_tip_dict = {
    "Thumb": "THtip",
    "Index": "IFtip",
    "Middle": "MFtip",
    "Ring": "RFtip",
    "Little": "LFtip",
}

register_env_with_variants(
    id="MuscleHandReachFixed-v0",
    entry_point="envs.reach:MuscleReachEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "target_reach_range": tip_range_fixed_dict,
        "normalize_act": True,
        "far_th": 0.35,
    },
)

register_env_with_variants(
    id="MuscleHandReachRandom-v0",
    entry_point="envs.reach:MuscleReachEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pose.xml",
        "target_reach_range": tip_range_random_dict,
        "normalize_act": True,
        "far_th": 0.35,
    },
)

for i in range(1, len(name_tip_dict) + 1):
    for name_list in list(combinations(name_tip_dict, i)):
        tip_list = [name_tip_dict[name] for name in name_list]
        register_env_with_variants(
            id=f"myoHand{''.join(name_list)}ReachFixed-v0",
            entry_point="envs.reach:CleanReachEnv",
            max_episode_steps=100,
            kwargs={
                "model_path": local_myosuite_asset_path
                + "/assets/hand/myohand_pose.xml",
                "target_reach_range": {
                    tip: tip_range_fixed_dict[tip] for tip in tip_list
                },
                "normalize_act": True,
                "far_th": 0.35,
            },
        )
        register_env_with_variants(
            id=f"MuscleHand{''.join(name_list)}ReachFixed-v0",
            entry_point="envs.reach:MuscleReachEnv",
            max_episode_steps=100,
            kwargs={
                "model_path": local_myosuite_asset_path
                + "/assets/hand/myohand_pose.xml",
                "target_reach_range": {
                    tip: tip_range_fixed_dict[tip] for tip in tip_list
                },
                "normalize_act": True,
                "far_th": 0.35,
            },
        )
        register_env_with_variants(
            id=f"myoHand{''.join(name_list)}ReachRandom-v0",
            entry_point="envs.reach:CleanReachEnv",
            max_episode_steps=100,
            kwargs={
                "model_path": local_myosuite_asset_path
                + "/assets/hand/myohand_pose.xml",
                "target_reach_range": {
                    tip: tip_range_random_dict[tip] for tip in tip_list
                },
                "normalize_act": True,
                "far_th": 0.35,
            },
        )
        register_env_with_variants(
            id=f"MuscleHand{''.join(name_list)}ReachRandom-v0",
            entry_point="envs.reach:MuscleReachEnv",
            max_episode_steps=100,
            kwargs={
                "model_path": local_myosuite_asset_path
                + "/assets/hand/myohand_pose.xml",
                "target_reach_range": {
                    tip: tip_range_random_dict[tip] for tip in tip_list
                },
                "normalize_act": True,
                "far_th": 0.35,
            },
        )

# MyoChallenge Baoding: muscle observation static RSI
register_env_with_variants(
    id="MuscleBaodingP0-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (1e6, 1e6),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum1-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (30, 30),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0.9 * np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum2-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (25, 25),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0.7 * np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum3-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (20, 20),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0.5 * np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum4-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (15, 15),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0.3 * np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum5-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (10, 10),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0.1 * np.pi,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum6-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (9, 9),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum7-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (8, 8),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum8-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (7, 7),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

register_env_with_variants(
    id="MuscleBaodingCurriculum9-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (6, 6),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

# MyoChallenge Baoding: muscle observation phase 1 CCW rotation
register_env_with_variants(
    id="MuscleBaodingP1-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (5, 5),
        "task_choice": "ccw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

# MyoChallenge Baoding: muscle observation phase 1 CW rotation
register_env_with_variants(
    id="MuscleBaodingP1CW-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (5, 5),
        "task_choice": "cw",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

# MyoChallenge Baoding: muscle observation phase 1 both directions
register_env_with_variants(
    id="MuscleBaodingP1Both-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (5, 5),
        "task_choice": "random_dir",
        "limit_init_angle": 0,
        "limit_sds_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

# MyoChallenge Baoding: muscle observation random task, random physics and overlapping balls
register_env_with_variants(
    id="MuscleBaodingP2Overlap-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
        "limit_init_angle": 0,
        "initial_phase": np.pi / 2,
    },
)

# MyoChallenge Baoding: muscle observation phase 2
register_env_with_variants(
    id="MuscleBaodingP2-v1",
    entry_point="envs.baoding:MuscleBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_baoding.xml",
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
    },
)

# MyoChallenge Reorient: joint observation pi/2 range
register_env_with_variants(
    id="CustomMyoChallengeDieReorientP0-v0",
    entry_point="envs.reorient:CustomReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_die.xml",
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": np.inf,  # ignore position error threshold
        "goal_pos": (0, 0),  # 0 cm
        "goal_rot": (-0.785, 0.785),  # +-45 degrees
    },
)

# MyoChallenge Reorient: muscle observation pi/2 range
register_env_with_variants(
    id="MuscleDieReorientP0-v0",
    entry_point="envs.reorient:MuscleReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_die.xml",
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": np.inf,  # ignore position error threshold
        "goal_pos": (0, 0),  # 0 cm
        "goal_rot": (-0.785, 0.785),  # +-45 degrees
    },
)

# MyoChallenge Reorient: muscle observation pi range
register_env_with_variants(
    id="MuscleDieReorientP1-v0",
    entry_point="envs.reorient:MuscleReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_die.xml",
        "normalize_act": True,
        "frame_skip": 5,
        "goal_pos": (-0.010, 0.010),  # +- 1 cm
        "goal_rot": (-1.57, 1.57),  # +-90 degrees
    },
)

# MyoChallenge Reorient: muscle observation 2pi range random physics
register_env_with_variants(
    id="MuscleDieReorientP2-v0",
    entry_point="envs.reorient:MuscleReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_die.xml",
        "normalize_act": True,
        "frame_skip": 5,
        # Randomization in goals
        "goal_pos": (-0.020, 0.020),  # +- 2 cm
        "goal_rot": (-3.14, 3.14),  # +-180 degrees
        # Randomization in physical properties of the die
        "obj_size_change": 0.007,  # +-7mm delta change in object size
        "obj_mass_range": (0.050, 0.250),  # 50gms to 250 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
    },
)

# Gait Torso Reaching ==============================
register_env_with_variants(
    id="MuscleLegDemo-v0",
    entry_point="envs.walk:MuscleLegsReachEnv",
    max_episode_steps=3000,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/leg/myolegs.xml",
        "target_reach_range": {
            # "pelvis": ((-.05, -.05, .75), (0.05, 0.05, .92)), # stabalize around mean posture
            "pelvis": ((-0.005, -0.005, 0.9), (0.005, 0.005, 0.9)),  # stand still
        },
        "normalize_act": True,
        "far_th": 0.44,
    },
)

# Gait Torso Walking ==============================
register_env_with_variants(
    id="MuscleLegWalk-v0",
    entry_point="envs.walk:MuscleLegsWalkEnv",
    max_episode_steps=1000,
    kwargs={
        "model_path": local_myosuite_asset_path
        + "/../../simhive/myo_sim/leg/myolegs.xml",
        "normalize_act": True,
        "min_height": 0.8,  # minimum center of mass height before reset
        "max_rot": 0.8,  # maximum rotation before reset
        "hip_period": 100,  # desired periodic hip angle movement
        "reset_type": "init",  # none, init, random
        "target_x_vel": 0.0,  # desired x velocity in m/s
        "target_y_vel": 1.2,  # desired y velocity in m/s
        "target_rot": None,  # if None then the initial root pos will be taken, otherwise provide quat
    },
)


# Pen twirl
register_env_with_variants(
    id="CustomMyoHandPenTwirlRandom-v0",
    entry_point="envs.pen:CustomPenEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pen.xml",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

register_env_with_variants(
    id="MusclePenTwirl-v0",
    entry_point="envs.pen:MusclePenEnv",
    max_episode_steps=100,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/hand/myohand_pen.xml",
        "normalize_act": True,
        "frame_skip": 5,
    },
)

# Relocate phase 2
register_env_with_variants(
    id="CustomRelocate-v0",
    entry_point="envs.relocate:CustomRelocateEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/arm/myoarm_relocate.xml",
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": 0.1,  # cover entire base of the receptacle
        "rot_th": np.inf,  # ignore rotation errors
        "qpos_noise_range": 0.01,  # jnt initialization range
        "target_xyz_range": {"high": [0.3, -0.1, 1.05], "low": [0.0, -0.45, 0.9]},
        "target_rxryrz_range": {"high": [0.2, 0.2, 0.2], "low": [-0.2, -0.2, -0.2]},
        "obj_xyz_range": {"high": [0.1, -0.15, 1.0], "low": [-0.1, -0.35, 1.0]},
        "obj_geom_range": {"high": [0.025, 0.025, 0.025], "low": [0.015, 0.015, 0.015]},
        "obj_mass_range": {"high": 0.200, "low": 0.050},  # 50gms to 200 gms
        "obj_friction_range": {
            "high": [1.2, 0.006, 0.00012],
            "low": [0.8, 0.004, 0.00008],
        },
    },
)
register_env_with_variants(
    id="MuscleRelocate-v0",
    entry_point="envs.relocate:MuscleRelocateEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": local_myosuite_asset_path + "/assets/arm/myoarm_relocate.xml",
        "normalize_act": True,
        "frame_skip": 5,
        "pos_th": 0.1,  # cover entire base of the receptacle
        "rot_th": np.inf,  # ignore rotation errors
        "qpos_noise_range": 0.01,  # jnt initialization range
        "target_xyz_range": {"high": [0.3, -0.1, 1.05], "low": [0.0, -0.45, 0.9]},
        "target_rxryrz_range": {"high": [0.2, 0.2, 0.2], "low": [-0.2, -0.2, -0.2]},
        "obj_xyz_range": {"high": [0.1, -0.15, 1.0], "low": [-0.1, -0.35, 1.0]},
        "obj_geom_range": {"high": [0.025, 0.025, 0.025], "low": [0.015, 0.015, 0.015]},
        "obj_mass_range": {"high": 0.200, "low": 0.050},  # 50gms to 200 gms
        "obj_friction_range": {
            "high": [1.2, 0.006, 0.00012],
            "low": [0.8, 0.004, 0.00008],
        },
    },
)
