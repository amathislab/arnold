import os
from itertools import combinations

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
MASK_SUFFIX = "_mask"
OBS_KEY = "obs"
OBS_ID_KEY = "obs_ids"
ACTION_ID_KEY = "action_ids"
VALUE_KEY = "value"
PADDING_KEY = "padding"
MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
VOCABULARY_FILE_NAME = "vocabulary.json"
ARGS_FILE_NAME = "args.json"
STOCHASTIC_DF_NAME = "df_stochastic.pkl"
DETERMINISTIC_DF_NAME = "df_deterministic.pkl"
MODEL_FILE_NAME = "model.zip"
ENV_FILE_NAME = "env.pkl"
ENV_CONFIG_NAME = "env_config.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"
ENV_CONFIG_PATH = os.path.join(ROOT_DIR, "data", "env_configs")
EXPERT_CONFIG_PATH = os.path.join(ROOT_DIR, "data", "expert_configs")
EXPERT_PER_CONFIG_PATH = os.path.join(EXPERT_CONFIG_PATH, "default_experts.json")
EXPERT_POLICIES_PATH = os.path.join(ROOT_DIR, "data", "expert_policies")


POSE_INFO_KEYS = {
    "pose",
    "bonus",
    "penalty",
    "act_reg",
    "done",
    "solved",
    "sparse",
}

REACH_INFO_KEYS = {
    "reach",
    "bonus",
    "penalty",
    "act_reg",
    "done",
    "solved",
    "sparse",
}

BAODING_INFO_KEYS = {
    "pos_dist_1",
    "pos_dist_2",
    "act_reg",
    "alive",
    "solved",
}

PEN_INFO_KEYS = {
    "pos_align",
    "rot_align",
    "act_reg",
    "alive",
    "solved",
}

REORIENT_INFO_KEYS = {
    "pos_dist",
    "rot_dist",
    "pos_dist_diff",
    "rot_dist_diff",
    "act_reg",
    "alive",
    "solved",
}

WALK_INFO_KEYS = {
    "done",
    "solved",
    "sparse",
    "vel_reward",
    "cyclic_hip",
    # "ref_foot",
    "joint_angle_rew",
    "act_mag",
}

RELOCATE_INFO_KEYS = {
    "pos_dist",
    "act_reg",
    "done",
    "solved",
    "sparse",
}

KINESIS_INFO_KEYS = {
    "tracking_reward",
    "upright_reward",
    "energy_reward",
    "solved",
    "alive",
}

NAME_FINGER_TIP_DICT = {
    "Thumb": "THtip",
    "Index": "IFtip",
    "Middle": "MFtip",
    "Ring": "RFtip",
    "Little": "LFtip",
}

ENV_NAME_TO_ID = {
    # "FingerPoseFixed": "myoFingerPoseFixed-v0",
    # "FingerPoseRandom": "myoFingerPoseRandom-v0",
    # "FingerReachFixed": "myoFingerReachFixed-v0",
    # "FingerReachRandom": "myoFingerReachRandom-v0",
    "HandPoseFixed": "myoHandPoseFixed-v0",
    "HandPoseRandom": "myoHandPoseRandomHalfRange-v0",
    # "HandReachFixed": "myoHandReachFixed-v0",
    "HandReachRandom": "myoHandReachRandom-v0",
    # "ElbowPoseFixed": "myoElbowPose1D6MFixed-v0",
    # "ElbowPoseRandom": "myoElbowPose1D6MRandom-v0",
    # "HandKeyTurnFixed": "myoHandKeyTurnFixed-v0",
    # "HandKeyTurnRandom": "myoHandKeyTurnRandom-v0",
    # "BaodingBallsP1": "myoChallengeBaodingP1-v1",
    # "CustomMyoBaodingBallsP1": "CustomMyoChallengeBaodingP1-v1",
    "CustomMyoReorientP0": "CustomMyoChallengeDieReorientP0-v0",
    # "CustomMyoReorientP1": "CustomMyoChallengeDieReorientP1-v0",
    # "CustomMyoReorientP2": "CustomMyoChallengeDieReorientP2-v0",
    # "MyoBaodingBallsP2": "myoChallengeBaodingP2-v1",
    # "CustomMyoBaodingBallsP2": "CustomMyoChallengeBaodingP2-v1",
    # "MixtureModelBaodingEnv": "MixtureModelBaoding-v1",
    # "CustomMyoElbowPoseFixed": "CustomMyoElbowPoseFixed-v0",
    # "CustomMyoElbowPoseRandom": "CustomMyoElbowPoseRandom-v0",
    # "CustomMyoFingerPoseFixed": "CustomMyoFingerPoseFixed-v0",
    # "CustomMyoFingerPoseRandom": "CustomMyoFingerPoseRandom-v0",
    # "CustomMyoHandPoseFixed": "CustomMyoHandPoseFixed-v0",
    "CustomMyoHandPenTwirlRandom": "CustomMyoHandPenTwirlRandom-v0",
    "CustomMyoHandPoseRandom": "CustomMyoHandPoseRandom-v0",
    "CustomMyoPenTwirlRandom": "CustomMyoHandPenTwirlRandom-v0",
    "CleanBaodingBalls": "CleanBaodingBalls-v1",
    "CleanBaodingBallsP1CCW": "CleanBaodingBallsP1CCW-v1",
    "CleanBaodingBallsP1CW": "CleanBaodingBallsP1CW-v1",
    "CleanBaodingBallsP2": "CleanBaodingBallsP2-v1",
    "CleanBaodingBallsP2Overlap": "CleanBaodingBallsP2Overlap-v1",
    "CustomRelocate": "CustomRelocate-v0",
    "CleanElbowPose": "CleanElbowPose-v0",
    "CleanFingerPose": "CleanFingerPose-v0",
    # Muscle environments
    "MuscleElbowPoseFixed": "MuscleElbowPoseFixed-v0",
    "MuscleElbowPoseRandom": "MuscleElbowPoseRandom-v0",
    "MuscleFingerPoseFixed": "MuscleFingerPoseFixed-v0",
    "MuscleFingerPoseRandom": "MuscleFingerPoseRandom-v0",
    "MuscleHandPoseFixed": "MuscleHandPoseFixed-v0",
    "MuscleHandPoseRandom": "MuscleHandPoseRandom-v0",
    "MuscleHandPoseRandomHalfRange": "MuscleHandPoseRandomHalfRange-v0",
    "MuscleFingerReachFixed": "MuscleFingerReachFixed-v0",
    "MuscleFingerReachRandom": "MuscleFingerReachRandom-v0",
    "MuscleHandReachFixed": "MuscleHandReachFixed-v0",
    "MuscleHandReachRandom": "MuscleHandReachRandom-v0",
    "MuscleBaodingP0": "MuscleBaodingP0-v1",
    "MuscleBaodingP1": "MuscleBaodingP1-v1",
    "MuscleBaodingP1CW": "MuscleBaodingP1CW-v1",
    "MuscleBaodingP1Both": "MuscleBaodingP1Both-v1",
    "MuscleBaodingP2Overlap": "MuscleBaodingP2Overlap-v1",
    "MuscleBaodingP2": "MuscleBaodingP2-v1",
    "MuscleReorientP0": "MuscleDieReorientP0-v0",
    "MuscleReorientP1": "MuscleDieReorientP1-v0",
    "MuscleReorientP2": "MuscleDieReorientP2-v0",
    "MusclePenTwirl": "MusclePenTwirl-v0",
    "MuscleLegsStand": "MuscleLegDemo-v0",
    "MuscleLegsWalk": "MuscleLegWalk-v0",
    "MuscleRelocate": "MuscleRelocate-v0",
    # PyBullet environments
    "WalkerBulletEnv": "Walker2DBulletEnv-v0",
    "HalfCheetahBulletEnv": "HalfCheetahBulletEnv-v0",
    "AntBulletEnv": "AntBulletEnv-v0",
    "HopperBulletEnv": "HopperBulletEnv-v0",
    "HumanoidBulletEnv": "HumanoidBulletEnv-v0",
    "HumanoidFlagrunBulletEnv": "HumanoidFlagrunBulletEnv-v0",
    "HumanoidFlagrunHarderBulletEnv": "HumanoidFlagrunHarderBulletEnv-v0",
    # Gym environments
    "CartPole": "CartPole-v0",
    "MountainCarContinuous": "MountainCarContinuous-v0",
    # Kinesis environments
    "Kinesis": "Kinesis-v0",
    "MuscleKinesis": "MuscleKinesis-v0",
}

ENV_INFO = {
    "MuscleElbowPoseFixed": POSE_INFO_KEYS,
    "MuscleElbowPoseRandom": POSE_INFO_KEYS,
    "MuscleFingerPoseFixed": POSE_INFO_KEYS,
    "MuscleFingerPoseRandom": POSE_INFO_KEYS,
    "MuscleHandPoseFixed": POSE_INFO_KEYS,
    "MuscleHandPoseRandom": POSE_INFO_KEYS,
    "MuscleHandPoseRandomHalfRange": POSE_INFO_KEYS,
    "MuscleFingerReachFixed": REACH_INFO_KEYS,
    "MuscleFingerReachRandom": REACH_INFO_KEYS,
    "MuscleHandReachFixed": REACH_INFO_KEYS,
    "MuscleHandReachRandom": REACH_INFO_KEYS,
    "MuscleHandThumbReachFixed": REACH_INFO_KEYS,
    "MuscleHandThumbReachRandom": REACH_INFO_KEYS,
    "MuscleHandIndexReachFixed": REACH_INFO_KEYS,
    "MuscleHandIndexReachRandom": REACH_INFO_KEYS,
    "MuscleHandMiddleReachFixed": REACH_INFO_KEYS,
    "MuscleHandMiddleReachRandom": REACH_INFO_KEYS,
    "MuscleHandRingReachFixed": REACH_INFO_KEYS,
    "MuscleHandRingReachRandom": REACH_INFO_KEYS,
    "MuscleHandLittleReachFixed": REACH_INFO_KEYS,
    "MuscleHandLittleReachRandom": REACH_INFO_KEYS,
    "MuscleBaodingP0": BAODING_INFO_KEYS,
    "MuscleBaodingP1": BAODING_INFO_KEYS,
    "MuscleBaodingP1CW": BAODING_INFO_KEYS,
    "MuscleBaodingP1Both": BAODING_INFO_KEYS,
    "MuscleBaodingP2Overlap": BAODING_INFO_KEYS,
    "MuscleBaodingP2": BAODING_INFO_KEYS,
    "MuscleReorientP0": REORIENT_INFO_KEYS,
    "MuscleReorientP1": REORIENT_INFO_KEYS,
    "MuscleReorientP2": REORIENT_INFO_KEYS,
    "MusclePenTwirl": PEN_INFO_KEYS,
    "MuscleLegsStandEnv": REACH_INFO_KEYS,
    "MuscleLegsWalkEnv": WALK_INFO_KEYS,
    "CustomRelocate": RELOCATE_INFO_KEYS,
    "MuscleRelocate": RELOCATE_INFO_KEYS,
    "Kinesis": KINESIS_INFO_KEYS,
    "MuscleKinesis": KINESIS_INFO_KEYS,
}

prefix_list = ["myoHand", "MuscleHand"]
suffix_list = ["ReachFixed", "ReachRandom"]

reach_id_dict = {}
reach_info_dict = {}
for i in range(1, len(NAME_FINGER_TIP_DICT) + 1):
    for name_list in list(combinations(NAME_FINGER_TIP_DICT, i)):
        for prefix in prefix_list:
            for suffix in suffix_list:
                name = f"{prefix}{''.join(name_list)}{suffix}"
                env_id = name + "-v0"
                reach_id_dict[name] = env_id
                reach_info_dict[name] = REACH_INFO_KEYS

ENV_NAME_TO_ID.update(reach_id_dict)
ENV_INFO.update(reach_info_dict)

baoding_id_dict = {}
baoding_info_dict = {}

for curr_idx in range(1, 10):
    name = f"MuscleBaodingCurriculum{curr_idx}"
    env_id = name + "-v1"
    baoding_id_dict[name] = env_id
    baoding_info_dict[name] = BAODING_INFO_KEYS

ENV_NAME_TO_ID.update(baoding_id_dict)
ENV_INFO.update(baoding_info_dict)

os.environ["ALLOW_WEIGHT_RECORDING"] = "True"
