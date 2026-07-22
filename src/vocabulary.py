KEYS_LIST = [
    "padding",  # 0
    "value",
    "extn",  # Finger muscles
    "adabR",
    "adabL",
    "mflx",  # 5
    "dflx",
    "IFadb",  # Finger joints
    "IFmcp",
    "IFpip",
    "IFdip",  # 10
    "TRIlong",  # Elbow muscles
    "TRIlat",
    "TRImed",
    "BIClong",
    "BICshort",  # 15
    "BRA",
    "r_elbow_flex",  # Elbow joint
    "ECRL",  # Hand muscles
    "ECRB",
    "ECU",  # 20
    "FCR",
    "FCU",
    "PL",
    "PT",
    "PQ",  # 25
    "FDS5",
    "FDS4",
    "FDS3",
    "FDS2",
    "FDP5",  # 30
    "FDP4",
    "FDP3",
    "FDP2",
    "EDC5",
    "EDC4",  # 35
    "EDC3",
    "EDC2",
    "EDM",
    "EIP",
    "EPL",  # 40
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",  # 45
    "LU_RB2",
    "UI_UB2",
    "RI3",
    "LU_RB3",
    "UI_UB3",  # 50
    "RI4",
    "LU_RB4",
    "UI_UB4",
    "RI5",
    "LU_RB5",  # 55
    "UI_UB5",
    "pro_sup",  # Hand joints
    "deviation",
    "flexion",
    "cmc_abduction",  # 60
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",  # 65
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",  # 70
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",  # 75
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
    "THtip",  # Hand reach targets  # 80
    "IFtip",  # This is also shared with finger reach
    "MFtip",
    "RFtip",
    "LFtip",
    "ball",  # baoding objects  # 85
    "die",  # reorient object
    "addbrev",  # legs muscles
    "addlong",
    "addmagDist",
    "addmagIsch",  # 90
    "addmagMid",
    "addmagProx",
    "bflh",
    "bfsh",
    "edl",  # 95
    "ehl",
    "fdl",
    "fhl",
    "gaslat",
    "gasmed",  # 100
    "glmax1",
    "glmax2",
    "glmax3",
    "glmed1",
    "glmed2",  # 105
    "glmed3",
    "glmin1",
    "glmin2",
    "glmin3",
    "grac",  # 110
    "iliacus",
    "perbrev",
    "perlong",
    "piri",
    "psoas",  # 115
    "recfem",
    "sart",
    "semimem",
    "semiten",
    "soleus",  # 120
    "tfl",
    "tibant",
    "tibpost",
    "vasint",
    "vaslat",  # 125
    "vasmed",
    "pelvis",  # legs reach target
    "joint",  # other keywords
    "muscle",
    "tip",  # 130
    "target",
    "error",
    "position",
    "velocity",
    "acceleration",  # 135
    "force",
    "activation",
    "linear",
    "angular",
    "x",  # 140
    "y",
    "z",
    "left",
    "right",
    "object",  # 145
    "id_1",  # object ids
    "id_2",
    "id_3",
    "id_4",
    "id_5",  # 150
    "id_6",
    "id_7",
    "id_8",
    "id_9",
    "id_10",  # 155
    "pen",  # pen object
    "DELT1",  # muscles in the arm not in the hand
    "DELT2",
    "DELT3",
    "SUPSP",
    "INFSP",
    "SUBSC",
    "TMIN",
    "TMAJ",
    "PECM1",
    "PECM2",
    "PECM3",
    "LAT1",
    "LAT2",
    "LAT3",
    "CORB",
    "ANC",
    "SUP",
    "BRD",
    "sternoclavicular_r2",  # joints in the arm not in the hand
    "sternoclavicular_r3",
    "unrotscap_r3",
    "unrotscap_r2",
    "acromioclavicular_r2",
    "acromioclavicular_r3",
    "acromioclavicular_r1",
    "unrothum_r1",
    "unrothum_r3",
    "unrothum_r2",
    "elv_angle",
    "shoulder_elv",
    "shoulder1_r2",
    "shoulder_rot",
    "elbow_flexion",
    "OBJTx",
    "OBJTy",
    "OBJTz",
    "OBJRx",
    "OBJRy",
    "OBJRz",
    "hip_flexion", # legs joint names
    "hip_adduction",
    "hip_rotation",
    "knee_angle_translation2",
    "knee_angle_translation1",
    "knee_angle",
    "knee_angle_rotation2",
    "knee_angle_rotation3",
    "ankle_angle",
    "subtalar_angle",
    "mtp_angle",
    "knee_angle_beta_translation2",
    "knee_angle_beta_translation1",
    "knee_angle_beta_rotation1",
    "foot",
    "toes",
    "root",
    "contact",
]

# Group definitions
FINGER_MUSCLES = [
    "extn", "adabR", "adabL", "mflx", "dflx", "IFadb"
]

FINGER_JOINTS = [
    "IFmcp", "IFpip", "IFdip"
]

ELBOW_MUSCLES = [
    "TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"
]

ELBOW_JOINTS = [
    "r_elbow_flex"
]

HAND_MUSCLES = [
    "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ",
    "FDS5", "FDS4", "FDS3", "FDS2", "FDP5", "FDP4", "FDP3", "FDP2",
    "EDC5", "EDC4", "EDC3", "EDC2", "EDM", "EIP", "EPL", "EPB",
    "FPL", "APL", "OP", "RI2", "LU_RB2", "UI_UB2", "RI3", "LU_RB3",
    "UI_UB3", "RI4", "LU_RB4", "UI_UB4", "RI5", "LU_RB5", "UI_UB5"
]

HAND_JOINTS = [
    "pro_sup", "deviation", "flexion", "cmc_abduction", "cmc_flexion",
    "mp_flexion", "ip_flexion", "mcp2_flexion", "mcp2_abduction",
    "pm2_flexion", "md2_flexion", "mcp3_flexion", "mcp3_abduction",
    "pm3_flexion", "md3_flexion", "mcp4_flexion", "mcp4_abduction",
    "pm4_flexion", "md4_flexion", "mcp5_flexion", "mcp5_abduction",
    "pm5_flexion", "md5_flexion"
]

# Keep existing KEYS_LIST but add group information as a dictionary
GROUPS = {
    "finger_muscles": FINGER_MUSCLES,
    "finger_joints": FINGER_JOINTS,
    "elbow_muscles": ELBOW_MUSCLES,
    "elbow_joints": ELBOW_JOINTS,
    "hand_muscles": HAND_MUSCLES,
    "hand_joints": HAND_JOINTS
}

VOCABULARY = {key: idx for idx, key in enumerate(KEYS_LIST)}
