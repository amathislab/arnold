from stable_baselines3 import SAC, TD3, PPO
from models.ppo.ppo import MultiEnvPPO, PCGradMultiEnvPPO, SeparatedValueFinetunePPO
from algos.bc_ppo import BCPPO, MultiTaskBCPPO
from sb3_contrib import RecurrentPPO


class AlgoFactory:
    algo_class_dict = {
        "ppo": PPO,
        "multi_env_ppo": MultiEnvPPO,
        "separated_value_ppo": SeparatedValueFinetunePPO,
        "pcgrad_ppo": PCGradMultiEnvPPO,
        "recurrent_ppo": RecurrentPPO,
        "sac": SAC,
        "td3": TD3,
        "bc_ppo": BCPPO,
        "multi_task_bc_ppo": MultiTaskBCPPO,
    }

    @staticmethod
    def get_algo_class(algo: str):
        if algo not in AlgoFactory.algo_class_dict:
            raise ValueError(
                "Unknown algorithm ",
                algo,
                " available algorithms are: ",
                AlgoFactory.algo_class_dict.keys(),
            )
        return AlgoFactory.algo_class_dict[algo]
