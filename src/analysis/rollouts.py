"""Analysis adapters for the existing policy and environment interfaces."""


def find_vecnormalize(model_path):
    # Keep checkpoint naming/fallback behavior shared with the MT benchmark.
    from benchmark_multi_task_mlp import find_vecnormalize as locate
    return locate(str(model_path))


def simulation_env(env):
    base = env.unwrapped
    return getattr(base, "gym_env", base).unwrapped


def get_episode_horizon(env):
    pending, seen = [env], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        attrs = vars(current)
        horizon = attrs.get("_max_episode_steps")
        if horizon is None:
            horizon = getattr(attrs.get("spec"), "max_episode_steps", None)
        if horizon is not None:
            return int(horizon)
        pending.extend(attrs[key] for key in ("env", "gym_env") if key in attrs)
    raise ValueError("The environment has no fixed episode horizon")


def load_policy(path, device="cpu", vocabulary=None):
    from models.ppo.policies import MuscleTransformerPolicy
    from algos.bc_ppo import MultiTaskBCPPO
    try:
        policy = MuscleTransformerPolicy.load(str(path), device=device)
    except Exception:
        policy = MultiTaskBCPPO.load(str(path), device=device,
                                   custom_objects={"vocabulary": vocabulary}).policy
    policy.to(device)
    policy.set_training_mode(False)
    return policy
