import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class EnvDumpCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _on_step(self):
        env_path = os.path.join(self.save_path, "training_env.pkl")
        if self.verbose > 0:
            print("Saving the training environment to path ", env_path)
        self.training_env.save(env_path)
        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}

    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}

    def _on_step(self):
        for key in self.info_keywords:
            vals = [
                info.get(key)
                for info in self.locals["infos"]
                if info.get(key) is not None
        ]
            self.rollout_info[key].extend(vals)
        return True

    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record(key, np.mean(self.rollout_info[key]))


class CustomCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path, exclude=["_last_obs", "_last_original_obs"])
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path(
                    "replay_buffer_", extension="pkl"
                )
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(
                        f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                    )

            if (
                self.save_vecnormalize
                and self.model.get_vec_normalize_env() is not None
            ):
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path(
                    "vecnormalize_", extension="pkl"
                )
                vecnormalize = self.model.get_vec_normalize_env()

                # remove old obs to save disk space, then restore
                old_obs = vecnormalize.old_obs
                vecnormalize.old_obs = None
                vecnormalize.save(vec_normalize_path)  # type: ignore[union-attr]
                vecnormalize.old_obs = old_obs

                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
