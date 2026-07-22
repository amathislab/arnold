import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import List, Tuple
from vocabulary import VOCABULARY


class RunningMeanStdFloat32(RunningMeanStd):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon


class SpecsRunningMeanStd:
    """obs_id_signatures: obs_ids per environment"""

    def __init__(
        self,
        obs_id_signatures: List[np.ndarray],
        obs_mask: np.ndarray,
        epsilon: float = 1e-4,
        obs_shape: Tuple[int, ...] = (),
    ):
        self.sig_ids_dict = {(0,): 0}  # Maps an obs signature to an id
        self.obs_shape = obs_shape
        self.obs_spec = self._signatures_to_specs(obs_id_signatures)
        self.obs_mask = obs_mask
        self.epsilon = epsilon
        self.mean_mat = np.zeros((len(self.sig_ids_dict), *obs_shape[1:]), np.float32)
        self.var_mat = np.ones((len(self.sig_ids_dict), *obs_shape[1:]), np.float32)
        self.count_mat = epsilon * np.ones(
            len(self.sig_ids_dict)
        )  # count per signature

    def _signatures_to_specs(self, signatures):
        # First convert the signatures to tuples
        num_envs = len(signatures)
        obs_len = self.obs_shape[0]
        obs_specs = np.zeros((num_envs, obs_len), dtype=np.int32)

        for env_id, env_sig_vec in enumerate(signatures):
            for obs_id, obs_sig in enumerate(env_sig_vec):
                if (obs_sig != 0).any():
                    obs_sig_tuple = tuple(sorted(obs_sig[obs_sig != 0]))
                    sig_id = self.sig_ids_dict.get(obs_sig_tuple)
                    if sig_id is None:
                        sig_id = len(self.sig_ids_dict)
                        self.sig_ids_dict.update({obs_sig_tuple: sig_id})
                    obs_specs[env_id, obs_id] = sig_id
        return obs_specs

    @property
    def mean(self):
        return self.mean_mat[self.obs_spec]

    @property
    def var(self):
        return self.var_mat[self.obs_spec]

    @property
    def count(self):
        return self.count_mat[self.obs_spec]

    def mean_single_env(self, env_id: int):
        return self.mean_mat[self.obs_spec[env_id]]

    def var_single_env(self, env_id: int):
        return self.var_mat[self.obs_spec[env_id]]

    def count_single_env(self, env_id: int):
        return self.count_mat[self.obs_spec[env_id]]

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(
            obs_spec=self.obs_spec.copy(),
            obs_mask=self.obs_mask.copy(),
            shape=self.mean_mat.shape,
        )
        new_object.mean = self.mean_mat.copy()
        new_object.var = self.var_mat.copy()
        new_object.count = float(self.count_mat)
        return new_object

    def combine(self, other: "SpecsRunningMeanStd", old_vocabulary=None) -> None:
        """Update the dictionary of  signatures with those that are only present in other.
        Also update the mean and std connected to the signatures present in other.
        If there used to be a different vocabulary, interpret the old signatures accordingly
        """

        # Init the mean, var, count where we will store the values corresponding to the
        # known signatures
        batch_mean = np.zeros_like(self.mean_mat)
        batch_var = np.zeros_like(self.var_mat)
        batch_count = np.zeros_like(self.count_mat)

        unknown_means = []
        unknown_vars = []
        unknown_counts = []

        if old_vocabulary is not None:
            # If there are multiple keys corresponding to the same value,
            # the last key will be used
            reversed_vocabulary = {v: k for k, v in old_vocabulary.items()}

        for signature, sig_id in other.sig_ids_dict.items():
            if old_vocabulary is not None:
                sig_list = [VOCABULARY[reversed_vocabulary[obs_id]] for obs_id in signature]
                signature = tuple(sorted(sig_list))
            if signature not in self.sig_ids_dict:
                # If a signature is unknown, create a new entry
                self.sig_ids_dict.update({signature: len(self.sig_ids_dict)})
                unknown_means.append(other.mean_mat[sig_id][None, ...])
                unknown_vars.append(other.var_mat[sig_id][None, ...])
                unknown_counts.append(other.count_mat[sig_id][None, ...])
            else:
                # Set the batch mean, var, count for the known signatures
                batch_mean[self.sig_ids_dict[signature]] = other.mean_mat[sig_id]
                batch_var[self.sig_ids_dict[signature]] = other.var_mat[sig_id]
                batch_count[self.sig_ids_dict[signature]] = other.count_mat[sig_id]

        # Update the mean, var, count for the known signatures
        self.update_from_moments(batch_mean, batch_var, batch_count)
        self.mean_mat = np.concatenate([self.mean_mat, *unknown_means], axis=0)
        self.var_mat = np.concatenate([self.var_mat, *unknown_vars], axis=0)
        self.count_mat = np.concatenate([self.count_mat, *unknown_counts], axis=0)

    def update(self, arr: np.ndarray) -> None:
        batch_count, batch_mean, batch_var = self.compute_batch_count_mean_var(arr)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: np.ndarray
    ) -> None:
        updated_idx = batch_count > 0
        batch_mean_up = batch_mean[updated_idx]
        batch_var_up = batch_var[updated_idx]
        batch_count_up = batch_count[updated_idx]
        self_mean_up = self.mean_mat[updated_idx]
        self_var_up = self.var_mat[updated_idx]
        self_count_up = self.count_mat[updated_idx]

        delta = batch_mean_up - self_mean_up
        tot_count = self_count_up + batch_count_up
        new_mean = self_mean_up + delta * (batch_count_up / tot_count)[:, None]

        m_a = self_var_up * self_count_up[:, None]
        m_b = batch_var_up * batch_count_up[:, None]
        m_2 = (
            m_a
            + m_b
            + np.square(delta)
            * self_count_up[:, None]
            * batch_count_up[:, None]
            / (self_count_up + batch_count_up)[:, None]
        )
        new_var = m_2 / (self_count_up + batch_count_up)[:, None]
        new_count = batch_count_up + self_count_up
        self.mean_mat[updated_idx] = new_mean
        self.var_mat[updated_idx] = new_var
        self.count_mat[updated_idx] = new_count

    def compute_batch_count_mean_var(self, arr: np.ndarray) -> None:
        masked_arr = arr[self.obs_mask]
        masked_spec = self.obs_spec[self.obs_mask]
        batch_count = np.zeros_like(self.count_mat)
        np.add.at(batch_count, masked_spec, np.ones_like(self.obs_spec)[self.obs_mask])
        updated_idx = batch_count > 0

        batch_mean = np.zeros_like(self.mean_mat)
        np.add.at(batch_mean, masked_spec, masked_arr)
        batch_mean[updated_idx] /= batch_count[updated_idx, None]

        batch_var = np.zeros_like(self.var_mat)
        np.subtract.at(batch_var, masked_spec, 2 * masked_arr)  # -2 \sum_i(x_i)
        batch_var[updated_idx] *= batch_mean[updated_idx]  # -2 x_m \sum_i(x_i)
        batch_var[updated_idx] += (
            batch_count[updated_idx, None] * batch_mean[updated_idx] ** 2
        )  # \sum_i(-2 x_m x_i + x_m^2)
        np.add.at(
            batch_var, masked_spec, masked_arr**2
        )  # \sum_i(x_i^2 - 2 x_m x_i + x_m^2)
        batch_var.clip(
            min=0, max=None, out=batch_var
        )  # avoid some small negative vals due to numerical approx.
        batch_var[updated_idx] /= batch_count[updated_idx, None]
        return batch_count, batch_mean, batch_var
