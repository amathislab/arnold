import torch
import numpy as np
import torch.nn as nn
from torch.utils import data as th_data
import os

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        if os.environ["ALLOW_WEIGHT_RECORDING"] == "True" :
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

        # print("value1", os.environ["ALLOW_WEIGHT_RECORDING"])
        outputs, attention_weights = forward_orig(*args, **kwargs)
        # print("value2", os.environ["ALLOW_WEIGHT_RECORDING"])
        return outputs, attention_weights

    m.forward = wrap


class SaveOutput():
    def __init__(self):
        self.outputs = {}

    def my_call(self, module, module_in, module_out, module_name):

        if module_out[1] is not None:
            if module_name not in self.outputs :
                self.outputs[module_name] = []
            self.outputs[module_name].append(module_out[1])
        
    def get_named_caller(self, module_name) :

        print("Get named caller", module_name)

        def func(module, module_in, module_out) :

                self.my_call(module, module_in, module_out, module_name)
        
        return func

    def clear(self):
        self.outputs = {}

class ContinuousBatchSampler(th_data.Sampler):
    def __init__(self, data_source, batch_size, time_skip=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.time_skip = time_skip
        self.generator = None

        self.indices = torch.arange(len(self.data_source))
        self.batch_range = self.time_skip * (self.batch_size-1) + 1
        self.n = len(self.indices) - self.batch_range + 1

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        sample_perm = torch.randperm(self.n, generator=generator)[:self.n // self.batch_size]
        # print("Sampler size:", sample_perm)
        for i in sample_perm:
            yield self.indices[i:i + self.batch_range:self.time_skip].tolist()

    def __len__(self):
        return self.n // self.batch_size


def merge_dimensions(arr: np.ndarray, l: int, r: int) -> np.ndarray :

    assert(l < r)
    assert(l >= 0)
    assert(r <= len(arr.shape))

    new_shape = list(arr.shape[:l]) + [np.prod(arr.shape[l:r])] + list(arr.shape[r:])
    return arr.reshape(new_shape)

def select_by_mask(tensor, mask) :

    dim_mask = len(mask.shape)
    assert(tensor.shape[:dim_mask] == mask.shape)
    assert(mask.dtype == torch.bool)
    return tensor[mask].view(*mask.shape[:-1], -1, *tensor.shape[dim_mask:])

def generate_token_mask_vectorized(N, M, X):
    """
    Generate a tensor of shape (N, M) with X ones per row.
    """
    if X > M:
        raise ValueError("X cannot be greater than M")
    
    # Create a tensor of zeros
    array = torch.zeros((N, M), dtype=torch.bool)
    
    # Create random indices for X-1 ones (excluding the last column)
    rand_indices = torch.topk(torch.rand(N, M - 1), X - 1, dim=1).indices
    
    # Fill the selected positions with 1
    row_indices = torch.arange(N).unsqueeze(1).expand_as(rand_indices)
    array[row_indices, rand_indices] = 1
    
    # Ensure the last column is 1
    array[:, -1] = 1
    
    return array


def register_output_weight(model, output_saver) :

    assert(isinstance(
        model, (
            nn.TransformerEncoder,
            nn.TransformerDecoder
        )
    ))
    patch_attention(model.layers[-1].self_attn)
    patch_attention(model.layers[-1].multihead_attn)
    model.layers[-1].self_attn.register_forward_hook(output_saver.get_named_caller("self_attn"))
    model.layers[-1].multihead_attn.register_forward_hook(output_saver.get_named_caller("cross_attn"))


def my_safe_to_tensor(array, **kwargs) -> torch.Tensor:
    """Converts a NumPy array to a PyTorch tensor.

    The data is copied in the case where the array is non-writable. Unfortunately if
    you just use `th.as_tensor` for this, an ugly warning is logged and there's
    undefined behavior if you try to write to the tensor.

    Args:
        array: The array to convert to a PyTorch tensor.
        kwargs: Additional keyword arguments to pass to `th.as_tensor`.

    Returns:
        A PyTorch tensor with the same content as `array`.
    """
    if isinstance(array, torch.Tensor):
        if "device" in kwargs:
            return array.to(kwargs["device"])
        else:
            return array

    if not array.flags.writeable:
        array = array.copy()

    return torch.as_tensor(array, **kwargs)
