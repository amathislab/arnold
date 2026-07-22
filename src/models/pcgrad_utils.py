import torch
import random


def param_inner_prod(params_left, params_right):
    return sum(
        [
            torch.sum(p_left * p_right)
            for p_left, p_right in zip(params_left, params_right)
            if p_left is not None and p_right is not None
        ]
    )


def param_sum(params_list):
    return [
        torch.sum(torch.stack(p_tuple), axis=0)
        for p_tuple in zip(*params_list)
        if all([p is not None for p in p_tuple])
    ]


def param_frac_by_scalar(params, val):
    return [p / val for p in params if p is not None]


def param_prod_by_scalar(params, val):
    return [p * val for p in params if p is not None]


def param_copy(params):
    return [p.clone() if p is not None else None for p in params]


def get_param_grad(params):
    return [p.grad for p in params]


def set_param_grad(params, grads):
    for p, g in zip(params, grads):
        p.grad = g


def proj_grad(grad_list):
    grad_norms = [param_inner_prod(g, g) for g in grad_list]
    pc_grad_list = []  # List of gradients removing conflicting projections
    for i1, grad1 in enumerate(grad_list):
        pc_grad = param_copy(grad1)
        # Go through all the other gradients in random order
        idx_list = list(range(len(grad_list)))
        random.shuffle(idx_list)
        for i2 in idx_list:
            if i2 != i1:
                proj_coef = param_inner_prod(pc_grad, grad_list[i2]) / grad_norms[i2]
                if proj_coef < 0:  # If conflicting
                    conflict_correction = param_prod_by_scalar(grad_list[i2], -proj_coef)
                    pc_grad = param_sum([pc_grad, conflict_correction])
        pc_grad_list.append(pc_grad)
    return pc_grad_list
