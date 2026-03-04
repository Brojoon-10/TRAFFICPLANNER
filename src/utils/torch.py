# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import torch
import numpy as np

c2c = lambda tensor: tensor.detach().cpu().numpy()

def get_device():
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def save_state(file_out, model, optimizer, cur_epoch=0, min_val_loss=float('Inf'), ignore_keys=None, global_step=0):
    model_state_dict = model.state_dict()
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}

    full_checkpoint_dict = {
        'model' : model_state_dict,
        'optim' : optimizer.state_dict(),
        'epoch' : cur_epoch,
        'min_val_loss' : min_val_loss,
        'global_step' : global_step,
    }
    torch.save(full_checkpoint_dict, file_out)

def load_state(load_path, model, optimizer=None, map_location=None, ignore_keys=None):
    if not os.path.exists(load_path):
        print('Could not find checkpoint at path ' + load_path)

    full_checkpoint_dict = torch.load(load_path, map_location=map_location)
    model_state_dict = full_checkpoint_dict['model']
    optim_state_dict = full_checkpoint_dict['optim']
    
    if ignore_keys is not None:
        model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}
        
    # overwrite entries in the existing state dict
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if ignore_keys is not None:
        missing_keys = [k for k in missing_keys if k.split('.')[0] not in ignore_keys]
        unexpected_keys = [k for k in unexpected_keys if k.split('.')[0] not in ignore_keys]
    if len(missing_keys) > 0:
        print('WARNING: The following keys could not be found in the given state dict - ignoring...')
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print('WARNING: The following keys were found in the given state dict but not in the current model - ignoring...')
        print(unexpected_keys)

    # load optimizer weights
    if optimizer is not None:
        optimizer.load_state_dict(optim_state_dict)

    global_step = full_checkpoint_dict.get('global_step', 0)
    return full_checkpoint_dict['epoch'], full_checkpoint_dict['min_val_loss'], global_step

def calc_conv_out(in_size, kernel_size, stride, padding_size=0):
    return int(((in_size - kernel_size - 2*padding_size) // stride) + 1)

def compute_kl_weight(global_step, anneal_steps, final_kl_weight, kl_beta_floor=0.0):
    ''' Step-based linear KL annealing with floor.
        global_step: current training step (batch count)
        anneal_steps: total steps for full ramp (0 = immediate full weight)
        final_kl_weight: target β at end of annealing
        kl_beta_floor: minimum β value (never zero, prevents decoder from ignoring z)
        Returns: kl_beta_floor + (final - floor) * min(step/anneal_steps, 1.0)
    '''
    if anneal_steps <= 0:
        return final_kl_weight
    ramp = min(1.0, float(global_step) / anneal_steps)
    return kl_beta_floor + (final_kl_weight - kl_beta_floor) * ramp

def tensor_clamp(x, xmin, xmax):
    # https://github.com/pytorch/pytorch/issues/2793
    return torch.max(torch.min(x, xmax), xmin)