# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Training script for TrafficPlannerModel
# Based on train_fit_traffic_boston_trans.py with modifications for:
# - TrafficPlannerModel with z_global + z_local architecture
# - Phase-based training (Phase 1: pretrain, Phase 2: finetune)
# - Potential-based collision avoidance loss

import os, argparse, time, csv, subprocess, socket, webbrowser

import gc
import math
import tqdm
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader as GraphDataLoader

# TrafficPlannerModel with z_global + z_local architecture
from models.trafficplanner_model import TrafficPlannerModel
# TrafficPlannerLoss with potential-based loss
from losses.trafficplanner_loss import TrafficPlannerLoss

from nuscenes.nuscenes import NuScenes
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir


def str2bool(v):
    """Convert string to boolean for argparse.
    Handles YAML boolean values: true/false, True/False, yes/no, 1/0
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, save_state, load_state, compute_kl_weight, c2c
from utils.config import get_parser, add_base_args
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from viz_attn_intent import (
    render_map_bg, compute_soft_label_grid, remap_attn_to_gt_frame,
    visualize_map_attention, visualize_intent_codebook, visualize_intent_grid,
    world_to_crop_pixel
)

def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = get_parser('Train TrafficPlanner Model')
    parser = add_base_args(parser)

    # additional dataset options
    parser.add_argument('--scenario_dir', type=str, default=None,
                        help='additional adv gen scenarios to train on')
    parser.add_argument('--data_noise_std', type=float, default=0.0, help='std of noise to add to model input data.')
    parser.add_argument('--seq_interval', type=int, default=1,
                        help='Number of steps between sequences in the dataset (data augmentation control).')

    # Training options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--val_every', type=int, default=3, help='Number of epochs between validations.')
    parser.add_argument('--save_every', type=int, default=10, help='Number of epochs between saving model checkpoint.')
    parser.add_argument('--print_every', type=int, default=10, help='Number of batches between printing stats.')

    # Device options
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (e.g., 0, 1, 2)')

    # Loss plot suffix (will be saved as loss_<suffix>.jpg)
    parser.add_argument('--loss_plot_suffix', type=str, default='trafficplanner',
                        help='Suffix for loss plot filename (will be saved as loss_<suffix>.jpg)')

    # Optimizer options
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for ADAM')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay on params.')

    # LR scheduler (step-based cosine annealing with warmup)
    parser.add_argument('--use_lr_anneal', type=str2bool, default=False,
                        help='Enable step-based cosine annealing LR scheduler with warmup')
    parser.add_argument('--lr_max', type=float, default=None,
                        help='Peak LR after warmup (default: use --lr)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Min LR at end of cosine decay')
    parser.add_argument('--lr_warmup_steps', type=int, default=2500,
                        help='Number of warmup steps (linear 0 → lr_max)')
    parser.add_argument('--lr_total_steps', type=int, default=None,
                        help='Total steps for cosine decay (default: estimated from epochs * data_size / batch_size)')

    # TrafficPlannerModel architecture parameters (past_feat_size, future_feat_size in base_args)
    parser.add_argument('--z_local_size', type=int, default=32,
                        help='Latent dimension for z_local (ego-only reactive)')
    # Phase-based training options
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase: 1=pretrain (z_global), 2=finetune (z_local with frozen z_global)')
    parser.add_argument('--phase1_ckpt', type=str, default=None,
                        help='Checkpoint from Phase 1 to load for Phase 2 training')
    parser.add_argument('--freeze_z_global', type=str2bool, default=False,
                        help='Freeze z_global encoder (automatic in Phase 2)')

    # Losses - base
    parser.add_argument('--loss_kl', type=float, default=0.004, help='KL loss weight (target beta)')
    parser.add_argument('--kl_anneal_steps', type=int, default=5000, help='Steps for linear KL annealing (0 = immediate full weight)')
    parser.add_argument('--kl_floor', type=float, default=0.0, help='Minimum KL beta floor (never zero)')
    parser.add_argument('--kl_free_bits', type=float, default=0.0,
                        help='Free bits per latent dim (0=off). KL_dim = max(KL_dim - free_bits, 0)')
    parser.add_argument('--enc_dropout', type=float, default=0.1,
                        help='Encoder PositionalEncoding dropout (0=off)')
    # dec_past_dropout removed (Enc-Dec context blocks direct past access)
    parser.add_argument('--loss_recon', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--recon_pos_weight', type=float, default=1.0,
                        help='Position (x,y) weight in recon_loss (compensates normalization std)')
    parser.add_argument('--loss_veh_coll_prior', type=float, default=0.05, help='Vehicle collision loss weight for sample from prior')
    parser.add_argument('--loss_env_coll_prior', type=float, default=0.1, help='Map collision loss weight for sample from prior')

    # Losses - TrafficPlanner specific (potential-based)
    parser.add_argument('--use_potential_loss', type=str2bool, default=True,
                        help='Enable potential-based repulsion loss (ego-only)')
    parser.add_argument('--use_veh_potential', type=str2bool, default=False,
                        help='Enable vehicle potential computation (if False, skip computation even if weight > 0)')
    parser.add_argument('--loss_potential_veh', type=float, default=0.1,
                        help='Vehicle repulsion potential loss weight (ego-only)')
    parser.add_argument('--loss_potential_env', type=float, default=0.1,
                        help='Environment boundary potential loss weight (ego-only)')

    # Losses - z_local sparsity (off by default)
    parser.add_argument('--use_sparse_loss', type=str2bool, default=False,
                        help='Enable z_local sparsity loss')
    parser.add_argument('--loss_sparse', type=float, default=0.01,
                        help='z_local sparsity loss weight')
    parser.add_argument('--target_sparsity', type=float, default=0.1,
                        help='Target sparsity level for z_local')

    # Teacher forcing (prevents error accumulation in autoregressive decoding)
    parser.add_argument('--use_teacher_forcing', type=str2bool, default=False,
                        help='Enable teacher forcing during training')
    # Potential field parameters (optional tuning)
    parser.add_argument('--k_veh_repel', type=float, default=2.0, help='Vehicle repulsion strength')
    parser.add_argument('--sigma_veh', type=float, default=3.0, help='Vehicle repulsion decay rate')
    parser.add_argument('--k_env_boundary', type=float, default=2.0, help='Boundary repulsion strength')
    parser.add_argument('--sigma_env_boundary', type=float, default=2.0, help='Boundary repulsion decay rate')
    parser.add_argument('--k_env_solid', type=float, default=1.5, help='Solid line repulsion strength')
    parser.add_argument('--sigma_env_solid', type=float, default=1.5, help='Solid line repulsion decay rate')
    parser.add_argument('--k_env_dashed', type=float, default=0.3, help='Dashed line repulsion strength (weak)')
    parser.add_argument('--sigma_env_dashed', type=float, default=1.0, help='Dashed line repulsion decay rate')
    parser.add_argument('--use_lane_lines', type=str2bool, default=True,
                        help='Use lane line repulsion (requires 3-channel map)')
    parser.add_argument('--env_loss_ego_only', type=str2bool, default=False,
                        help='Apply env potential loss to ego only (default: all agents)')

    # Redesign: new model architecture params
    parser.add_argument('--num_intents', type=int, default=8,
                        help='Number of intent codebook entries (K)')
    parser.add_argument('--sur_pred_dim', type=int, default=2,
                        help='Predicted surrounding agent delta dimension (dx, dy)')
    parser.add_argument('--map_recrop', type=str2bool, default=False,
                        help='Re-crop map tokens at each decode step')

    # Transformer Decoder params
    parser.add_argument('--trans_num_layers', type=int, default=4,
                        help='Number of Transformer decoder layers')
    parser.add_argument('--trans_d_model', type=int, default=128,
                        help='Transformer decoder d_model dimension')
    parser.add_argument('--trans_nhead', type=int, default=8,
                        help='Number of attention heads in Transformer decoder')
    parser.add_argument('--trans_ffn_dim', type=int, default=512,
                        help='FFN intermediate dimension in Transformer decoder')
    parser.add_argument('--trans_dropout', type=float, default=0.1,
                        help='Dropout rate for Transformer decoder')
    parser.add_argument('--use_ego_z_local', type=str2bool, default=True,
                        help='Enable ego z_local (IntentCodebook)')
    parser.add_argument('--use_sur_z_local', type=str2bool, default=False,
                        help='Enable sur z_local via separate IntentCodebook')

    # Redesign: auxiliary loss weights
    parser.add_argument('--loss_sur_pred', type=float, default=0.1,
                        help='Sur prediction auxiliary loss weight')
    parser.add_argument('--loss_ego_pred', type=float, default=0.1,
                        help='Ego prediction auxiliary loss weight (Phase 1 only)')
    parser.add_argument('--loss_intent_ce', type=float, default=0.0,
                        help='Intent classification CE loss weight (Phase 2 only)')
    parser.add_argument('--intent_range', type=float, default=1.0,
                        help='Intent prototype grid range in std-scaled units (default: 1.0)')
    parser.add_argument('--loss_map_attn', type=float, default=0.0,
                        help='Map attention guidance loss weight')
    parser.add_argument('--map_gt_steps', type=int, default=6,
                        help='Number of future GT steps for map attention soft label')
    parser.add_argument('--map_gt_decay_lambda', type=float, default=0.3,
                        help='Exponential decay lambda for map attention GT weights: w_t = exp(-lambda*t) / sum')
    parser.add_argument('--map_attn_anneal', type=str2bool, default=False,
                        help='Enable cosine annealing for map attn loss (full → 0)')
    parser.add_argument('--map_attn_anneal_epochs', type=int, default=0,
                        help='[LEGACY] Epochs for map attn cosine decay (0 = use total epochs)')
    parser.add_argument('--map_attn_anneal_steps', type=int, default=0,
                        help='Steps for map attn cosine decay (overrides epoch-based if > 0)')

    # V5 Redesign: A2A Relative Bias, z_aux, Action Blending, Enc-Dec Context
    # (use_adaln removed: replaced by LayerNorm in Enc-Dec redesign)
    parser.add_argument('--use_a2a_rel_bias', type=str2bool, default=False,
                        help='Enable A2A relative physical bias (8-feature MLP → attention bias)')
    # z separation: z(Q) × context(KV) → z_context → A2Z in decoder
    parser.add_argument('--num_z_queries', type=int, default=4,
                        help='Number of learnable z_query tokens for cross-attention z generation')
    parser.add_argument('--context_num_layers', type=int, default=2,
                        help='Number of TransformerEncoder layers in context encoder')
    parser.add_argument('--map_summary_tokens', type=int, default=8,
                        help='Number of learnable query tokens for map summary pooling')
    parser.add_argument('--loss_z_aux', type=float, default=0.0,
                        help='z_global auxiliary head loss weight (Phase 1)')
    parser.add_argument('--loss_map_dist_aux', type=float, default=0.0,
                        help='Map distance field auxiliary loss weight (Phase 1)')
    parser.add_argument('--loss_dist_aux', type=float, default=0.0,
                        help='Decoder dist_aux loss weight (Phase 2, last cross_attn → distance)')
    parser.add_argument('--action_blending', type=str2bool, default=False,
                        help='Enable action blending (GT/predicted action interpolation)')
    parser.add_argument('--blend_start_step', type=int, default=50000,
                        help='Global step to start action blending')
    parser.add_argument('--blend_anneal_steps', type=int, default=100000,
                        help='Steps over which blend_alpha ramps to target')
    parser.add_argument('--blend_target_alpha', type=float, default=0.5,
                        help='Target blend_alpha (0=pure GT, 1=pure predicted)')
    parser.add_argument('--blend_alpha_floor', type=float, default=0.0,
                        help='Minimum blend_alpha from step 0 (e.g. 0.05 for 5%% pred from start)')

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict


def _compute_gt_intent_label(model, scene_graph, ego_idx, device, cfg):
    """Compute GT intent soft label for a single ego agent. Mirrors loss logic."""
    from datasets.utils import CARLA_NORM_STATS
    ninfo = CARLA_NORM_STATS[('car', 'truck')]

    num_intents = getattr(cfg, 'num_intents', 9)
    intent_range = getattr(cfg, 'intent_range', 2.0)
    intent_sigma = getattr(cfg, 'intent_sigma', 0.5)

    n_acc = int(np.sqrt(num_intents))
    n_yaw = num_intents // n_acc
    acc_vals = np.linspace(-intent_range, intent_range, n_acc)
    yaw_vals = np.linspace(-intent_range, intent_range, n_yaw)
    acc_grid, yaw_grid = np.meshgrid(acc_vals, yaw_vals, indexing='ij')
    prototypes = np.stack([acc_grid.ravel(), yaw_grid.ravel()], axis=-1)  # (K, 2)

    normalizer = model.get_normalizer()
    s_mean = normalizer.mean_vals[4].item()
    s_std = normalizer.std_vals[4].item()
    hdot_mean = normalizer.mean_vals[5].item()
    hdot_std = normalizer.std_vals[5].item()
    dt = model.dt

    ego_future = scene_graph.future[ego_idx].cpu().numpy()  # (FT, 6) normalized
    ego_past_speed_norm = scene_graph.past[ego_idx, -1, 4].item()
    FT = ego_future.shape[0]

    # Unnormalize speed → compute acc
    ego_speed_raw = ego_future[:, 4] * s_std + s_mean
    prev_speeds = np.concatenate([[ego_past_speed_norm * s_std + s_mean], ego_speed_raw[:-1]])
    raw_acc = (ego_speed_raw - prev_speeds) / dt
    ego_acc = raw_acc / ninfo['a'][1]

    # Unnormalize yaw rate
    raw_yaw_rate = ego_future[:, 5] * hdot_std + hdot_mean
    ego_yaw_rate = raw_yaw_rate / ninfo['hdot'][1]

    gt_xy = np.stack([ego_acc, ego_yaw_rate], axis=-1)  # (FT, 2)
    dist_sq = ((gt_xy[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=-1)  # (FT, K)
    # softmax
    logits = -dist_sq / (2 * intent_sigma ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    gt_soft_label = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return gt_soft_label, gt_xy, prototypes  # (FT, K), (FT, 2), (K, 2)


def _visualize_train_sample(model, scene_graph, map_idx, map_env, device,
                            out_path, global_step, cfg):
    """Generate map attn + intent viz for current train batch (first ego only)."""
    try:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            # AR forward on this batch
            pred = model.reconstruct(scene_graph, map_idx, map_env)
            ego_map_attn = model.get_ego_map_attn_weights()
            intent_weights = model.get_intent_weights()

        if ego_map_attn is None:
            if was_training:
                model.train()
            return

        # Find first ego agent
        ego_mask = scene_graph.sem[:, 0] == 1
        if not ego_mask.any():
            ego_mask = torch.zeros(scene_graph.past.size(0), dtype=torch.bool)
            ego_mask[0] = True
        ego_idx = ego_mask.nonzero(as_tuple=True)[0][0].item()

        state_norm = model.get_normalizer()
        bounds = cfg.map_obs_bounds
        pix = cfg.map_obs_size_pix

        # Unnormalize ego data
        with torch.no_grad():
            ego_pos = state_norm.unnormalize(scene_graph.past[ego_idx, -1:]).squeeze(0).cpu().numpy()
            gt_future = state_norm.unnormalize(scene_graph.future[ego_idx]).cpu().numpy()
            pred_future = state_norm.unnormalize(pred['future_pred'][ego_idx]).cpu().numpy()

        FT_len = gt_future.shape[0]
        mapix = map_idx[scene_graph.batch[ego_idx]].unsqueeze(0)

        # Attn tensor
        if isinstance(ego_map_attn, torch.Tensor):
            attn_tensor = ego_map_attn
        elif isinstance(ego_map_attn, list) and len(ego_map_attn) > 0:
            attn_tensor = torch.cat(ego_map_attn, dim=0)
        else:
            if was_training:
                model.train()
            return

        if attn_tensor.dim() == 3:
            attn_np = attn_tensor[:, 0, :].cpu().numpy()
        elif attn_tensor.dim() == 2:
            attn_np = attn_tensor.cpu().numpy()
        else:
            if was_training:
                model.train()
            return

        num_tokens = attn_np.shape[1]
        gs = int(np.sqrt(num_tokens))
        _rf_s = getattr(model, 'map_rf_stride', None)
        _rf_o = getattr(model, 'map_rf_offset', None)

        # Per-step map crops
        show_steps = [s for s in [0, 3, 6, 9, 11] if s < FT_len]
        map_obs_dict = {}

        ego_frames_world = np.zeros((FT_len + 1, 4))
        ego_frames_world[0] = ego_pos[:4]
        ego_frames_world[1:] = gt_future[:, :4]

        pred_frames_world = np.zeros((FT_len + 1, 4))
        pred_frames_world[0] = ego_pos[:4]
        pred_frames_world[1:] = pred_future[:, :4]

        with torch.no_grad():
            # t=0 map crop
            ego_t0 = torch.tensor(ego_pos[:4], device=device, dtype=torch.float32).unsqueeze(0)
            map_obs_t0 = map_env.get_map_crop_pos(ego_t0, mapix).cpu().numpy()[0]
            map_obs_dict['traj'] = map_obs_t0
            map_obs_dict[0] = map_obs_t0

            for t in show_steps:
                if t == 0:
                    continue
                frame_pos = gt_future[t - 1, :4]
                frame_tensor = torch.tensor(frame_pos, dtype=torch.float32, device=device).unsqueeze(0)
                map_obs_dict[t] = map_env.get_map_crop_pos(frame_tensor, mapix).cpu().numpy()[0]

        # GT soft labels
        soft_labels = compute_soft_label_grid(
            gt_future, ego_frames_world, grid_size=gs,
            bounds=bounds, pix_size=pix,
            sigma_d=getattr(cfg, 'map_gt_sigma_d', 1.0),
            decay_lambda=getattr(cfg, 'map_gt_decay_lambda', 0.075),
            rf_stride=_rf_s, rf_offset=_rf_o
        )

        # Remap attn from pred frame to GT frame
        attn_remapped = attn_np.copy()
        pred_pix_per_step = {}
        for t in range(FT_len):
            gt_frame = ego_frames_world[t]
            pred_frame = pred_frames_world[t]
            attn_grid = attn_np[t].reshape(gs, gs)
            remapped = remap_attn_to_gt_frame(
                attn_grid, pred_frame, gt_frame, gs, bounds,
                pix_size=pix, rf_stride=_rf_s, rf_offset=_rf_o)
            attn_remapped[t] = remapped.reshape(-1)

            if t in show_steps:
                dx = pred_future[t, 0] - gt_frame[0]
                dy = pred_future[t, 1] - gt_frame[1]
                local_l = dx * gt_frame[2] + dy * gt_frame[3]
                local_w = -dx * gt_frame[3] + dy * gt_frame[2]
                pix_x = (local_l - bounds[0]) / (bounds[2] - bounds[0]) * pix
                pix_y = (local_w - bounds[1]) / (bounds[3] - bounds[1]) * pix
                pred_pix_per_step[t] = (pix_x, pix_y)

        gt_traj_pix = world_to_crop_pixel(gt_future[:, :2], ego_pos, bounds, pix, pix)
        pred_traj_pix = world_to_crop_pixel(pred_future[:, :2], ego_pos, bounds, pix, pix)

        # Save
        viz_dir = os.path.join(out_path, 'viz_train_steps')
        map_attn_dir = os.path.join(viz_dir, 'map_attention')
        intent_dir = os.path.join(viz_dir, 'intent')
        os.makedirs(map_attn_dir, exist_ok=True)
        os.makedirs(intent_dir, exist_ok=True)

        map_attn_path = os.path.join(map_attn_dir, f'step{global_step:07d}_map_attn.png')
        visualize_map_attention(
            map_obs_dict, attn_remapped, map_attn_path,
            agent_idx=ego_idx,
            soft_labels=soft_labels,
            pred_pix_per_step=pred_pix_per_step,
            gt_traj_pix=gt_traj_pix,
            pred_traj_pix=pred_traj_pix,
            rf_stride=_rf_s, rf_offset=_rf_o,
            pix_size=pix, bounds=bounds,
        )

        # Intent viz (with GT intent soft label)
        if intent_weights is not None:
            if isinstance(intent_weights, torch.Tensor):
                iw = intent_weights
            elif isinstance(intent_weights, list) and len(intent_weights) > 0:
                iw = torch.stack(intent_weights, dim=0)
            else:
                iw = None

            if iw is not None:
                if iw.dim() == 3:
                    iw_np = iw[:, 0, :].cpu().numpy()
                elif iw.dim() == 2:
                    iw_np = iw.cpu().numpy()
                else:
                    iw_np = None

                if iw_np is not None:
                    # Compute GT intent soft label (same logic as loss)
                    gt_intent_np, gt_xy_np, proto_np = _compute_gt_intent_label(
                        model, scene_graph, ego_idx, device, cfg)

                    intent_path = os.path.join(intent_dir, f'step{global_step:07d}_intent.png')
                    visualize_intent_codebook(iw_np, intent_path, agent_idx=ego_idx,
                                              num_intents=getattr(cfg, 'num_intents', 9),
                                              gt_intent_label=gt_intent_np,
                                              map_obs=map_obs_dict.get('traj'),
                                              gt_traj_pix=gt_traj_pix,
                                              pred_traj_pix=pred_traj_pix)

                    # Intent grid scatter (GT positions on prototype grid)
                    intent_grid_dir = os.path.join(viz_dir, 'intent_grid')
                    os.makedirs(intent_grid_dir, exist_ok=True)
                    grid_path = os.path.join(intent_grid_dir, f'step{global_step:07d}_intent_grid.png')
                    visualize_intent_grid(gt_xy_np, proto_np, iw_np, grid_path,
                                          agent_idx=ego_idx,
                                          intent_range=getattr(cfg, 'intent_range', 2.0),
                                          map_obs=map_obs_dict.get('traj'),
                                          gt_traj_pix=gt_traj_pix,
                                          pred_traj_pix=pred_traj_pix)

        if was_training:
            model.train()

    except Exception as e:
        import traceback
        print(f'[Viz] Error at step {global_step}: {e}')
        traceback.print_exc()
        if model.training != was_training:
            model.train() if was_training else model.eval()


def run_one_epoch(data_loader, model, map_env, loss_fn, device, out_path,
                  train=True,
                  optimizer=None,
                  step_counter=0,
                  use_wandb=False,
                  use_teacher_forcing=False,
                  current_epoch=0,
                  tb_writer=None,
                  global_step=0,
                  scheduler=None,
                  blend_alpha=0.0,
                  cfg=None):
    '''
    Run through dataset and for a single epoch. Trains if desired.
    '''
    if use_wandb:
        import wandb
    if train and optimizer is None:
        throw_err('Must give optimizer to train!')
    prefix = "Train" if train else "Eval"

    pbar = tqdm.tqdm(data_loader)

    if train:
        assert optimizer is not None

    # Vector of all losses and IoU values for the batch
    metrics = {}

    empty_cache = False
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        pred = loss_dict = None
        scene_name_debug = scene_graph.scene_name if hasattr(scene_graph, 'scene_name') else ['Unknown']

        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()
        try:
            # Step-based annealing: compute blend_alpha, map_attn_w, kl_weight per batch
            cur_blend_alpha = 0.0
            if train and cfg is not None:
                # KL annealing: step-based linear ramp with floor
                kl_anneal_steps = getattr(cfg, 'kl_anneal_steps', 0)
                if kl_anneal_steps > 0:
                    cur_kl = compute_kl_weight(global_step, kl_anneal_steps, cfg.loss_kl,
                                               kl_beta_floor=getattr(cfg, 'kl_floor', 0.0))
                    loss_fn.loss_weights['kl'] = cur_kl

                # Action blending: step-based alpha ramp
                if getattr(cfg, 'action_blending', False):
                    blend_floor = getattr(cfg, 'blend_alpha_floor', 0.0)
                    progress = min(global_step / max(cfg.blend_anneal_steps, 1), 1.0)
                    cur_blend_alpha = blend_floor + (cfg.blend_target_alpha - blend_floor) * progress

                # Map attn annealing: step-based cosine decay (overrides epoch-based)
                if getattr(cfg, 'map_attn_anneal', False) and cfg.loss_map_attn > 0:
                    anneal_steps = getattr(cfg, 'map_attn_anneal_steps', 0)
                    if anneal_steps > 0:
                        progress = min(global_step / max(anneal_steps, 1), 1.0)
                        map_attn_w = cfg.loss_map_attn * 0.5 * (1 + math.cos(math.pi * progress))
                        loss_fn.loss_weights['map_attn'] = map_attn_w

            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            B = map_idx.size(0)

            do_sample = loss_fn.loss_weights.get('coll_veh_prior', 0.0) > 0.0 or \
                        loss_fn.loss_weights.get('coll_env_prior', 0.0) > 0.0

            pred = model(scene_graph, map_idx, map_env, future_sample=do_sample,
                         teacher_forcing=use_teacher_forcing if train else False,
                         current_epoch=current_epoch,
                         blend_alpha=cur_blend_alpha if train else 0.0)

            # Pass model to loss_fn for z_local sparsity computation
            # use_teacher_forcing only during training
            loss_dict = loss_fn(scene_graph, pred,
                                map_idx=map_idx,
                                map_env=map_env,
                                model=model,
                                use_teacher_forcing=use_teacher_forcing if train else False)

            loss = loss_dict['loss'][0]

            # --- START: ROBUST NAN DEBUGGING BLOCK ---
            if not train:
                for loss_name, loss_value in loss_dict.items():
                    if loss_value is not None and torch.isnan(loss_value).any():
                        print(f"\n!!! NaN DETECTED !!!")
                        print(f"Scene causing error: {scene_name_debug}")
                        print(f"Error occurred in validation batch index: {i}")
                        print(f"The loss component '{loss_name}' became NaN.")
                        raise RuntimeError(f"NaN detected in {loss_name}")

                # --- Start: Detailed Loss Monitoring ---
                parts = [f"Total Loss: {loss.item():.4f}"]
                for k, v in loss_dict.items():
                    if k == 'loss':
                        continue
                    parts.append(f"{k}: {torch.mean(v).item():.4f}")
                print(" | ".join(parts))
            # --- END: ROBUST NAN DEBUGGING BLOCK ---

            if train:
                # training step for generator
                optimizer.zero_grad()
                loss.backward()

                # Gradient norm logging (every 50 steps)
                if global_step % 50 == 0 and tb_writer is not None:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    tb_writer.add_scalar('grad_norm/total', total_norm, global_step)
                    # Per-module gradient norms
                    per_mod = model.compute_per_module_grad_norms()
                    for mk, mv in per_mod.items():
                        tb_writer.add_scalar(f'grad_norm/{mk}', mv, global_step)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # Step-based LR scheduler (warmup + cosine decay)
                if scheduler is not None:
                    scheduler.step()

            # compute interpretable errors
            err_dict = loss_fn.compute_err(scene_graph, pred,
                                            model.get_normalizer())
        except RuntimeError as e:
            import traceback
            Logger.log('Caught error in training batch %s!' % (str(e)))
            Logger.log('Traceback:')
            traceback.print_exc()
            Logger.log('Skipping')
            if pred is not None:
                del pred
            if loss_dict is not None:
                del loss_dict
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            empty_cache = True
            continue

        # metrics to save
        batch_metrics = {**loss_dict, **err_dict}

        if use_wandb:
            # per-batch wandb metrics
            wandb_batch_metrics = {}
            if train:
                step_counter += B
                for k, v in batch_metrics.items():
                    if v is not None:
                        wandb_batch_metrics[prefix + " Batch Mean " + k] = torch.mean(v).item()

            if len(wandb_batch_metrics) > 0:
                wandb.log(wandb_batch_metrics, step=step_counter)

        # Metrics to log to the tqdm progress bar
        progress_bar_metrics = {}
        # Keep track of metrics over whole epoch
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(c2c(v))
            progress_bar_metrics[k] = torch.mean(v).item()
        # Log the loss to the tqdm progress bar
        pbar.set_postfix(progress_bar_metrics)

        # TensorBoard: per-batch logging (train only, every 10 batches)
        if tb_writer is not None and train:
            global_step += 1
            if global_step % 10 == 0:
                for k, v in progress_bar_metrics.items():
                    tb_writer.add_scalar(f'batch/{k}', v, global_step)
                if scheduler is not None:
                    tb_writer.add_scalar('batch/lr', scheduler.get_last_lr()[0], global_step)
                # Log annealing values
                if cur_blend_alpha > 0:
                    tb_writer.add_scalar('anneal/blend_alpha', cur_blend_alpha, global_step)
                if cfg is not None and getattr(cfg, 'kl_anneal_steps', 0) > 0:
                    tb_writer.add_scalar('anneal/kl_weight',
                                         loss_fn.loss_weights.get('kl', 0), global_step)
                if cfg is not None and getattr(cfg, 'map_attn_anneal', False):
                    tb_writer.add_scalar('anneal/map_attn_weight',
                                         loss_fn.loss_weights.get('map_attn', 0), global_step)
            # Health metrics: log every 500 steps for finer tracking
            if global_step % 500 == 0:
                health = model.compute_health_metrics()
                for hk, hv in health.items():
                    tb_writer.add_scalar(f'health_step/{hk}', hv, global_step)

        # Visualization: every 200 steps during training (skip Phase 1 — no decoder)
        viz_every = getattr(cfg, 'viz_every_steps', 200) if cfg is not None else 200
        current_phase = getattr(cfg, 'phase', 2)
        if train and global_step > 0 and global_step % viz_every == 0 and current_phase != 1:
            _visualize_train_sample(model, scene_graph, map_idx, map_env,
                                    device, out_path, global_step, cfg)

    wandb_epoch_metrics = {}
    epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = np.concatenate(metrics[k])
        epoch_metrics[k] = np.mean(metrics[k])
        wandb_epoch_metrics[prefix + " Epoch Mean " + k] = epoch_metrics[k]

    mean_epoch_loss = wandb_epoch_metrics[prefix + " Epoch Mean loss"]

    if use_wandb and len(wandb_epoch_metrics) > 0:
        wandb.log(wandb_epoch_metrics, step=step_counter)

    return step_counter, mean_epoch_loss, epoch_metrics, global_step


def main():
    cfg, cfg_dict = parse_cfg()

    print(f'=== TrafficPlanner Training ===')
    print(f'past_len: {cfg.past_len}')
    print(f'future_len: {cfg.future_len}')
    print(f'z_local_size: {cfg.z_local_size}')
    print(f'use_potential_loss: {cfg.use_potential_loss}')
    print(f'  - use_veh_potential: {cfg.use_veh_potential}')
    print(f'use_sparse_loss: {cfg.use_sparse_loss}')
    print(f'use_teacher_forcing: {cfg.use_teacher_forcing}')

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        # wandb setup
        wandb.init(project=cfg.wandb_project, config=cfg_dict,
                    mode='offline' if cfg.wandb_offline else 'online',
                    name=cfg.wandb_name)

    # create output directory and logging
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'train_log.txt')
    Logger.init(log_path)
    # save arguments used
    Logger.log('Args: ' + str(cfg_dict))

    # device setup
    device = f'cuda:{cfg.gpu}'
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    # first create map environment
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')

    map_env = FITMapEnv(data_path,
                            bounds=cfg.map_obs_bounds,
                            L=cfg.map_obs_size_pix,
                            W=cfg.map_obs_size_pix,
                            layers=cfg.map_layers,
                            device=device,
                            )

    # create nuscenes object out here and pass into dataset to save memory
    Logger.log('Creating dataset...')

    seq_interval = getattr(cfg, 'seq_interval', 1)
    train_dataset = FITDataset(data_path, map_env,
                            split='train',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            dt = cfg.dt,
                            noise_std=cfg.data_noise_std,
                            use_challenge_splits=cfg.use_challenge_splits,
                            reduce_cats=cfg.reduce_cats,
                            seq_interval=seq_interval
                            )
    val_dataset = FITDataset(data_path, map_env,
                            split='val',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            dt = cfg.dt,
                            use_challenge_splits=cfg.use_challenge_splits,
                            reduce_cats=cfg.reduce_cats,
                            seq_interval=seq_interval
                            )

    # create loaders
    train_loader = GraphDataLoader(train_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug
    val_loader = GraphDataLoader(val_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=False,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    #
    # Initialize TrafficPlannerModel
    #
    Logger.log('Initializing TrafficPlanner Model...')
    model = TrafficPlannerModel(
        cfg.past_len,
        cfg.future_len,
        cfg.map_obs_size_pix,
        len(train_dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,          # z_global size (32)
        z_local_size=cfg.z_local_size,         # intent_dim (32)
        output_bicycle=cfg.model_output_bicycle,
        dt=cfg.dt,
        # Map Conv parameters (from base config)
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
        # Redesign params
        num_intents=cfg.num_intents,
        sur_pred_dim=cfg.sur_pred_dim,
        map_recrop=cfg.map_recrop,
        # Transformer decoder params
        trans_num_layers=cfg.trans_num_layers,
        trans_d_model=cfg.trans_d_model,
        trans_nhead=cfg.trans_nhead,
        trans_ffn_dim=cfg.trans_ffn_dim,
        trans_dropout=cfg.trans_dropout,
        use_ego_z_local=cfg.use_ego_z_local,
        use_sur_z_local=cfg.use_sur_z_local,
        # V5/V6 redesign (Enc-Dec Cross-Attention)
        use_a2a_rel_bias=cfg.use_a2a_rel_bias,
        num_z_queries=getattr(cfg, 'num_z_queries', 4),
        enc_dropout=getattr(cfg, 'enc_dropout', 0.1),
        context_num_layers=getattr(cfg, 'context_num_layers', 2),
        map_summary_tokens=getattr(cfg, 'map_summary_tokens', 8),
    ).to(device)

    train_loss = []
    valid_loss = []

    #
    # Setup loss weights
    #
    loss_weights = {
        'recon': cfg.loss_recon,
        'kl': cfg.loss_kl,
        'coll_veh_prior': cfg.loss_veh_coll_prior,
        'coll_env_prior': cfg.loss_env_coll_prior,
        # TrafficPlanner specific - potential-based loss
        'potential_veh': cfg.loss_potential_veh if cfg.use_potential_loss else 0.0,
        'potential_env': cfg.loss_potential_env if cfg.use_potential_loss else 0.0,
        # z_local sparsity loss (off by default)
        'sparse': cfg.loss_sparse if cfg.use_sparse_loss else 0.0,
        # Auxiliary losses (Redesign)
        'sur_pred': cfg.loss_sur_pred,
        'ego_pred': cfg.loss_ego_pred,
        'intent_ce': cfg.loss_intent_ce,
        'map_attn': cfg.loss_map_attn,
        'z_aux': cfg.loss_z_aux,
        'map_dist_aux': getattr(cfg, 'loss_map_dist_aux', 0.0),
        'dist_aux': getattr(cfg, 'loss_dist_aux', 0.0),
    }

    # Potential field configuration
    potential_cfg = {
        'k_veh_repel': cfg.k_veh_repel,
        'sigma_veh': cfg.sigma_veh,
        'min_dist_veh': 1.0,
        'k_env_boundary': cfg.k_env_boundary,
        'sigma_env_boundary': cfg.sigma_env_boundary,
        'k_env_solid': cfg.k_env_solid,
        'sigma_env_solid': cfg.sigma_env_solid,
        'k_env_dashed': cfg.k_env_dashed,
        'sigma_env_dashed': cfg.sigma_env_dashed,
        'use_lane_lines': cfg.use_lane_lines,
        'env_loss_ego_only': cfg.env_loss_ego_only,
    }


    # Sparsity loss configuration
    sparsity_cfg = {
        'target_sparsity': cfg.target_sparsity,
        'kl_weight': 0.1,
    }

    #
    # Initialize TrafficPlannerLoss
    #
    aux_cfg = {
        'map_gt_steps': getattr(cfg, 'map_gt_steps', 6),
        'map_gt_decay_lambda': getattr(cfg, 'map_gt_decay_lambda', 0.3),
        'intent_sigma': getattr(cfg, 'intent_sigma', 0.5),
        'intent_range': getattr(cfg, 'intent_range', 1.0),
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights,
        train_dataset.get_state_normalizer(),
        train_dataset.get_att_normalizer(),
        phase=cfg.phase,
        use_sparse_loss=cfg.use_sparse_loss,
        use_potential_loss=cfg.use_potential_loss,
        use_veh_potential=cfg.use_veh_potential,
        potential_cfg=potential_cfg,
        sparsity_cfg=sparsity_cfg,
        aux_cfg=aux_cfg,
        recon_pos_weight=getattr(cfg, 'recon_pos_weight', 1.0),
        kl_free_bits=getattr(cfg, 'kl_free_bits', 0.0),
    ).to(device)

    Logger.log('Num model params: %d' % (count_params(model)))

    # Phase 1: freeze decoder, train encoder only
    if cfg.phase == 1:
        model.phase = 1
        model.freeze_for_phase1()

    # create optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params,
                           lr=cfg.lr,
                           betas=(0.9, 0.95),
                           weight_decay=cfg.weight_decay)
    Logger.log('Optimizer: %d trainable parameter groups' % len(trainable_params))

    # load model weights & optimizer to start from, if given
    ckpt_epoch = 0
    ckpt_eval_loss = float('inf')
    global_step = 0  # batch-level step counter (for step-based LR scheduler)

    # Phase 2: Load Phase 1 checkpoint
    if cfg.phase == 2 and cfg.phase1_ckpt is not None:
        Logger.log('Loading Phase 1 checkpoint for Phase 2 training...')
        ckpt_epoch, ckpt_eval_loss, _ = load_state(cfg.phase1_ckpt, model,
                                                    optimizer=None,  # Don't load optimizer for Phase 2
                                                    map_location=device)
        Logger.log('Loaded Phase 1 checkpoint from epoch %d' % (ckpt_epoch))
        ckpt_epoch = 0  # Reset epoch counter for Phase 2
        ckpt_eval_loss = float('inf')  # Reset eval loss tracking
        global_step = 0  # Reset step counter for Phase 2

        # Set Phase 2: reinit decoder, freeze encoder
        model.phase = 2
        model.reinit_decoder()
        model.freeze_z_global()
        loss_fn.set_phase(2)
        Logger.log('Phase 2: decoder reinitialized, encoder frozen, loss phase=2')

        # Recreate optimizer with only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params,
                               lr=cfg.lr,
                               betas=(0.9, 0.95),
                               weight_decay=cfg.weight_decay)
        Logger.log('Created optimizer with %d trainable parameters' % len(trainable_params))
    elif cfg.ckpt is not None:
        ckpt_epoch, ckpt_eval_loss, global_step = load_state(cfg.ckpt, model,
                                                              optimizer=optimizer,
                                                              map_location=device)
        Logger.log('Loaded checkpoint from epoch %d, global_step %d, validation loss %f...' % (ckpt_epoch, global_step, ckpt_eval_loss))

    # LR scheduler (step-based cosine annealing with linear warmup)
    scheduler = None
    lr_schedule_fn = None
    if cfg.use_lr_anneal:
        lr_max = cfg.lr_max if cfg.lr_max is not None else cfg.lr
        lr_min = cfg.lr_min
        warmup_steps = cfg.lr_warmup_steps

        # Estimate total steps if not provided
        if cfg.lr_total_steps is not None:
            total_steps = cfg.lr_total_steps
        else:
            est_steps_per_epoch = len(train_dataset) // cfg.batch_size
            total_steps = est_steps_per_epoch * cfg.epochs
            Logger.log(f'LR: estimated {est_steps_per_epoch} steps/epoch, {total_steps} total steps')

        # Step-based warmup + cosine decay function
        # ~0 → lr_max (warmup) → lr_min (cosine decay)
        warmup_floor = 5e-7 / lr_max  # near-zero start for warmup
        decay_floor = lr_min / lr_max  # cosine decay end
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup: ~0 → lr_max
                return max(warmup_floor, step / max(warmup_steps, 1))
            else:
                # Cosine decay: lr_max → lr_min
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                progress = min(progress, 1.0)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(decay_floor, cosine_decay)

        # Set optimizer to lr_max (lambda will scale it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_max
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        lr_schedule_fn = lr_lambda  # for logging

        # If resuming, advance scheduler to correct step
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()

        Logger.log(f'LR Step-based: warmup={warmup_steps} steps, '
                   f'max={lr_max}, min={lr_min}, total={total_steps} steps')

    # Freeze z_global if requested (for custom training scenarios)
    if cfg.freeze_z_global:
        model.freeze_z_global()
        Logger.log('Frozen z_global encoder (manual override)')

    # so can unnormalize as needed
    model.set_normalizer(train_dataset.get_state_normalizer())
    model.set_att_normalizer(train_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        # from datasets.utils import NUSC_BIKE_PARAMS
        # model.set_bicycle_params(NUSC_BIKE_PARAMS)
        from datasets.utils import CARLA_BIKE_PARAMS
        model.set_bicycle_params(CARLA_BIKE_PARAMS)

    #
    # Print ACTUAL applied configuration from initialized objects
    #
    Logger.log('=' * 80)
    Logger.log('APPLIED CONFIGURATION (from initialized objects)')
    Logger.log('=' * 80)

    # Model parameters
    Logger.log('\n[Model Architecture]')
    Logger.log(f'  PT (past_len): {model.PT}')
    Logger.log(f'  FT (future_len): {model.FT}')
    Logger.log(f'  z_local_size: {model.z_local_size}')
    Logger.log(f'  num_intents: {model.num_intents}')
    Logger.log(f'  intent_dim: {model.intent_dim}')
    Logger.log(f'  map_num_tokens: {model.map_num_tokens}')
    Logger.log(f'  sur_pred_dim: {model.sur_pred_dim}')
    Logger.log(f'  map_recrop: {model.map_recrop}')
    Logger.log(f'  use_a2a_rel_bias: {model.use_a2a_rel_bias}')
    Logger.log(f'  context_encoder: {model.context_encoder}')
    Logger.log(f'  map_summary_pooling: {model.map_summary_pooling.num_queries} tokens')
    Logger.log(f'  num_z_queries: {model.num_z_queries}')

    # Action blending
    Logger.log('\n[Action Blending]')
    Logger.log(f'  action_blending: {cfg.action_blending}')
    if cfg.action_blending:
        Logger.log(f'  blend_start_step: {cfg.blend_start_step}')
        Logger.log(f'  blend_anneal_steps: {cfg.blend_anneal_steps}')
        Logger.log(f'  blend_target_alpha: {cfg.blend_target_alpha}')
        Logger.log(f'  blend_alpha_floor: {getattr(cfg, "blend_alpha_floor", 0.0)}')

    # Loss function parameters
    Logger.log('\n[Loss Function]')
    Logger.log(f'  use_sparse_loss: {loss_fn.use_sparse_loss}')
    Logger.log(f'  use_potential_loss: {loss_fn.use_potential_loss}')
    Logger.log(f'  use_veh_potential: {loss_fn.use_veh_potential}')

    # Loss weights
    Logger.log('\n[Loss Weights]')
    for key, value in loss_fn.loss_weights.items():
        Logger.log(f'  {key}: {value}')

    # Potential field configuration
    Logger.log('\n[Potential Field Config]')
    Logger.log(f'  k_veh_repel: {loss_fn.potential_cfg["k_veh_repel"]}')
    Logger.log(f'  sigma_veh: {loss_fn.potential_cfg["sigma_veh"]}')
    Logger.log(f'  k_env_boundary: {loss_fn.potential_cfg["k_env_boundary"]}')
    Logger.log(f'  sigma_env_boundary: {loss_fn.potential_cfg["sigma_env_boundary"]}')
    Logger.log(f'  k_env_solid: {loss_fn.potential_cfg["k_env_solid"]}')
    Logger.log(f'  sigma_env_solid: {loss_fn.potential_cfg["sigma_env_solid"]}')
    Logger.log(f'  k_env_dashed: {loss_fn.potential_cfg["k_env_dashed"]}')
    Logger.log(f'  sigma_env_dashed: {loss_fn.potential_cfg["sigma_env_dashed"]}')
    Logger.log(f'  use_lane_lines: {loss_fn.potential_cfg["use_lane_lines"]}')
    Logger.log(f'  env_loss_ego_only: {loss_fn.potential_cfg["env_loss_ego_only"]}')

    # Teacher forcing (from cfg, not stored in model/loss_fn)
    Logger.log('\n[Teacher Forcing]')
    Logger.log(f'  use_teacher_forcing: {cfg.use_teacher_forcing}')

    # Optimizer
    Logger.log('\n[Optimizer]')
    actual_lr = optimizer.param_groups[0]['lr']
    Logger.log(f'  lr (actual): {actual_lr}')
    Logger.log(f'  weight_decay: {cfg.weight_decay}')
    if cfg.use_lr_anneal:
        Logger.log(f'  step-based cosine: warmup={cfg.lr_warmup_steps} steps, '
                   f'{cfg.lr_max} → {cfg.lr_min}')

    Logger.log('=' * 80)
    Logger.log('')

    # KL loss annealing (step-based, computed per batch in run_one_epoch)
    use_kl_anneal = getattr(cfg, 'kl_anneal_steps', 0) > 0
    if use_kl_anneal:
        Logger.log(f'Using step-based KL annealing: {cfg.kl_anneal_steps} steps, '
                   f'floor={cfg.kl_floor}, target={cfg.loss_kl}')

    # run training
    ckpts_path = os.path.join(cfg.out, 'checkpoints_trafficplanner')
    mkdir(ckpts_path)
    step_counter = 0
    min_eval_loss = ckpt_eval_loss

    # TensorBoard
    tb_log_dir = os.path.join(cfg.out, 'tb_logs')
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    Logger.log(f'TensorBoard logs: {tb_log_dir}')

    # Auto-start TensorBoard server (port 6006, skip if already running)
    tb_proc = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        port_in_use = s.connect_ex(('localhost', 6006)) == 0
        s.close()
        if not port_in_use:
            tb_proc = subprocess.Popen(
                ['tensorboard', '--logdir', cfg.out, '--host', '0.0.0.0', '--port', '6006'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            Logger.log(f'TensorBoard server started on http://0.0.0.0:6006 (PID: {tb_proc.pid})')
            time.sleep(2)
            webbrowser.open('http://localhost:6006')
        else:
            Logger.log('TensorBoard server already running on port 6006')
            webbrowser.open('http://localhost:6006')
    except Exception as e:
        Logger.log(f'TensorBoard auto-start failed: {e}')

    # CSV logging
    csv_dir = os.path.join(cfg.out, 'csv_logs')
    mkdir(csv_dir)
    csv_train_path = os.path.join(csv_dir, 'train_losses.csv')
    csv_val_path = os.path.join(csv_dir, 'val_losses.csv')
    csv_train_file = open(csv_train_path, 'w', newline='')
    csv_val_file = open(csv_val_path, 'w', newline='')
    csv_train_writer = None  # initialized on first epoch (to get column names)
    csv_val_writer = None
    Logger.log(f'CSV logs: {csv_dir}')

    # Matplotlib for loss visualization
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # global_step is initialized above (from checkpoint or 0)

    for epoch in range(ckpt_epoch, cfg.epochs):
        Logger.log('Starting epoch %d...' % epoch)

        # KL annealing: step-based (computed per batch in run_one_epoch)
        # Just log current KL weight at epoch start
        if use_kl_anneal:
            cur_beta = compute_kl_weight(global_step, cfg.kl_anneal_steps, cfg.loss_kl,
                                          kl_beta_floor=cfg.kl_floor)
            Logger.log(f'KL weight (step {global_step}): {cur_beta:.6f}')

        # Map attention loss annealing: epoch-based (legacy) or step-based
        # Step-based overrides epoch-based if map_attn_anneal_steps > 0
        if getattr(cfg, 'map_attn_anneal', False) and cfg.loss_map_attn > 0:
            anneal_steps = getattr(cfg, 'map_attn_anneal_steps', 0)
            if anneal_steps > 0:
                # Step-based: computed inside run_one_epoch per batch
                Logger.log('Map attn: step-based annealing (computed per batch)')
            else:
                # Legacy epoch-based
                anneal_total = cfg.map_attn_anneal_epochs if cfg.map_attn_anneal_epochs > 0 else cfg.epochs
                progress = min(epoch / max(anneal_total, 1), 1.0)
                map_attn_w = cfg.loss_map_attn * 0.5 * (1 + math.cos(math.pi * progress))
                loss_fn.loss_weights['map_attn'] = map_attn_w
                Logger.log('Map attn weight %.6f...' % map_attn_w)
            if tb_writer is not None:
                tb_writer.add_scalar('train/map_attn_weight', loss_fn.loss_weights.get('map_attn', 0), epoch)

        # train for one epoch
        start_t = time.time()
        model.train()
        step_counter, mean_train_loss, train_epoch_metrics, global_step = run_one_epoch(
                                        train_loader, model, map_env, loss_fn, device, cfg.out,
                                        train=True,
                                        optimizer=optimizer,
                                        step_counter=step_counter,
                                        use_wandb=use_wandb,
                                        use_teacher_forcing=cfg.use_teacher_forcing,
                                        current_epoch=epoch,
                                        tb_writer=tb_writer,
                                        global_step=global_step,
                                        scheduler=scheduler,
                                        cfg=cfg)
        train_loss.append(mean_train_loss)

        # TensorBoard: log train metrics
        if tb_writer is not None:
            for k, v in train_epoch_metrics.items():
                tb_writer.add_scalar(f'train/{k}', v, epoch)
            if scheduler is not None:
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
            # Health monitoring metrics
            health = model.compute_health_metrics()
            for k, v in health.items():
                tb_writer.add_scalar(f'health/{k}', v, epoch)

        # CSV: log train metrics
        if csv_train_writer is None:
            csv_train_writer = csv.DictWriter(csv_train_file, fieldnames=['epoch'] + sorted(train_epoch_metrics.keys()))
            csv_train_writer.writeheader()
        row = {'epoch': epoch}
        row.update(train_epoch_metrics)
        csv_train_writer.writerow(row)
        csv_train_file.flush()
        ax1.clear()
        ax1.plot(train_loss)
        ax1.set_title("Train Loss")
        plt.savefig(f'loss_{cfg.loss_plot_suffix}.jpg', format='jpeg')

        # lot of excess memory used by pygeometric
        torch.cuda.empty_cache()
        Logger.log('Epoch time: %f' % (time.time() - start_t))

        # validate if desired
        if epoch % cfg.val_every == 0:
            Logger.log('Validating...')
            with torch.no_grad():
                model.eval()
                step_counter, mean_eval_loss, val_epoch_metrics, _ = run_one_epoch(
                                                            val_loader, model, map_env, loss_fn, device, cfg.out,
                                                            train=False,
                                                            step_counter=step_counter,
                                                            use_wandb=use_wandb,
                                                            use_teacher_forcing=False,
                                                            current_epoch=epoch)

                # TensorBoard: log val metrics
                if tb_writer is not None:
                    for k, v in val_epoch_metrics.items():
                        tb_writer.add_scalar(f'val/{k}', v, epoch)

                # CSV: log val metrics
                if csv_val_writer is None:
                    csv_val_writer = csv.DictWriter(csv_val_file, fieldnames=['epoch'] + sorted(val_epoch_metrics.keys()))
                    csv_val_writer.writeheader()
                val_row = {'epoch': epoch}
                val_row.update(val_epoch_metrics)
                csv_val_writer.writerow(val_row)
                csv_val_file.flush()
                valid_loss.append(mean_eval_loss)
                print(f'min_eval_loss = ', min_eval_loss)
                print(f'mean_eval_loss = ', mean_eval_loss)
                ax2.clear()
                ax2.plot(valid_loss)
                ax2.set_title("Validation Loss")
                plt.savefig(f'loss_{cfg.loss_plot_suffix}.jpg', format='jpeg')

                if mean_eval_loss < min_eval_loss:
                    Logger.log('Lowest eval loss so far! Saving checkpoint...')
                    min_eval_loss = mean_eval_loss
                    save_file = os.path.join(ckpts_path, 'best_eval_model.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=global_step)
                    if use_wandb:
                        wandb.save(save_file)
                torch.cuda.empty_cache()

        # save checkpoint if desired
        if epoch % cfg.save_every == 0:
            Logger.log('Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=global_step)
            save_file = os.path.join(ckpts_path, 'latest_model.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=global_step)
            if use_wandb:
                wandb.save(save_file)

    # Close loggers
    if tb_writer is not None:
        tb_writer.close()
    csv_train_file.close()
    csv_val_file.close()
    if tb_proc is not None:
        tb_proc.terminate()
        Logger.log('TensorBoard server stopped.')
    Logger.log(f'Training complete. TensorBoard: tensorboard --logdir {tb_log_dir}')

    if use_wandb:
        # save full log after training
        wandb.save(log_path)

if __name__ == "__main__":
    main()
