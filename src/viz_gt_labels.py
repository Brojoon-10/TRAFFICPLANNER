#!/usr/bin/env python3
"""
GT Soft Label Visualization — Intent + Map Attention

Visualizes GT-derived soft labels that the model is trained to match:
1. Intent soft label: 3x3 heatmap from (acc, yaw_rate) → 9 prototypes
2. Map attn soft label: NxN heatmap from GT future trajectory (N depends on conv_stride_list)
   - With and without potential weighting (drivable/solid/dashed)

No model checkpoint needed — purely data-driven GT labels.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets.nuscenes_utils import normalize_scene_graph
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from datasets.utils import NUSC_NORM_STATS
from utils.common import dict2obj, mkdir
from utils.config import get_parser, add_base_args
from utils.torch import get_device
from utils.transforms import transform2frame
from utils.torch import calc_conv_out


def compute_map_token_spatial(conv_kernel_list, conv_stride_list, map_obs_size_pix=256):
    """Calculate map token grid size from CNN config (early conv layers 0-2)."""
    s = map_obs_size_pix
    for i in range(3):
        s = calc_conv_out(s, conv_kernel_list[i], conv_stride_list[i])
    return s


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_cfg():
    parser = get_parser('Visualize GT Soft Labels (Intent + Map Attn)')
    parser = add_base_args(parser)
    parser.add_argument('--test_on_val', type=str2bool, default=False)
    parser.add_argument('--shuffle_test', type=str2bool, default=False)
    parser.add_argument('--seq_interval', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_scenes', type=int, default=10,
                        help='Number of scenes to visualize')
    # Intent params
    parser.add_argument('--num_intents', type=int, default=9)
    parser.add_argument('--intent_sigma', type=float, default=0.5)
    # Map attn soft label params
    parser.add_argument('--map_gt_steps', type=int, default=6)
    parser.add_argument('--map_gt_decay_lambda', type=float, default=0.3)
    parser.add_argument('--map_gauss_sigma_d', type=float, default=0.8)
    # Potential weighting params
    parser.add_argument('--k_env_solid', type=float, default=0.8)
    parser.add_argument('--k_env_dashed', type=float, default=0.1)
    # conv_kernel_list and conv_stride_list are already in add_base_args

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'Ignoring unknown args from config: {unknown[:5]}...')
    return dict2obj(vars(args)), vars(args)


# =================================================================
# Map rendering
# =================================================================

def render_map_bg(map_obs_np):
    """Render map layers as RGB background image.
    map_obs_np: (C, H, W). After .T: display (W, H).
    Returns: (W, H, 3) RGB for imshow(origin='lower').
    """
    map_color_list = ['darkgray', 'coral', 'orange', 'gold', 'lightblue', 'lightblue']
    map_alpha_list = [1.0, 0.6, 0.6, 0.6, 1.0, 0.5]
    disp_h, disp_w = map_obs_np.shape[2], map_obs_np.shape[1]
    img = np.ones((disp_h, disp_w, 3))
    for i in range(map_obs_np.shape[0]):
        c = np.array(mcolors.to_rgba(map_color_list[i % len(map_color_list)])[:3])
        alpha = map_alpha_list[i % len(map_alpha_list)]
        mask = map_obs_np[i].T
        for ch in range(3):
            img[:, :, ch] = img[:, :, ch] * (1 - mask * alpha) + c[ch] * mask * alpha
    return img


def world_to_crop_pixel(pos_world, center_pos, bounds, L, W):
    """Convert world coordinates to crop pixel coordinates."""
    xy = pos_world[:, :2] if pos_world.shape[-1] > 2 else pos_world
    cx, cy = center_pos[0], center_pos[1]
    hx, hy = center_pos[2], center_pos[3]
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    rel_l = dx * hx + dy * hy
    rel_w = -dx * hy + dy * hx
    pix_l = (rel_l - bounds[0]) / (bounds[2] - bounds[0]) * L
    pix_w = (rel_w - bounds[1]) / (bounds[3] - bounds[1]) * W
    return np.stack([pix_l, pix_w], axis=1)


# =================================================================
# Intent GT soft label computation
# =================================================================

def compute_intent_gt_labels(ego_past_last, ego_future, state_normalizer,
                             dt=0.5, num_intents=9, sigma=0.5, device='cpu'):
    """
    Compute GT intent soft labels from (acc, yaw_rate).

    :param ego_past_last: (N_ego, 6) last past state (normalized)
    :param ego_future: (N_ego, FT, 6) future states (normalized)
    :param state_normalizer: normalizer with mean_vals/std_vals
    :param dt: timestep
    :param num_intents: 9
    :param sigma: Gaussian softmax sigma
    :return: gt_soft_label (N_ego, FT, 9), ego_acc (N_ego, FT), ego_yaw (N_ego, FT)
    """
    s_mean = state_normalizer.mean_vals[4].to(device)
    s_std = state_normalizer.std_vals[4].to(device)
    hdot_mean = state_normalizer.mean_vals[5].to(device)
    hdot_std = state_normalizer.std_vals[5].to(device)
    ninfo = NUSC_NORM_STATS[('car', 'truck')]
    a_std_norm = ninfo['a'][1]
    hdot_std_norm = ninfo['hdot'][1]

    # Acceleration: (speed_next - speed_prev) / dt, normalized by a_std
    ego_speed_raw = ego_future[:, :, 4] * s_std + s_mean  # (N, FT)
    prev_speed_raw = ego_past_last[:, 4:5] * s_std + s_mean  # (N, 1)
    ego_prev_speed_raw = torch.cat([prev_speed_raw, ego_speed_raw[:, :-1]], dim=1)  # (N, FT)
    raw_acc = (ego_speed_raw - ego_prev_speed_raw) / dt
    ego_acc = raw_acc / a_std_norm  # (N, FT)

    # Yaw rate: unnormalize hdot, then normalize by hdot_std from NUSC_NORM_STATS
    raw_yaw_rate = ego_future[:, :, 5] * hdot_std + hdot_mean  # (N, FT)
    ego_yaw = raw_yaw_rate / hdot_std_norm  # (N, FT)

    # Build prototypes: 3x3 grid in [-1, 1]
    n_acc = int(np.sqrt(num_intents))
    n_yaw = num_intents // n_acc
    acc_vals = torch.linspace(-1, 1, n_acc, device=device)
    yaw_vals = torch.linspace(-1, 1, n_yaw, device=device)
    prototypes = torch.stack(torch.meshgrid(acc_vals, yaw_vals), dim=-1).reshape(-1, 2)

    # Gaussian softmax over prototypes
    gt_xy = torch.stack([ego_acc, ego_yaw], dim=-1)  # (N, FT, 2)
    dist_sq = ((gt_xy.unsqueeze(-2) - prototypes.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
    gt_soft_label = torch.softmax(-dist_sq / (2 * sigma ** 2), dim=-1)  # (N, FT, 9)

    return gt_soft_label.cpu().numpy(), ego_acc.cpu().numpy(), ego_yaw.cpu().numpy()


# =================================================================
# Map attn GT soft label computation (mirrors loss.py _make_soft_label)
# =================================================================

def compute_rf_params(conv_kernel_list, conv_stride_list):
    """Compute CNN receptive field params for early conv layers (0-2).
    Returns (total_stride, rf_offset) such that:
        token[m] RF center pixel = m * total_stride + rf_offset
    """
    total_stride = 1
    rf_offset = 0.0
    for i in range(3):
        rf_offset += (conv_kernel_list[i] - 1) / 2.0 * total_stride
        total_stride *= conv_stride_list[i]
    return total_stride, rf_offset


def meters_to_token(local_m, bounds_min, bounds_max, pix_size, total_stride, rf_offset):
    """Convert local-frame meters to token index using CNN RF centers.
    local_m: position in meters (local frame, 0 = agent)
    Returns: fractional token index
    """
    pixel = (local_m - bounds_min) / (bounds_max - bounds_min) * pix_size
    return (pixel - rf_offset) / total_stride


def compute_map_soft_label(ego_frames_world, gt_future_world, grid_size=29,
                           bounds=None, pix_size=256,
                           sigma_d=0.8, decay_lambda=0.3, map_gt_steps=6,
                           conv_kernel_list=None, conv_stride_list=None):
    """
    Compute per-step map attn GT soft label for a single ego agent.

    :param ego_frames_world: (FT+1, 4) [past_last, gt_future[0], ..., gt_future[FT-1]]
    :param gt_future_world: (FT, 4+) GT future positions in world coords
    :param grid_size: 29 or 57
    :param bounds: [-17, -38.5, 60, 38.5]
    :param pix_size: 256
    :param conv_kernel_list: [7, 5, 5, ...] for RF center calculation
    :param conv_stride_list: [2, 2, 1, ...] for RF center calculation
    :return: (FT, grid_size, grid_size)
    """
    if bounds is None:
        bounds = [-17.0, -38.5, 60.0, 38.5]
    if conv_kernel_list is None:
        conv_kernel_list = [7, 5, 5, 3, 3, 3]
    if conv_stride_list is None:
        conv_stride_list = [2, 2, 2, 2, 2, 2]

    FT = gt_future_world.shape[0]
    total_stride, rf_offset = compute_rf_params(conv_kernel_list, conv_stride_list)

    gi_range = np.arange(grid_size, dtype=np.float32)
    gj_range = np.arange(grid_size, dtype=np.float32)
    grid_i, grid_j = np.meshgrid(gi_range, gj_range, indexing='ij')
    g_i = grid_i.reshape(-1)
    g_j = grid_j.reshape(-1)
    inv_sigma_d_sq = 1.0 / (sigma_d ** 2)

    all_labels = np.zeros((FT, grid_size, grid_size))

    for t in range(FT):
        remaining = min(map_gt_steps, FT - t - 1)
        if remaining <= 0:
            continue

        frame = ego_frames_world[t]
        fx, fy, fhx, fhy = frame[0], frame[1], frame[2], frame[3]

        agent_gi = meters_to_token(0.0, bounds[0], bounds[2], pix_size, total_stride, rf_offset)
        agent_gj = meters_to_token(0.0, bounds[1], bounds[3], pix_size, total_stride, rf_offset)

        wp_i = [agent_gi]
        wp_j = [agent_gj]
        for k in range(remaining):
            ft_idx = t + 1 + k
            if ft_idx >= FT:
                break
            dx = gt_future_world[ft_idx, 0] - fx
            dy = gt_future_world[ft_idx, 1] - fy
            local_l = dx * fhx + dy * fhy
            local_w = -dx * fhy + dy * fhx
            gi_f = meters_to_token(local_l, bounds[0], bounds[2], pix_size, total_stride, rf_offset)
            gj_f = meters_to_token(local_w, bounds[1], bounds[3], pix_size, total_stride, rf_offset)
            wp_i.append(gi_f)
            wp_j.append(gj_f)

        wp_i = np.array(wp_i)
        wp_j = np.array(wp_j)
        K = len(wp_i) - 1
        if K == 0:
            continue

        seg_di = wp_i[1:] - wp_i[:-1]
        seg_dj = wp_j[1:] - wp_j[:-1]
        seg_len = np.sqrt(seg_di**2 + seg_dj**2 + 1e-8)
        cum_len = np.concatenate([[0], np.cumsum(seg_len)])
        total_len = cum_len[-1]

        to_grid_i = g_i[:, None] - wp_i[:K][None, :]
        to_grid_j = g_j[:, None] - wp_j[:K][None, :]
        dir_i = seg_di / (seg_len + 1e-8)
        dir_j = seg_dj / (seg_len + 1e-8)
        proj = to_grid_i * dir_i[None, :] + to_grid_j * dir_j[None, :]
        proj_clamped = np.clip(proj, 0, seg_len[None, :])

        near_i = wp_i[:K][None, :] + proj_clamped * dir_i[None, :]
        near_j = wp_j[:K][None, :] + proj_clamped * dir_j[None, :]

        d_sq = (g_i[:, None] - near_i)**2 + (g_j[:, None] - near_j)**2
        arc_at_proj = cum_len[:K][None, :] + proj_clamped

        nearest_seg = d_sq.argmin(axis=1)
        arange_idx = np.arange(len(g_i))
        min_d_sq = d_sq[arange_idx, nearest_seg]
        min_arc = arc_at_proj[arange_idx, nearest_seg]

        raw_proj_seg0 = proj[:, 0]
        behind_mask = (nearest_seg == 0) & (raw_proj_seg0 < 0)

        lateral = np.exp(-0.5 * min_d_sq * inv_sigma_d_sq)
        arc_normalized = min_arc / (total_len + 1e-8) * K
        longitudinal = np.exp(-decay_lambda * arc_normalized)

        cell_value = longitudinal * lateral
        cell_value[behind_mask] = 0.0

        s = cell_value.sum()
        if s > 0:
            cell_value /= s

        all_labels[t] = cell_value.reshape(grid_size, grid_size)

    return all_labels


def apply_potential_weight(soft_label_grid, map_raster_29, k_solid=0.8, k_dashed=0.1):
    """
    Apply drivable potential weighting to soft label.

    :param soft_label_grid: (grid_size, grid_size) soft label
    :param map_raster_29: (C, grid_size, grid_size) pooled raster
    :param k_solid: solid line penalty
    :param k_dashed: dashed line penalty
    :return: (grid_size, grid_size) weighted soft label
    """
    drivable = map_raster_29[0]
    solid = map_raster_29[1] if map_raster_29.shape[0] > 1 else np.zeros_like(drivable)
    dashed = map_raster_29[2] if map_raster_29.shape[0] > 2 else np.zeros_like(drivable)

    w = drivable * (1.0 - k_solid * solid) * (1.0 - k_dashed * dashed)
    w = np.clip(w, 0.01, None)

    weighted = soft_label_grid * w
    s = weighted.sum()
    if s > 0:
        weighted /= s
    return weighted


# =================================================================
# Visualization
# =================================================================

def visualize_intent_gt(intent_labels, ego_acc, ego_yaw, out_path, agent_idx=0):
    """
    Visualize intent GT soft labels as 3x3 heatmaps per timestep.

    :param intent_labels: (FT, 9) soft labels
    :param ego_acc: (FT,) normalized acceleration
    :param ego_yaw: (FT,) normalized yaw rate
    """
    FT = intent_labels.shape[0]
    show_steps = [0, 2, 4, 6, 8, 10, 11]
    show_steps = [s for s in show_steps if s < FT]
    n_cols = len(show_steps)

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7),
                             gridspec_kw={'height_ratios': [3, 1]})

    for panel_idx, t in enumerate(show_steps):
        # Top row: 3x3 heatmap
        ax = axes[0, panel_idx]
        grid = intent_labels[t].reshape(3, 3)
        im = ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1.0,
                       origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
        # Mark GT position
        ax.plot(ego_yaw[t], ego_acc[t], 'k*', markersize=12, markeredgecolor='white',
                markeredgewidth=0.8)
        # Grid labels
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xticklabels(['L', '0', 'R'], fontsize=8)
        ax.set_yticklabels(['Dec', '0', 'Acc'], fontsize=8)
        ax.set_title(f't={t*0.5:.1f}s\nacc={ego_acc[t]:.2f} yaw={ego_yaw[t]:.2f}',
                     fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j - 1, i - 1, f'{grid[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if grid[i, j] < 0.5 else 'white')

        # Bottom row: acc/yaw time series up to this step
        ax2 = axes[1, panel_idx]
        ax2.plot(np.arange(t+1) * 0.5, ego_acc[:t+1], 'b-o', markersize=2, label='acc')
        ax2.plot(np.arange(t+1) * 0.5, ego_yaw[:t+1], 'r-o', markersize=2, label='yaw')
        ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax2.set_xlim(-0.1, FT * 0.5)
        ax2.set_ylim(-2.5, 2.5)
        ax2.set_xlabel('time(s)', fontsize=7)
        ax2.tick_params(labelsize=6)
        if panel_idx == 0:
            ax2.legend(fontsize=6, loc='upper left')

    plt.suptitle(f'GT Intent Soft Label — Agent {agent_idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def visualize_map_soft_label(map_obs_dict, soft_labels, soft_labels_weighted,
                             gt_traj_pix, out_path, agent_idx=0,
                             map_raster_29_dict=None,
                             conv_kernel_list=None, conv_stride_list=None,
                             pix_size=256):
    """
    Visualize map attn GT soft labels: original vs potential-weighted.

    Row 0: Original soft label on GT-position map crops
    Row 1: Potential-weighted soft label on GT-position map crops
    Row 2 (optional): Pooled raster channels (drivable/solid/dashed)
    """
    from scipy.ndimage import zoom

    show_steps = [0, 3, 6, 9, 11]
    FT = soft_labels.shape[0]
    show_steps = [s for s in show_steps if s < FT]
    n_cols = len(show_steps) + 1
    has_raster = map_raster_29_dict is not None
    n_rows = 3 if has_raster else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    traj_map_bg = render_map_bg(map_obs_dict['traj'])
    bounds = [-17.0, -38.5, 60.0, 38.5]

    # RF center mapping for heatmap overlay
    if conv_kernel_list is None:
        conv_kernel_list = [7, 5, 5, 3, 3, 3]
    if conv_stride_list is None:
        conv_stride_list = [2, 2, 2, 2, 2, 2]
    total_stride, rf_offset = compute_rf_params(conv_kernel_list, conv_stride_list)
    grid_size = soft_labels.shape[1]

    # Token[m] RF center = m * total_stride + rf_offset (in pixels)
    # extent for imshow: token[0] center → pixel rf_offset, token[G-1] center → pixel (G-1)*ts + rf_offset
    # imshow extent = [left, right, bottom, top] where pixel centers are at the edges
    token0_pix = rf_offset
    tokenN_pix = (grid_size - 1) * total_stride + rf_offset
    # half-token padding for extent (imshow maps pixel centers to extent edges)
    half_ts = total_stride / 2.0

    def _overlay_heatmap(ax, map_bg_t, grid_data, title):
        disp_h_t, disp_w_t = map_bg_t.shape[:2]
        ax.imshow(map_bg_t, origin='lower', extent=[0, pix_size, 0, pix_size])
        grid_disp = grid_data.T
        vmax = grid_disp.max()
        if vmax > 0:
            grid_disp = grid_disp / vmax
        cmap = plt.cm.jet
        heatmap = cmap(grid_disp)
        heatmap[:, :, 3] = grid_disp * 0.7
        # Place heatmap at correct RF center positions
        # extent: [x_left, x_right, y_bottom, y_top]
        # x-axis = longitudinal (grid dim 0), y-axis = lateral (grid dim 1)
        # After .T: grid_disp[j, i] → x=token_i, y=token_j
        extent_x0 = token0_pix - half_ts
        extent_x1 = tokenN_pix + half_ts
        extent_y0 = token0_pix - half_ts  # same CNN for both dims
        extent_y1 = tokenN_pix + half_ts
        ax.imshow(heatmap, origin='lower', interpolation='bilinear',
                  extent=[extent_x0, extent_x1, extent_y0, extent_y1])
        # Mark GT agent position (white star) — pixel coords
        agent_pix_x = (0.0 - bounds[0]) / (bounds[2] - bounds[0]) * pix_size
        agent_pix_y = (0.0 - bounds[1]) / (bounds[3] - bounds[1]) * pix_size
        ax.plot(agent_pix_x, agent_pix_y, 'w*', markersize=10)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, pix_size)
        ax.set_ylim(0, pix_size)
        ax.axis('off')

    def _draw_traj(ax, map_bg):
        ax.imshow(map_bg, origin='lower', extent=[0, pix_size, 0, pix_size])
        if gt_traj_pix is not None:
            ax.plot(gt_traj_pix[:, 0], gt_traj_pix[:, 1],
                    'w-o', markersize=3, linewidth=1.5, label='GT')
            ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(0, pix_size)
        ax.set_ylim(0, pix_size)
        ax.axis('off')

    # Row 0: Original soft label
    ax = axes[0, 0]
    _draw_traj(ax, traj_map_bg)
    ax.set_title(f'Agent {agent_idx}\nGT trajectory', fontsize=10)

    for panel_idx, t in enumerate(show_steps):
        map_bg_t = render_map_bg(map_obs_dict[t])
        _overlay_heatmap(axes[0, panel_idx + 1], map_bg_t, soft_labels[t],
                        f'Original\nt={t*0.5:.1f}s')

    # Row 1: Potential-weighted soft label
    ax = axes[1, 0]
    _draw_traj(ax, traj_map_bg)
    ax.set_title(f'Agent {agent_idx}\n+ potential weight', fontsize=10)

    for panel_idx, t in enumerate(show_steps):
        map_bg_t = render_map_bg(map_obs_dict[t])
        _overlay_heatmap(axes[1, panel_idx + 1], map_bg_t, soft_labels_weighted[t],
                        f'Weighted\nt={t*0.5:.1f}s')

    # Row 2: Pooled raster channels (drivable / solid / dashed)
    if has_raster:
        ax = axes[2, 0]
        ax.axis('off')
        ax.set_title(f'Pooled raster\n({soft_labels.shape[1]}x{soft_labels.shape[1]})', fontsize=10)

        ch_names = ['Drivable', 'Solid', 'Dashed']
        ch_cmaps = ['Greens', 'Reds', 'Oranges']
        for panel_idx, t in enumerate(show_steps):
            ax = axes[2, panel_idx + 1]
            raster = map_raster_29_dict.get(t)
            if raster is not None:
                # Show composite: drivable(green) + solid(red) + dashed(orange)
                n_ch = raster.shape[0]
                gs = raster.shape[1]
                disp = np.zeros((gs, gs, 3))
                if n_ch > 0:
                    disp[:, :, 1] = raster[0].T * 0.6  # drivable → green
                if n_ch > 1:
                    disp[:, :, 0] = raster[1].T * 0.8  # solid → red
                if n_ch > 2:
                    disp[:, :, 0] += raster[2].T * 0.3  # dashed → orange-ish
                    disp[:, :, 1] += raster[2].T * 0.2
                disp = np.clip(disp, 0, 1)
                ax.imshow(disp, origin='lower')
                ax.set_title(f'Raster t={t*0.5:.1f}s', fontsize=8)
            else:
                ax.set_title(f'(no raster)', fontsize=8)
            ax.axis('off')

    plt.suptitle(f'Map Attn GT Soft Label — Agent {agent_idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# =================================================================
# Main
# =================================================================

def main():
    cfg, cfg_dict = parse_cfg()

    device = f'cuda:{cfg.gpu}'
    print(f'Using device: {device}')

    # Load map environment
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(data_path,
                        bounds=cfg.map_obs_bounds,
                        L=cfg.map_obs_size_pix,
                        W=cfg.map_obs_size_pix,
                        layers=cfg.map_layers,
                        device=device)

    # Load dataset
    split = 'train' if not cfg.test_on_val else 'val'
    print(f'Using split: {split}')
    dataset = FITDataset(data_path, map_env,
                         split=split,
                         categories=cfg.agent_types,
                         npast=cfg.past_len,
                         nfuture=cfg.future_len,
                         dt=cfg.dt,
                         reduce_cats=cfg.reduce_cats,
                         seq_interval=cfg.seq_interval)

    loader = GraphDataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             pin_memory=False,
                             worker_init_fn=lambda _: np.random.seed())

    state_normalizer = dataset.get_state_normalizer()
    att_normalizer = dataset.get_att_normalizer()

    # Output directory
    out_dir = os.path.join(cfg.out, 'viz_gt_labels')
    mkdir(out_dir)
    print(f'Output directory: {out_dir}')

    bounds = cfg.map_obs_bounds
    pix_size = cfg.map_obs_size_pix
    grid_size = compute_map_token_spatial(cfg.conv_kernel_list, cfg.conv_stride_list, pix_size)
    print(f'Map token grid_size: {grid_size}x{grid_size} (stride={cfg.conv_stride_list[:3]})')
    FT = cfg.future_len
    PT = cfg.past_len

    scene_count = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if scene_count >= cfg.num_scenes:
                break

            scene_graph, map_idx = data
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            NA = scene_graph.past.size(0)

            # Identify ego agent
            ego_mask = scene_graph.sem[:, 0] == 1
            if not ego_mask.any():
                ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
                ego_mask[0] = True

            ego_indices = ego_mask.nonzero(as_tuple=True)[0]
            num_ego = ego_indices.numel()

            # Unnormalize scene graph for world coords
            normalize_scene_graph(scene_graph, state_normalizer, att_normalizer, unnorm=True)

            print(f'\nScene {i}: NA={NA}, num_ego={num_ego}')

            for ego_local_idx in range(num_ego):
                ego_global_idx = ego_indices[ego_local_idx].item()

                ego_pos = scene_graph.past[ego_global_idx, -1, :4].cpu().numpy()  # world
                gt_future_world = scene_graph.future[ego_global_idx].cpu().numpy()  # (FT, 6)
                mapix = map_idx[scene_graph.batch[ego_global_idx]].unsqueeze(0)

                # ========================================
                # 1. Intent GT soft label
                # ========================================
                # Re-normalize for intent computation (needs normalized states)
                normalize_scene_graph(scene_graph, state_normalizer, att_normalizer, unnorm=False)
                ego_past_last = scene_graph.past[ego_global_idx:ego_global_idx+1, -1, :]
                ego_future_norm = scene_graph.future[ego_global_idx:ego_global_idx+1]

                intent_labels, ego_acc, ego_yaw = compute_intent_gt_labels(
                    ego_past_last, ego_future_norm, state_normalizer,
                    dt=cfg.dt, num_intents=cfg.num_intents,
                    sigma=cfg.intent_sigma, device=device
                )
                # Back to unnormalized
                normalize_scene_graph(scene_graph, state_normalizer, att_normalizer, unnorm=True)

                intent_path = os.path.join(out_dir,
                    f'scene{i:04d}_ego{ego_local_idx}_intent_gt.png')
                visualize_intent_gt(
                    intent_labels[0], ego_acc[0], ego_yaw[0],
                    intent_path, agent_idx=ego_global_idx
                )

                # ========================================
                # 2. Map attn GT soft label
                # ========================================
                ego_frames_world = np.zeros((FT + 1, 4))
                ego_frames_world[0] = ego_pos[:4]
                ego_frames_world[1:] = gt_future_world[:, :4]

                soft_labels = compute_map_soft_label(
                    ego_frames_world, gt_future_world[:, :4],
                    grid_size=grid_size, bounds=bounds, pix_size=pix_size,
                    sigma_d=cfg.map_gauss_sigma_d,
                    decay_lambda=cfg.map_gt_decay_lambda,
                    map_gt_steps=cfg.map_gt_steps,
                    conv_kernel_list=cfg.conv_kernel_list,
                    conv_stride_list=cfg.conv_stride_list
                )

                # Build per-step map crops and pooled raster
                show_steps = [0, 3, 6, 9, 11]
                show_steps = [s for s in show_steps if s < FT]

                map_obs_dict = {}
                map_raster_29_dict = {}

                # t=0 map crop
                ego_pos_tensor = torch.tensor(ego_pos[:4], device=device, dtype=torch.float32).unsqueeze(0)
                map_obs_t0 = map_env.get_map_crop_pos(ego_pos_tensor, mapix).cpu().numpy()[0]
                map_obs_dict['traj'] = map_obs_t0
                map_obs_dict[0] = map_obs_t0

                # Pool to 29x29
                map_obs_t0_tensor = torch.tensor(map_obs_t0.astype(np.float32)).unsqueeze(0)
                pooled_t0 = F.adaptive_avg_pool2d(map_obs_t0_tensor, grid_size).numpy()[0]
                map_raster_29_dict[0] = pooled_t0

                for t in show_steps:
                    if t == 0:
                        continue
                    frame_pos = gt_future_world[t - 1, :4]
                    frame_tensor = torch.tensor(frame_pos, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                    map_obs_t = map_env.get_map_crop_pos(frame_tensor, mapix).cpu().numpy()[0]
                    map_obs_dict[t] = map_obs_t

                    map_obs_t_tensor = torch.tensor(map_obs_t.astype(np.float32)).unsqueeze(0)
                    pooled_t = F.adaptive_avg_pool2d(map_obs_t_tensor, grid_size).numpy()[0]
                    map_raster_29_dict[t] = pooled_t

                # Apply potential weighting per step
                soft_labels_weighted = soft_labels.copy()
                for t in range(FT):
                    # Get raster for this step
                    if t == 0:
                        raster_t = pooled_t0
                    elif t in map_raster_29_dict:
                        raster_t = map_raster_29_dict[t]
                    else:
                        # Compute raster for non-show steps too
                        if t > 0:
                            frame_pos = gt_future_world[t - 1, :4]
                            frame_tensor = torch.tensor(frame_pos, dtype=torch.float32,
                                                        device=device).unsqueeze(0)
                            map_obs_t = map_env.get_map_crop_pos(frame_tensor, mapix).cpu().float()
                            raster_t = F.adaptive_avg_pool2d(map_obs_t, grid_size).numpy()[0]
                        else:
                            raster_t = pooled_t0

                    soft_labels_weighted[t] = apply_potential_weight(
                        soft_labels[t], raster_t,
                        k_solid=cfg.k_env_solid, k_dashed=cfg.k_env_dashed
                    )

                # GT trajectory pixel coords on t=0 map
                gt_traj_pix = world_to_crop_pixel(
                    gt_future_world[:, :2], ego_pos, bounds, pix_size, pix_size)

                map_path = os.path.join(out_dir,
                    f'scene{i:04d}_ego{ego_local_idx}_map_soft_label.png')
                visualize_map_soft_label(
                    map_obs_dict, soft_labels, soft_labels_weighted,
                    gt_traj_pix, map_path,
                    agent_idx=ego_global_idx,
                    map_raster_29_dict=map_raster_29_dict,
                    conv_kernel_list=cfg.conv_kernel_list,
                    conv_stride_list=cfg.conv_stride_list,
                    pix_size=pix_size
                )

            scene_count += 1

    print(f'\nDone! {scene_count} scenes visualized -> {out_dir}')


if __name__ == '__main__':
    main()
