#!/usr/bin/env python3
"""
Map Attention Heatmap + Intent Codebook Visualization

모델의 map attention weights를 29x29 grid로 reshape하여
map crop 위에 overlay하고, intent codebook slot 선택 분포를 시각화합니다.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets.nuscenes_utils import (
    normalize_scene_graph, render_map_observation, make_rgba, get_map_obs
)
from models.trafficplanner_model import TrafficPlannerModel
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.config import get_parser, add_base_args
from utils.torch import get_device, load_state


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
    parser = get_parser('Visualize Map Attention & Intent Codebook')
    parser = add_base_args(parser)
    parser.add_argument('--test_on_val', type=str2bool, default=True)
    parser.add_argument('--shuffle_test', type=str2bool, default=False)
    parser.add_argument('--seq_interval', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    # Model arch params
    parser.add_argument('--z_local_size', type=int, default=32)
    parser.add_argument('--num_intents', type=int, default=9)
    parser.add_argument('--sur_pred_dim', type=int, default=2)
    parser.add_argument('--map_recrop', type=str2bool, default=True)
    parser.add_argument('--trans_num_layers', type=int, default=4)
    parser.add_argument('--trans_d_model', type=int, default=128)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--trans_ffn_dim', type=int, default=512)
    parser.add_argument('--trans_dropout', type=float, default=0.1)
    parser.add_argument('--use_ego_z_local', type=str2bool, default=True)
    parser.add_argument('--use_sur_z_local', type=str2bool, default=True)

    # Potential field (for config compat)
    parser.add_argument('--k_veh_repel', type=float, default=1.0)
    parser.add_argument('--sigma_veh', type=float, default=1.5)
    parser.add_argument('--k_env_boundary', type=float, default=1.0)
    parser.add_argument('--sigma_env_boundary', type=float, default=0.5)
    parser.add_argument('--k_env_solid', type=float, default=0.8)
    parser.add_argument('--sigma_env_solid', type=float, default=0.5)
    parser.add_argument('--k_env_dashed', type=float, default=0.1)
    parser.add_argument('--sigma_env_dashed', type=float, default=0.3)
    parser.add_argument('--use_lane_lines', type=str2bool, default=True)

    # Viz params
    parser.add_argument('--num_scenes', type=int, default=10,
                        help='Number of scenes to visualize')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'Ignoring unknown args from config: {unknown[:5]}...')
    return dict2obj(vars(args)), vars(args)


def render_map_bg(map_obs_np):
    """Render map layers as RGB background image.
    map_obs_np: (C, H, W) numpy array, H=longitudinal, W=lateral
    Map crops are stored transposed, so we must .T each layer.
    After .T: shape (W, H) → imshow rows=W(lat), cols=H(long)
    With origin='lower': x-axis=cols=H=longitudinal, y-axis=rows=W=lateral
    Returns: (W, H, 3) RGB image for use with imshow(origin='lower')
    """
    # Only render binary channels (ch0-2). ch3+ are distance fields (continuous 0~1), skip for viz.
    map_color_list = ['darkgray', 'coral', 'orange']
    map_alpha_list = [1.0, 0.6, 0.6]
    num_binary_ch = min(map_obs_np.shape[0], len(map_color_list))

    # After .T: display shape is (W, H) — matches nuscenes_utils convention
    disp_h, disp_w = map_obs_np.shape[2], map_obs_np.shape[1]  # W, H after transpose
    img = np.ones((disp_h, disp_w, 3))

    for i in range(num_binary_ch):
        c = np.array(mcolors.to_rgba(map_color_list[i])[:3])
        alpha = map_alpha_list[i]
        mask = map_obs_np[i].T  # (H,W) → (W,H), matching nuscenes_utils
        for ch in range(3):
            img[:, :, ch] = img[:, :, ch] * (1 - mask * alpha) + c[ch] * mask * alpha

    return img


def remap_attn_to_gt_frame(attn_grid, pred_frame, gt_frame, grid_size, bounds,
                           pix_size=256, rf_stride=None, rf_offset=None):
    """
    Remap model attention from predicted-frame local coords to GT-frame local coords.

    Uses CNN receptive field centers: token[m] center = rf_offset + m * rf_stride pixels.

    :param attn_grid: (grid_size, grid_size) attention in pred frame (H'=long, W'=lat)
    :param pred_frame: (4,) predicted frame (x, y, hx, hy) in world coords
    :param gt_frame: (4,) GT frame (x, y, hx, hy) in world coords
    :param grid_size: 29
    :param bounds: [low_l, low_w, high_l, high_w]
    :param pix_size: pixel size of map crop (256)
    :param rf_stride: CNN receptive field stride
    :param rf_offset: CNN receptive field offset
    :returns: (grid_size, grid_size) attention remapped to GT frame
    """
    if rf_stride is None:
        rf_stride = pix_size / grid_size
    if rf_offset is None:
        rf_offset = 0.0

    # Token index → pixel center → local meters (pred frame)
    gi = np.arange(grid_size, dtype=np.float32)
    gj = np.arange(grid_size, dtype=np.float32)
    gi_grid, gj_grid = np.meshgrid(gi, gj, indexing='ij')  # (GS, GS)

    pix_l = rf_offset + gi_grid * rf_stride
    pix_w = rf_offset + gj_grid * rf_stride
    local_l = pix_l / pix_size * (bounds[2] - bounds[0]) + bounds[0]
    local_w = pix_w / pix_size * (bounds[3] - bounds[1]) + bounds[1]

    # Pred frame local → world
    px, py, phx, phy = pred_frame
    world_x = px + local_l * phx - local_w * phy
    world_y = py + local_l * phy + local_w * phx

    # World → GT frame local
    gx, gy, ghx, ghy = gt_frame
    dx = world_x - gx
    dy = world_y - gy
    gt_local_l = dx * ghx + dy * ghy
    gt_local_w = -dx * ghy + dy * ghx

    # GT local meters → pixel → GT grid indices (continuous, RF-aware)
    gt_pix_l = (gt_local_l - bounds[0]) / (bounds[2] - bounds[0]) * pix_size
    gt_pix_w = (gt_local_w - bounds[1]) / (bounds[3] - bounds[1]) * pix_size
    gt_gi = (gt_pix_l - rf_offset) / rf_stride
    gt_gj = (gt_pix_w - rf_offset) / rf_stride

    # Scatter attention weights onto GT grid (bilinear splatting)
    out = np.zeros((grid_size, grid_size), dtype=np.float64)
    weights_sum = np.zeros((grid_size, grid_size), dtype=np.float64)

    gi_floor = np.floor(gt_gi).astype(np.int32)
    gj_floor = np.floor(gt_gj).astype(np.int32)
    gi_frac = gt_gi - gi_floor
    gj_frac = gt_gj - gj_floor

    for di in range(2):
        for dj in range(2):
            ii = gi_floor + di
            jj = gj_floor + dj
            wi = (1 - gi_frac) if di == 0 else gi_frac
            wj = (1 - gj_frac) if dj == 0 else gj_frac
            w = wi * wj
            mask = (ii >= 0) & (ii < grid_size) & (jj >= 0) & (jj < grid_size)
            np.add.at(out, (ii[mask], jj[mask]), (attn_grid * w)[mask])
            np.add.at(weights_sum, (ii[mask], jj[mask]), w[mask])

    # Normalize to preserve total attention mass
    valid = weights_sum > 0
    out[valid] /= weights_sum[valid]

    return out.astype(np.float32)


def compute_soft_label_grid(gt_future_world, ego_frames_world, grid_size, bounds, pix_size,
                             sigma_d=0.8, decay_lambda=0.3, map_gt_steps=6,
                             rf_stride=None, rf_offset=None):
    """
    Compute soft label for a single agent at each timestep.
    Mirrors _make_soft_label logic from trafficplanner_loss.py.

    Uses CNN receptive field centers: token[m] center = rf_offset + m * rf_stride pixels.

    :param gt_future_world: (FT, 4+) GT future in world coords (x,y,hx,hy,...)
    :param ego_frames_world: (FT+1, 4) frame positions: [past_last, gt_future[0], ..., gt_future[FT-1]]
    :param grid_size: 29
    :param bounds: [-17, -38.5, 60, 38.5]
    :param pix_size: 256
    :param rf_stride: CNN receptive field stride
    :param rf_offset: CNN receptive field offset
    :return: (FT, grid_size, grid_size) soft labels in (H', W') = (long, lat) order
    """
    if rf_stride is None:
        rf_stride = pix_size / grid_size
    if rf_offset is None:
        rf_offset = 0.0

    FT = gt_future_world.shape[0]

    def _m2token_l(meters):
        pixel = (meters - bounds[0]) / (bounds[2] - bounds[0]) * pix_size
        return (pixel - rf_offset) / rf_stride

    def _m2token_w(meters):
        pixel = (meters - bounds[1]) / (bounds[3] - bounds[1]) * pix_size
        return (pixel - rf_offset) / rf_stride

    # Grid coordinates (token indices)
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

        # Frame at timestep t: t=0 → past last, t>0 → gt_future[t-1]
        frame = ego_frames_world[t]  # (4,): x, y, hx, hy
        fx, fy, fhx, fhy = frame[0], frame[1], frame[2], frame[3]

        # Agent pos in its own local frame is always (0, 0)
        agent_gi = _m2token_l(0.0)
        agent_gj = _m2token_w(0.0)

        # Polyline: agent pos + future waypoints in grid coords (relative to this frame)
        wp_i = [agent_gi]
        wp_j = [agent_gj]
        for k in range(remaining):
            ft_idx = t + 1 + k
            if ft_idx >= FT:
                break
            # Transform future waypoint to frame's local coords
            dx = gt_future_world[ft_idx, 0] - fx
            dy = gt_future_world[ft_idx, 1] - fy
            local_l = dx * fhx + dy * fhy       # longitudinal
            local_w = -dx * fhy + dy * fhx      # lateral
            gi_f = _m2token_l(local_l)
            gj_f = _m2token_w(local_w)
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

        # For each grid cell find nearest point on polyline
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


def visualize_map_attention(map_obs_dict, attn_weights, out_path, agent_idx=0,
                             timestep_labels=None,
                             soft_labels=None,
                             pred_pix_per_step=None,
                             gt_traj_pix=None, pred_traj_pix=None,
                             tf_attn_weights=None,
                             rf_stride=None, rf_offset=None, pix_size=256,
                             bounds=None):
    """
    Map attention heatmap overlay on per-timestep GT-position map crops.

    Row 0: GT soft label
    Row 1: TF mode attn (if available)
    Row 2: AR mode attn + predicted position (red dot)

    :param map_obs_dict: dict {timestep: (C, H, W)} GT-position map crops
    :param attn_weights: (FT, num_tokens) AR attention weights
    :param out_path: output path prefix
    :param agent_idx: agent index (for title)
    :param soft_labels: (FT, grid_size, grid_size) GT soft labels (optional)
    :param pred_pix_per_step: dict {timestep: (pix_x, pix_y)} predicted position in local pixel coords
    :param gt_traj_pix: (FT, 2) GT trajectory pixel coords on t=0 map crop
    :param pred_traj_pix: (FT, 2) predicted trajectory pixel coords on t=0 map crop
    """
    FT, num_tokens = attn_weights.shape
    grid_size = int(np.sqrt(num_tokens))  # 29

    show_steps = [0, 3, 6, 9, 11]
    show_steps = [s for s in show_steps if s < FT]

    # Trajectory overview uses t=0 map crop
    traj_map_bg = render_map_bg(map_obs_dict['traj'])

    has_tf = tf_attn_weights is not None
    if soft_labels is not None:
        n_rows = 3 if has_tf else 2
    else:
        n_rows = 1
    n_cols = len(show_steps) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # make 2D for consistent indexing

    from scipy.ndimage import zoom

    # RF params for heatmap alignment (closure vars for _overlay_heatmap)
    _viz_rf_stride = rf_stride
    _viz_rf_offset = rf_offset
    _viz_pix_size = pix_size

    def _overlay_heatmap(ax, map_bg_t, grid_data, title, pred_pix=None):
        disp_h_t, disp_w_t = map_bg_t.shape[:2]
        ax.imshow(map_bg_t, origin='lower', extent=[0, disp_w_t, 0, disp_h_t])
        # grid_data: (H', W') = (long, lat), transpose for display
        grid_disp = grid_data.T  # (lat, long)
        gs_h, gs_w = grid_disp.shape
        # Upsample grid for smooth display (keep in token space)
        up_factor = 4
        upsampled = zoom(grid_disp, (up_factor, up_factor), order=1)
        vmax = upsampled.max()
        if vmax > 0:
            upsampled = upsampled / vmax
        cmap = plt.cm.jet
        heatmap = cmap(upsampled)
        heatmap[:, :, 3] = upsampled * 0.7
        # RF-aware extent: token[i] center = (rf_offset + i * rf_stride) pixels
        # Map from pixel coords to display coords (display = pixel * disp/pix)
        _rf_s = _viz_rf_stride if _viz_rf_stride is not None else (disp_w_t / gs_w)
        _rf_o = _viz_rf_offset if _viz_rf_offset is not None else 0.0
        _pix = _viz_pix_size if _viz_pix_size is not None else disp_w_t
        # Half-stride padding so each token covers its full RF cell
        half = _rf_s * 0.5
        x0 = (_rf_o - half) / _pix * disp_w_t
        x1 = (_rf_o + (gs_w - 1) * _rf_s + half) / _pix * disp_w_t
        y0 = (_rf_o - half) / _pix * disp_h_t
        y1 = (_rf_o + (gs_h - 1) * _rf_s + half) / _pix * disp_h_t
        ax.imshow(heatmap, origin='lower', extent=[x0, x1, y0, y1])
        # Mark GT agent position (white star)
        _b = bounds if bounds is not None else [-17, -38.5, 60, 38.5]
        agent_pix_x = (0.0 - _b[0]) / (_b[2] - _b[0]) * disp_w_t
        agent_pix_y = (0.0 - _b[1]) / (_b[3] - _b[1]) * disp_h_t
        ax.plot(agent_pix_x, agent_pix_y, 'w*', markersize=10)
        # Mark predicted position (red dot)
        if pred_pix is not None:
            ax.plot(pred_pix[0], pred_pix[1], 'ro', markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, disp_w_t)
        ax.set_ylim(0, disp_h_t)
        ax.axis('off')

    def _draw_traj(ax):
        """Draw GT and pred trajectories on the t=0 map overview."""
        if gt_traj_pix is not None:
            ax.plot(gt_traj_pix[:, 0], gt_traj_pix[:, 1],
                    'w-o', markersize=3, linewidth=1.5, label='GT')
        if pred_traj_pix is not None:
            ax.plot(pred_traj_pix[:, 0], pred_traj_pix[:, 1],
                    'r-o', markersize=3, linewidth=1.5, label='Pred')
        if gt_traj_pix is not None or pred_traj_pix is not None:
            ax.legend(fontsize=7, loc='upper right')

    if soft_labels is not None:
        disp_h, disp_w = traj_map_bg.shape[:2]

        # Row 0, col 0: trajectory overview
        ax = axes[0, 0]
        ax.imshow(traj_map_bg, origin='lower')
        _draw_traj(ax)
        ax.set_title(f'Agent {agent_idx}\nTrajectory (t=0 map)', fontsize=10)
        ax.set_xlim(0, disp_w); ax.set_ylim(0, disp_h); ax.axis('off')

        # Row 0: GT soft label per timestep
        for panel_idx, t in enumerate(show_steps):
            map_bg_t = render_map_bg(map_obs_dict[t])
            _overlay_heatmap(axes[0, panel_idx + 1], map_bg_t, soft_labels[t],
                           f'GT soft label\nt={t*0.5:.1f}s (step {t})')

        if has_tf:
            # Row 1: TF mode attention (GT position map crops, no remap needed)
            ax = axes[1, 0]
            ax.imshow(traj_map_bg, origin='lower')
            _draw_traj(ax)
            ax.set_title(f'Agent {agent_idx}\nTF mode', fontsize=10)
            ax.set_xlim(0, disp_w); ax.set_ylim(0, disp_h); ax.axis('off')

            for panel_idx, t in enumerate(show_steps):
                map_bg_t = render_map_bg(map_obs_dict[t])
                tf_grid = tf_attn_weights[t].reshape(grid_size, grid_size)
                _overlay_heatmap(axes[1, panel_idx + 1], map_bg_t, tf_grid,
                               f'TF attn\nt={t*0.5:.1f}s (step {t})')

            # Row 2: AR mode attention (remapped to GT frame)
            ar_row = 2
        else:
            ar_row = 1

        ax = axes[ar_row, 0]
        ax.imshow(traj_map_bg, origin='lower')
        _draw_traj(ax)
        ax.set_title(f'Agent {agent_idx}\nAR mode', fontsize=10)
        ax.set_xlim(0, disp_w); ax.set_ylim(0, disp_h); ax.axis('off')

        for panel_idx, t in enumerate(show_steps):
            map_bg_t = render_map_bg(map_obs_dict[t])
            attn_grid = attn_weights[t].reshape(grid_size, grid_size)
            ppix = pred_pix_per_step.get(t) if pred_pix_per_step else None
            _overlay_heatmap(axes[ar_row, panel_idx + 1], map_bg_t, attn_grid,
                           f'AR attn\nt={t*0.5:.1f}s (step {t})',
                           pred_pix=ppix)
    else:
        # Single row: model attention only
        ax = axes[0, 0]
        disp_h, disp_w = traj_map_bg.shape[:2]
        ax.imshow(traj_map_bg, origin='lower')
        _draw_traj(ax)
        ax.set_title(f'Agent {agent_idx}\nTrajectory (t=0 map)', fontsize=10)
        ax.set_xlim(0, disp_w); ax.set_ylim(0, disp_h); ax.axis('off')

        for panel_idx, t in enumerate(show_steps):
            map_bg_t = render_map_bg(map_obs_dict[t])
            attn_grid = attn_weights[t].reshape(grid_size, grid_size)
            _overlay_heatmap(axes[0, panel_idx + 1], map_bg_t, attn_grid,
                           f't={t*0.5:.1f}s\n(step {t})')

    plt.suptitle(f'Map Attention — Agent {agent_idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved map attn: {out_path}')


def visualize_intent_codebook(intent_weights, out_path, agent_idx=0, num_intents=9,
                              gt_intent_label=None,
                              map_obs=None, gt_traj_pix=None, pred_traj_pix=None):
    """
    Intent codebook slot selection visualization.

    :param intent_weights: (FT, K) intent weights for one ego agent
    :param out_path: output path
    :param agent_idx: agent index
    :param num_intents: K
    :param gt_intent_label: (FT, K) GT soft label (optional)
    :param map_obs: (C, H, W) map crop (optional)
    :param gt_traj_pix: (FT, 2) GT trajectory in pixel coords (optional)
    :param pred_traj_pix: (FT, 2) pred trajectory in pixel coords (optional)
    """
    FT, K = intent_weights.shape
    has_gt = gt_intent_label is not None
    has_map = map_obs is not None

    n_cols = (1 if has_map else 0) + (1 if has_gt else 0) + 2  # map? + gt? + model + bar
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col = 0

    # Col 0: Map + Traj (optional)
    if has_map:
        ax = axes[col]
        bg = render_map_bg(map_obs)
        ax.imshow(bg, origin='upper')
        if gt_traj_pix is not None:
            ax.plot(gt_traj_pix[:, 0], gt_traj_pix[:, 1], 'o-',
                    color='white', markersize=3, linewidth=1.5, label='GT')
        if pred_traj_pix is not None:
            ax.plot(pred_traj_pix[:, 0], pred_traj_pix[:, 1], 'x-',
                    color='red', markersize=3, linewidth=1.5, label='Pred')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f'Map + Trajectory\n(Agent {agent_idx})', fontsize=10)
        ax.axis('off')
        col += 1

    # GT intent soft label heatmap (if available)
    if has_gt:
        ax = axes[col]
        im = ax.imshow(gt_intent_label.T, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Future Timestep', fontsize=10)
        ax.set_ylabel('Intent Slot', fontsize=10)
        ax.set_yticks(range(K))
        ax.set_xticks(range(0, FT, 2))
        ax.set_xticklabels([f'{(t+1)*0.5:.1f}s' for t in range(0, FT, 2)], fontsize=8)
        ax.set_title(f'GT Intent Soft Label\n(Agent {agent_idx})', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        col += 1

    # Model intent weights heatmap
    ax = axes[col]
    im = ax.imshow(intent_weights.T, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Future Timestep', fontsize=10)
    ax.set_ylabel('Intent Slot', fontsize=10)
    ax.set_yticks(range(K))
    ax.set_xticks(range(0, FT, 2))
    ax.set_xticklabels([f'{(t+1)*0.5:.1f}s' for t in range(0, FT, 2)], fontsize=8)
    ax.set_title(f'Model Intent Weights\n(Agent {agent_idx})', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    col += 1

    # Bar chart of slot usage
    ax = axes[col]
    selected_slots = intent_weights.argmax(axis=1)
    slot_counts = np.bincount(selected_slots, minlength=K)
    colors = plt.cm.Set3(np.linspace(0, 1, K))
    bars = ax.bar(range(K), slot_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Intent Slot', fontsize=10)
    ax.set_ylabel('Times Selected', fontsize=10)
    ax.set_xticks(range(K))
    ax.set_title(f'Slot Usage Distribution\n(Agent {agent_idx})', fontsize=10)

    # Add entropy annotation
    probs = slot_counts / max(slot_counts.sum(), 1)
    probs_nonzero = probs[probs > 0]
    entropy = -(probs_nonzero * np.log(probs_nonzero + 1e-8)).sum()
    max_entropy = np.log(K)
    ax.text(0.95, 0.95, f'Entropy: {entropy:.2f} / {max_entropy:.2f}\nUsed: {(slot_counts > 0).sum()}/{K}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved intent: {out_path}')


def visualize_intent_grid(gt_xy, prototypes, intent_weights, out_path,
                          agent_idx=0, intent_range=2.0,
                          map_obs=None, gt_traj_pix=None, pred_traj_pix=None):
    """
    4-panel: [Map+Traj] [GT intent] [colorbar] [Model intent]

    :param gt_xy: (FT, 2) — GT (acc_norm, yaw_rate_norm) per timestep
    :param prototypes: (K, 2) — prototype positions
    :param intent_weights: (FT, K) — model intent weights
    :param out_path: output path
    :param agent_idx: agent index
    :param intent_range: grid range [-r, r]
    :param map_obs: (C, H, W) map crop at t=0 (optional)
    :param gt_traj_pix: (FT, 2) GT trajectory in pixel coords (optional)
    :param pred_traj_pix: (FT, 2) pred trajectory in pixel coords (optional)
    """
    FT = gt_xy.shape[0]
    K = prototypes.shape[0]
    n_side = int(np.sqrt(K))
    colors = plt.cm.Reds(np.linspace(0.2, 1.0, FT))
    grid_vals = np.linspace(-intent_range, intent_range, n_side)

    has_map = map_obs is not None
    if has_map:
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1], wspace=0.3)
        ax_map = fig.add_subplot(gs[0])
        ax_gt = fig.add_subplot(gs[1])
        cbar_ax = fig.add_subplot(gs[2])
        ax_model = fig.add_subplot(gs[3])
    else:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.3)
        ax_gt = fig.add_subplot(gs[0])
        cbar_ax = fig.add_subplot(gs[1])
        ax_model = fig.add_subplot(gs[2])

    # --- Map + Traj panel ---
    if has_map:
        bg = render_map_bg(map_obs)
        ax_map.imshow(bg, origin='upper')
        if gt_traj_pix is not None:
            ax_map.plot(gt_traj_pix[:, 0], gt_traj_pix[:, 1], 'o-',
                        color='white', markersize=3, linewidth=1.5, label='GT')
        if pred_traj_pix is not None:
            ax_map.plot(pred_traj_pix[:, 0], pred_traj_pix[:, 1], 'x-',
                        color='red', markersize=3, linewidth=1.5, label='Pred')
        ax_map.legend(fontsize=8, loc='upper right')
        ax_map.set_title(f'Map + Trajectory (Agent {agent_idx})', fontsize=11)
        ax_map.axis('off')

    # prototypes[:, 0]=acc, [:, 1]=yaw → plot x=yaw, y=acc
    def _draw_base(ax):
        for v in grid_vals:
            ax.axhline(v, color='lightgray', linewidth=0.5)
            ax.axvline(v, color='lightgray', linewidth=0.5)
        ax.scatter(prototypes[:, 1], prototypes[:, 0], marker='s', s=140,
                   c='lightblue', edgecolors='navy', linewidth=1.5, zorder=2)
        lim = intent_range * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('Yaw Rate (normalized)', fontsize=10)
        ax.set_ylabel('Acc (normalized)', fontsize=10)
        ax.set_aspect('equal')
        # 코너 방향 라벨
        ax.text(-lim * 0.95, lim * 0.95, 'Accel+Right', fontsize=7, ha='left', va='top', color='gray')
        ax.text(lim * 0.95, lim * 0.95, 'Accel+Left', fontsize=7, ha='right', va='top', color='gray')
        ax.text(-lim * 0.95, -lim * 0.95, 'Decel+Right', fontsize=7, ha='left', va='bottom', color='gray')
        ax.text(lim * 0.95, -lim * 0.95, 'Decel+Left', fontsize=7, ha='right', va='bottom', color='gray')

    # --- GT panel --- (x=yaw, y=acc)
    _draw_base(ax_gt)
    for t in range(FT):
        ax_gt.scatter(gt_xy[t, 1], gt_xy[t, 0], c=[colors[t]], s=140, zorder=3,
                      edgecolors='black', linewidth=0.8, alpha=0.85)
    ax_gt.set_title(f'GT (Agent {agent_idx})', fontsize=11)

    # --- Model panel --- (x=yaw, y=acc)
    _draw_base(ax_model)
    model_xy = intent_weights @ prototypes  # (FT, 2) → [:, 0]=acc, [:, 1]=yaw
    for t in range(FT):
        ax_model.scatter(model_xy[t, 1], model_xy[t, 0], c=[colors[t]], s=140, zorder=3,
                         edgecolors='black', linewidth=0.8, alpha=0.85)
    ax_model.set_title(f'Model Intent (Agent {agent_idx})', fontsize=11)

    # --- Colorbar (between GT and Model) ---
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0.5, FT * 0.5))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_title('Time (s)', fontsize=8, pad=5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved intent grid: {out_path}')


def visualize_intent_aggregate(all_intent_weights, out_path, num_intents=9):
    """
    Aggregate intent codebook statistics across all scenes.

    :param all_intent_weights: list of (FT, K) arrays
    :param out_path: output path
    """
    # Stack all
    all_weights = np.concatenate(all_intent_weights, axis=0)  # (total_steps, K)
    K = num_intents

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: aggregate slot usage
    ax = axes[0]
    selected = all_weights.argmax(axis=1)
    slot_counts = np.bincount(selected, minlength=K)
    colors = plt.cm.Set3(np.linspace(0, 1, K))
    ax.bar(range(K), slot_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Intent Slot', fontsize=10)
    ax.set_ylabel('Total Selections', fontsize=10)
    ax.set_title(f'Aggregate Slot Usage ({len(all_intent_weights)} agents)', fontsize=10)
    ax.set_xticks(range(K))

    probs = slot_counts / max(slot_counts.sum(), 1)
    probs_nonzero = probs[probs > 0]
    entropy = -(probs_nonzero * np.log(probs_nonzero + 1e-8)).sum()
    ax.text(0.95, 0.95, f'Entropy: {entropy:.2f} / {np.log(K):.2f}\nUsed: {(slot_counts > 0).sum()}/{K}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: mean attention per slot over time
    ax = axes[1]
    # Group by timestep within each agent
    FT = all_intent_weights[0].shape[0]
    per_step = np.zeros((FT, K))
    count = 0
    for w in all_intent_weights:
        if w.shape[0] == FT:
            per_step += w
            count += 1
    if count > 0:
        per_step /= count

    im = ax.imshow(per_step.T, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0)
    ax.set_xlabel('Future Timestep', fontsize=10)
    ax.set_ylabel('Intent Slot', fontsize=10)
    ax.set_xticks(range(0, FT, 2))
    ax.set_xticklabels([f'{(t+1)*0.5:.1f}s' for t in range(0, FT, 2)], fontsize=8)
    ax.set_yticks(range(K))
    ax.set_title(f'Mean Intent Weights Over Time', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved aggregate intent: {out_path}')


def world_to_crop_pixel(pos_world, center_pos, bounds, L, W):
    """
    Convert world coordinates to crop pixel coordinates.

    :param pos_world: (N, 2) or (N, 4) world coords
    :param center_pos: (4,) center agent position (x, y, hx, hy)
    :param bounds: [low_l, low_w, high_l, high_w]
    :param L, W: pixel dimensions
    :returns: (N, 2) pixel coordinates
    """
    xy = pos_world[:, :2] if pos_world.shape[-1] > 2 else pos_world
    cx, cy = center_pos[0], center_pos[1]
    hx, hy = center_pos[2], center_pos[3]

    # Relative to center
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy

    # Rotate to ego frame
    rel_l = dx * hx + dy * hy
    rel_w = -dx * hy + dy * hx

    # Convert to pixel
    pix_l = (rel_l - bounds[0]) / (bounds[2] - bounds[0]) * L
    pix_w = (rel_w - bounds[1]) / (bounds[3] - bounds[1]) * W

    return np.stack([pix_l, pix_w], axis=1)


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

    # Create model
    model = TrafficPlannerModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix,
        len(dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        z_local_size=cfg.z_local_size,
        output_bicycle=cfg.model_output_bicycle,
        dt=cfg.dt,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
        num_intents=cfg.num_intents,
        sur_pred_dim=cfg.sur_pred_dim,
        map_recrop=cfg.map_recrop,
        trans_num_layers=cfg.trans_num_layers,
        trans_d_model=cfg.trans_d_model,
        trans_nhead=cfg.trans_nhead,
        trans_ffn_dim=cfg.trans_ffn_dim,
        trans_dropout=cfg.trans_dropout,
        use_ego_z_local=cfg.use_ego_z_local,
        use_sur_z_local=cfg.use_sur_z_local,
        use_a2a_rel_bias=getattr(cfg, 'use_a2a_rel_bias', False),
        num_z_queries=getattr(cfg, 'num_z_queries', 4),
        context_num_layers=getattr(cfg, 'context_num_layers', 2),
        map_summary_tokens=getattr(cfg, 'map_summary_tokens', 8),
    ).to(device)

    # Load checkpoint
    if cfg.ckpt is None:
        print('ERROR: Must provide --ckpt path')
        sys.exit(1)

    ckpt_epoch, _, _ = load_state(cfg.ckpt, model, map_location=device)
    print(f'Loaded checkpoint from epoch {ckpt_epoch}')

    # Set normalizers
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        # from datasets.utils import NUSC_BIKE_PARAMS
        # model.set_bicycle_params(NUSC_BIKE_PARAMS)
        from datasets.utils import CARLA_BIKE_PARAMS
        model.set_bicycle_params(CARLA_BIKE_PARAMS)

    state_normalizer = dataset.get_state_normalizer()
    att_normalizer = dataset.get_att_normalizer()

    # Output directory
    out_dir = os.path.join(cfg.out, f'viz_attn_intent_epoch{ckpt_epoch:05d}')
    mkdir(out_dir)
    print(f'Output directory: {out_dir}')

    # Run inference and visualize
    model.eval()
    all_ego_intent_weights = []
    scene_count = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            if scene_count >= cfg.num_scenes:
                break

            scene_graph, map_idx = data
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            NA = scene_graph.past.size(0)

            # Forward TF mode (GT tokens, GT map crops) → TF attention
            pred_tf = model.forward(scene_graph, map_idx, map_env,
                                    use_post_mean=True, teacher_forcing=True)
            tf_ego_map_attn_raw = model.get_ego_map_attn_weights()

            # Forward AR mode (autoregressive) → AR attention + predicted trajectory
            pred = model.reconstruct(scene_graph, map_idx, map_env)
            ego_map_attn = model.get_ego_map_attn_weights()
            intent_weights = model.get_intent_weights()

            if ego_map_attn is None:
                print(f'  Scene {i}: No map attn weights available, skipping')
                continue

            # Identify ego agents
            ego_mask = scene_graph.sem[:, 0] == 1  # first semantic class = ego
            if not ego_mask.any():
                # fallback: first agent is ego
                ego_mask = torch.zeros(NA, dtype=torch.bool)
                ego_mask[0] = True

            ego_indices = ego_mask.nonzero(as_tuple=True)[0]
            num_ego = ego_indices.numel()

            print(f'\nScene {i}: NA={NA}, num_ego={num_ego}')
            print(f'  ego_map_attn shape: {ego_map_attn.shape if isinstance(ego_map_attn, torch.Tensor) else "list"}')
            print(f'  intent_weights shape: {intent_weights.shape if isinstance(intent_weights, torch.Tensor) else "list"}')

            # Unnormalize for trajectory pixel conversion
            scene_graph_unnorm = normalize_scene_graph(
                scene_graph, state_normalizer, att_normalizer, unnorm=True
            )

            # Resolve TF attn: (T_total, N_ego, num_tokens) → slice future only
            PT = model.PT
            FT_model = model.FT
            tf_attn_tensor = None
            if isinstance(tf_ego_map_attn_raw, torch.Tensor):
                T_total = PT + FT_model
                print(f'  TF attn raw shape: {tf_ego_map_attn_raw.shape}')
                if tf_ego_map_attn_raw.dim() == 3 and tf_ego_map_attn_raw.size(0) == T_total:
                    # Already (T_total, N_ego, num_tokens)
                    tf_attn_tensor = tf_ego_map_attn_raw[PT-1:-1]  # (FT, N_ego, num_tokens)
                elif tf_ego_map_attn_raw.dim() == 2 and tf_ego_map_attn_raw.size(0) == T_total:
                    # (T_total, num_tokens) — single ego, add dim
                    tf_attn_tensor = tf_ego_map_attn_raw[PT-1:-1].unsqueeze(1)  # (FT, 1, num_tokens)
                else:
                    tf_attn_tensor = tf_ego_map_attn_raw

            # Resolve AR attn tensor shapes
            if isinstance(ego_map_attn, torch.Tensor):
                attn_tensor = ego_map_attn  # (FT, N_ego_avail, num_tokens)
            elif isinstance(ego_map_attn, list) and len(ego_map_attn) > 0:
                attn_tensor = torch.cat(ego_map_attn, dim=0)
            else:
                attn_tensor = None

            if isinstance(intent_weights, torch.Tensor):
                iw_tensor = intent_weights
            elif isinstance(intent_weights, list) and len(intent_weights) > 0:
                iw_tensor = torch.stack(intent_weights, dim=0) if intent_weights[0].dim() == 1 else torch.cat(intent_weights, dim=0)
            else:
                iw_tensor = None

            # Determine how many egos have attn data
            n_ego_avail = attn_tensor.shape[1] if (attn_tensor is not None and attn_tensor.dim() == 3) else 1

            for ego_local_idx in range(min(num_ego, n_ego_avail)):
                ego_global_idx = ego_indices[ego_local_idx].item()

                # Get ego's last position for map crop center
                ego_pos = scene_graph_unnorm.past_gt[ego_global_idx, -1, :4].cpu().numpy()
                gt_future = scene_graph_unnorm.future_gt[ego_global_idx].cpu().numpy()  # (FT, 4+)
                FT_len = gt_future.shape[0]
                mapix = map_idx[scene_graph.batch[ego_global_idx]].unsqueeze(0)

                # Get AR attention weights for this ego
                if attn_tensor is not None:
                    if attn_tensor.dim() == 3:
                        attn_np = attn_tensor[:, ego_local_idx, :].cpu().numpy()
                    elif attn_tensor.dim() == 2:
                        attn_np = attn_tensor.cpu().numpy()
                    else:
                        print(f'  Unexpected attn shape: {attn_tensor.shape}, skipping')
                        continue
                else:
                    print(f'  No attn data for ego {ego_local_idx}, skipping')
                    continue

                # Get TF attention weights for this ego
                tf_attn_np = None
                if tf_attn_tensor is not None:
                    if tf_attn_tensor.dim() == 3:
                        tf_attn_np = tf_attn_tensor[:, ego_local_idx, :].cpu().numpy()
                    elif tf_attn_tensor.dim() == 2:
                        tf_attn_np = tf_attn_tensor.cpu().numpy()

                # Build per-timestep map crops (map_recrop: each step uses its own crop)
                # Frame at t=0: past last position, t>0: gt_future[t-1]
                show_steps = [0, 3, 6, 9, 11]
                show_steps = [s for s in show_steps if s < FT_len]

                map_obs_dict = {}  # {timestep: (C, H, W)}
                # t=0 map crop (also used for trajectory overview)
                ego_pos_tensor_t0 = torch.tensor(ego_pos[:4], device=device).unsqueeze(0)
                map_obs_t0 = map_env.get_map_crop_pos(
                    ego_pos_tensor_t0, mapix
                ).cpu().numpy()[0]
                map_obs_dict['traj'] = map_obs_t0
                map_obs_dict[0] = map_obs_t0

                for t in show_steps:
                    if t == 0:
                        continue  # already computed
                    # Frame at t>0: gt_future[t-1] position
                    frame_pos = gt_future[t - 1, :4]
                    frame_tensor = torch.tensor(frame_pos, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                    map_obs_t = map_env.get_map_crop_pos(
                        frame_tensor, mapix
                    ).cpu().numpy()[0]
                    map_obs_dict[t] = map_obs_t

                # Compute GT soft label (each timestep in its own local frame)
                ego_frames_world = np.zeros((FT_len + 1, 4))
                ego_frames_world[0] = ego_pos[:4]
                ego_frames_world[1:] = gt_future[:, :4]

                num_tokens = attn_np.shape[1]
                gs = int(np.sqrt(num_tokens))
                _rf_s = getattr(model, 'map_rf_stride', None)
                _rf_o = getattr(model, 'map_rf_offset', None)
                soft_labels = compute_soft_label_grid(
                    gt_future, ego_frames_world, grid_size=gs,
                    bounds=cfg.map_obs_bounds, pix_size=cfg.map_obs_size_pix,
                    rf_stride=_rf_s, rf_offset=_rf_o
                )

                # Remap model attention from pred frame to GT frame
                pred_future = state_normalizer.unnormalize(
                    pred['future_pred'][ego_global_idx]
                ).cpu().numpy()  # (FT, 4)
                bounds = cfg.map_obs_bounds
                L = cfg.map_obs_size_pix
                grid_size = gs

                # Build pred frames: t=0 → past_last, t>0 → pred_future[t-1]
                pred_frames_world = np.zeros((FT_len + 1, 4))
                pred_frames_world[0] = ego_pos[:4]
                pred_frames_world[1:] = pred_future[:, :4]

                # Remap attn weights and compute pred pixel positions
                attn_remapped = attn_np.copy()  # (FT, 841)
                pred_pix_per_step = {}
                for t in range(FT_len):
                    gt_frame = ego_frames_world[t]
                    pred_frame = pred_frames_world[t]
                    # Remap attn grid: pred frame → GT frame
                    attn_grid = attn_np[t].reshape(grid_size, grid_size)
                    remapped = remap_attn_to_gt_frame(
                        attn_grid, pred_frame, gt_frame, grid_size, bounds,
                        pix_size=L, rf_stride=_rf_s, rf_offset=_rf_o)
                    attn_remapped[t] = remapped.reshape(-1)

                    # Pred position in GT-frame local → pixel
                    if t in show_steps:
                        dx = pred_future[t, 0] - gt_frame[0]
                        dy = pred_future[t, 1] - gt_frame[1]
                        local_l = dx * gt_frame[2] + dy * gt_frame[3]
                        local_w = -dx * gt_frame[3] + dy * gt_frame[2]
                        pix_x = (local_l - bounds[0]) / (bounds[2] - bounds[0]) * L
                        pix_y = (local_w - bounds[1]) / (bounds[3] - bounds[1]) * L
                        pred_pix_per_step[t] = (pix_x, pix_y)

                # Compute trajectory pixel coords on t=0 map crop (ego_pos frame)
                gt_traj_pix = world_to_crop_pixel(
                    gt_future[:, :2], ego_pos, bounds, L, L)
                pred_traj_pix = world_to_crop_pixel(
                    pred_future[:, :2], ego_pos, bounds, L, L)

                # Visualize: both rows use GT-position map crops for fair comparison
                map_attn_path = os.path.join(out_dir,
                    f'scene{i:04d}_ego{ego_local_idx}_map_attn.png')
                visualize_map_attention(
                    map_obs_dict, attn_remapped, map_attn_path,
                    agent_idx=ego_global_idx,
                    soft_labels=soft_labels,
                    tf_attn_weights=tf_attn_np,
                    pred_pix_per_step=pred_pix_per_step,
                    gt_traj_pix=gt_traj_pix,
                    pred_traj_pix=pred_traj_pix,
                    rf_stride=_rf_s,
                    rf_offset=_rf_o,
                    pix_size=cfg.map_obs_size_pix,
                    bounds=cfg.map_obs_bounds,
                )

                # Visualize intent codebook
                if iw_tensor is not None:
                    if iw_tensor.dim() == 3:
                        if ego_local_idx < iw_tensor.shape[1]:
                            iw_np = iw_tensor[:, ego_local_idx, :].cpu().numpy()
                        else:
                            continue
                    elif iw_tensor.dim() == 2:
                        iw_np = iw_tensor.cpu().numpy()
                    else:
                        continue

                    intent_path = os.path.join(out_dir,
                        f'scene{i:04d}_ego{ego_local_idx}_intent.png')
                    visualize_intent_codebook(
                        iw_np, intent_path,
                        agent_idx=ego_global_idx,
                        num_intents=cfg.num_intents
                    )
                    all_ego_intent_weights.append(iw_np)

            scene_count += 1

    # Aggregate intent statistics
    if len(all_ego_intent_weights) > 0:
        agg_path = os.path.join(out_dir, 'aggregate_intent.png')
        visualize_intent_aggregate(all_ego_intent_weights, agg_path,
                                   num_intents=cfg.num_intents)

    print(f'\nDone! {scene_count} scenes visualized → {out_dir}')


if __name__ == '__main__':
    main()
