#!/usr/bin/env python3
"""
z_aux 예측 vs GT 시각화 스크립트

Phase 1 학습된 모델의 z_aux_head가 z_global로부터
실제 trajectory를 얼마나 잘 예측하는지 시각화합니다.

Usage:
    python viz_z_aux.py -c configs/train_trafficplanner_phase1.cfg \
        --ckpt ./out/trafficplanner_phase1_out/checkpoints_trafficplanner/best_eval_model.pth \
        --num_samples 16
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets.nuscenes_utils import normalize_scene_graph
from models.trafficplanner_model import TrafficPlannerModel
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.config import get_parser, add_base_args
from utils.torch import load_state
from utils.transforms import transform2frame


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_cfg():
    parser = get_parser('Visualize z_aux predictions')
    parser = add_base_args(parser)
    parser.add_argument('--test_on_val', type=str2bool, default=True)
    parser.add_argument('--shuffle_test', type=str2bool, default=True)
    parser.add_argument('--seq_interval', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    # Model arch params
    parser.add_argument('--z_local_size', type=int, default=32)
    parser.add_argument('--num_intents', type=int, default=9)
    parser.add_argument('--sur_pred_dim', type=int, default=2)
    parser.add_argument('--map_recrop', type=str2bool, default=True)
    parser.add_argument('--trans_num_layers', type=int, default=2)
    parser.add_argument('--trans_d_model', type=int, default=128)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--trans_ffn_dim', type=int, default=256)
    parser.add_argument('--trans_dropout', type=float, default=0.1)
    parser.add_argument('--use_ego_z_local', type=str2bool, default=True)
    parser.add_argument('--use_sur_z_local', type=str2bool, default=True)

    # Viz params
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'Ignoring unknown args: {unknown[:5]}...')
    return dict2obj(vars(args)), vars(args)


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
                             shuffle=cfg.shuffle_test,
                             num_workers=0,
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
        from datasets.utils import CARLA_BIKE_PARAMS
        model.set_bicycle_params(CARLA_BIKE_PARAMS)

    state_norm = dataset.get_state_normalizer()

    # Output directory
    out_dir = cfg.out_dir or os.path.join(cfg.out, 'viz_z_aux')
    mkdir(out_dir)
    print(f'Output directory: {out_dir}')

    # Collect samples
    model.eval()
    all_gt_local = []
    all_pred_local = []
    all_z_samp = []
    all_z_mu = []
    all_z_var = []

    count = 0
    lscale = state_norm.std_vals[0].item()  # 15.0
    s_mean = state_norm.mean_vals[4].item()
    s_std = state_norm.std_vals[4].item()

    with torch.no_grad():
        for data in loader:
            if count >= cfg.num_samples:
                break

            scene_graph, map_idx = data
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)

            # Forward (Phase 1)
            pred = model(scene_graph, map_idx, map_env)

            # z_aux pred
            z_aux_pred = model._z_aux_pred  # (N_ego, FT, 5)
            if z_aux_pred is None:
                continue

            # GT in local frame
            ego_inds = scene_graph.ptr[:-1]
            ego_mask = torch.zeros(scene_graph.past.size(0), dtype=torch.bool, device=device)
            ego_mask[ego_inds] = True

            gt_future = scene_graph.future_gt[ego_mask]  # (N_ego, FT, 6) global normalized
            ego_ref = scene_graph.past[ego_inds, -1, :4]  # (N_ego, 4)
            local_kin = transform2frame(ego_ref, gt_future[:, :, :4])  # (N_ego, FT, 4)
            gt_local = torch.cat([local_kin, gt_future[:, :, 4:5]], dim=-1)  # (N_ego, FT, 5)

            # Posterior stats
            post_mu, post_var = model._last_posterior_out

            for i in range(z_aux_pred.size(0)):
                gt_xy = gt_local[i, :, :2].cpu().numpy() * lscale
                pred_xy = z_aux_pred[i, :, :2].cpu().numpy() * lscale
                gt_head = gt_local[i, :, 2:4].cpu().numpy()
                pred_head = z_aux_pred[i, :, 2:4].cpu().numpy()
                gt_speed = gt_local[i, :, 4].cpu().numpy()
                pred_speed = z_aux_pred[i, :, 4].cpu().numpy()

                z = model._last_z_samp[ego_inds[i]].cpu().numpy()
                mu = post_mu[ego_inds[i]].cpu().numpy()
                var = post_var[ego_inds[i]].cpu().numpy()

                all_gt_local.append({'xy': gt_xy, 'head': gt_head, 'speed': gt_speed})
                all_pred_local.append({'xy': pred_xy, 'head': pred_head, 'speed': pred_speed})
                all_z_samp.append(z)
                all_z_mu.append(mu)
                all_z_var.append(var)
                count += 1
                if count >= cfg.num_samples:
                    break

    N = len(all_gt_local)
    print(f'Collected {N} samples')

    # ========================================
    # Plot 1: Trajectory comparison (x,y)
    # ========================================
    cols = 4
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    all_ade = []
    all_fde = []
    for i in range(N):
        ax = axes[i // cols, i % cols]
        gt_xy = all_gt_local[i]['xy']
        pred_xy = all_pred_local[i]['xy']

        ax.plot(0, 0, 'ko', markersize=8, label='ego')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'b.-', linewidth=2, markersize=4, label='GT')
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'r.--', linewidth=2, markersize=4, label='z_aux pred')

        ade = np.sqrt(((gt_xy - pred_xy) ** 2).sum(axis=1)).mean()
        fde = np.sqrt(((gt_xy[-1] - pred_xy[-1]) ** 2).sum())
        all_ade.append(ade)
        all_fde.append(fde)
        ax.set_title(f'#{i} ADE={ade:.2f}m FDE={fde:.2f}m', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for i in range(N, rows * cols):
        axes[i // cols, i % cols].axis('off')

    fig.suptitle(f'z_aux Trajectory (local frame, meters) | mean ADE={np.mean(all_ade):.2f}m FDE={np.mean(all_fde):.2f}m', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'z_aux_trajectory.png'), dpi=150)
    plt.close(fig)
    print(f'Saved: z_aux_trajectory.png')

    # ========================================
    # Plot 2: Speed comparison
    # ========================================
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(N):
        ax = axes[i // cols, i % cols]
        gt_s = all_gt_local[i]['speed'] * s_std + s_mean
        pred_s = all_pred_local[i]['speed'] * s_std + s_mean
        t = np.arange(1, len(gt_s) + 1) * 0.5

        ax.plot(t, gt_s, 'b.-', linewidth=2, label='GT')
        ax.plot(t, pred_s, 'r.--', linewidth=2, label='z_aux pred')
        ax.set_xlabel('t (s)', fontsize=8)
        ax.set_ylabel('speed (m/s)', fontsize=8)
        ax.set_title(f'#{i}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for i in range(N, rows * cols):
        axes[i // cols, i % cols].axis('off')

    fig.suptitle('z_aux Speed Prediction', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'z_aux_speed.png'), dpi=150)
    plt.close(fig)
    print(f'Saved: z_aux_speed.png')

    # ========================================
    # Plot 3: Heading comparison
    # ========================================
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(N):
        ax = axes[i // cols, i % cols]
        gt_h = np.degrees(np.arctan2(all_gt_local[i]['head'][:, 1], all_gt_local[i]['head'][:, 0]))
        pred_h = np.degrees(np.arctan2(all_pred_local[i]['head'][:, 1], all_pred_local[i]['head'][:, 0]))
        t = np.arange(1, len(gt_h) + 1) * 0.5

        ax.plot(t, gt_h, 'b.-', linewidth=2, label='GT')
        ax.plot(t, pred_h, 'r.--', linewidth=2, label='z_aux pred')
        ax.set_xlabel('t (s)', fontsize=8)
        ax.set_ylabel('heading (deg)', fontsize=8)
        ax.set_title(f'#{i}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for i in range(N, rows * cols):
        axes[i // cols, i % cols].axis('off')

    fig.suptitle('z_aux Heading Prediction (local frame)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'z_aux_heading.png'), dpi=150)
    plt.close(fig)
    print(f'Saved: z_aux_heading.png')

    # ========================================
    # Plot 4: z_global distribution
    # ========================================
    z_matrix = np.stack(all_z_samp, axis=0)  # (N, 32)
    mu_matrix = np.stack(all_z_mu, axis=0)
    var_matrix = np.stack(all_z_var, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # z_samp heatmap
    im = axes[0, 0].imshow(z_matrix, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    axes[0, 0].set_xlabel('z dim')
    axes[0, 0].set_ylabel('sample')
    axes[0, 0].set_title('z_samp per sample')
    plt.colorbar(im, ax=axes[0, 0])

    # Per-dim std of z_samp across samples
    per_dim_std = z_matrix.std(axis=0)
    axes[0, 1].bar(range(len(per_dim_std)), per_dim_std, color='steelblue')
    axes[0, 1].set_xlabel('z dim')
    axes[0, 1].set_ylabel('std across samples')
    axes[0, 1].set_title(f'z_samp per-dim std (mean={per_dim_std.mean():.3f})')
    axes[0, 1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='threshold=0.1')
    axes[0, 1].legend(fontsize=8)

    # Posterior variance (mean across samples per dim)
    mean_var = var_matrix.mean(axis=0)
    axes[1, 0].bar(range(len(mean_var)), mean_var, color='coral')
    axes[1, 0].set_xlabel('z dim')
    axes[1, 0].set_ylabel('posterior var')
    axes[1, 0].set_title(f'Posterior var per dim (mean={mean_var.mean():.4f})')

    # Per-dim mean
    per_dim_mean = z_matrix.mean(axis=0)
    axes[1, 1].bar(range(len(per_dim_mean)), per_dim_mean, color='seagreen')
    axes[1, 1].set_xlabel('z dim')
    axes[1, 1].set_ylabel('mean across samples')
    axes[1, 1].set_title(f'z_samp per-dim mean (norm={np.linalg.norm(per_dim_mean):.2f})')

    fig.suptitle(f'z_global Distribution ({N} samples)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'z_aux_z_distribution.png'), dpi=150)
    plt.close(fig)
    print(f'Saved: z_aux_z_distribution.png')

    # ========================================
    # Summary
    # ========================================
    print(f'\n=== Summary ({N} samples) ===')
    print(f'ADE: mean={np.mean(all_ade):.3f}m, std={np.std(all_ade):.3f}m')
    print(f'FDE: mean={np.mean(all_fde):.3f}m, std={np.std(all_fde):.3f}m')
    print(f'z_samp per-dim std: mean={per_dim_std.mean():.4f}, min={per_dim_std.min():.4f}, max={per_dim_std.max():.4f}')
    print(f'posterior var: mean={mean_var.mean():.4f}')
    print(f'z_samp norm: mean={np.linalg.norm(z_matrix, axis=1).mean():.3f}')
    dead_dims = (per_dim_std < 0.01).sum()
    print(f'Dead dims (std<0.01): {dead_dims}/32')


if __name__ == '__main__':
    main()
