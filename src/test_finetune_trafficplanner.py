# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Test script for fine-tuned TrafficPlannerModel
# Uses FinetuneTrafficPlannerDataset with data_types selection (normal / adv_sol / both)

'''
Runs through fine-tuning dataset and runs various evaluations on the TrafficPlanner model.
Supports data_types filtering: 'both', 'normal', 'adv_sol'
'''

import os, shutil, csv
import time
import tqdm
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets import nuscenes_utils as nutils
from models.trafficplanner_model import TrafficPlannerModel
from losses.trafficplanner_loss import TrafficPlannerLoss, compute_disp_err, compute_coll_rate_env, compute_coll_rate_veh
from datasets.finetune_trafficplanner_dataset import FinetuneTrafficPlannerDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, load_state
from utils.config import get_parser, add_base_args
import argparse


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
    parser = get_parser('Test Fine-tuned TrafficPlanner Model')
    parser = add_base_args(parser)

    # Fine-tuning data
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Path to adv_sol_success/ JSON directory')
    parser.add_argument('--data_types', type=str, default='both',
                        choices=['both', 'normal', 'adv_sol'],
                        help='Data types to test: both, normal, or adv_sol')

    # Test split
    parser.add_argument('--test_split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Which split to test on')
    parser.add_argument('--shuffle_test', type=str2bool, default=False)

    # TrafficPlannerModel architecture
    parser.add_argument('--z_local_size', type=int, default=32)
    parser.add_argument('--num_intents', type=int, default=9,
                        help='Number of intent codebook entries (K)')
    parser.add_argument('--sur_pred_dim', type=int, default=2,
                        help='Predicted surrounding agent delta dimension (dx, dy)')
    parser.add_argument('--map_recrop', type=str2bool, default=True,
                        help='Re-crop map tokens at each decode step')

    # Transformer decoder parameters
    parser.add_argument('--trans_num_layers', type=int, default=4)
    parser.add_argument('--trans_d_model', type=int, default=128)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--trans_ffn_dim', type=int, default=512)
    parser.add_argument('--trans_dropout', type=float, default=0.1)
    parser.add_argument('--use_ego_z_local', type=str2bool, default=True)
    parser.add_argument('--use_sur_z_local', type=str2bool, default=False)

    # Potential field parameters (for config compatibility)
    parser.add_argument('--k_veh_repel', type=float, default=1.0)
    parser.add_argument('--sigma_veh', type=float, default=1.5)
    parser.add_argument('--k_env_boundary', type=float, default=1.0)
    parser.add_argument('--sigma_env_boundary', type=float, default=0.5)
    parser.add_argument('--k_env_solid', type=float, default=0.8)
    parser.add_argument('--sigma_env_solid', type=float, default=0.5)
    parser.add_argument('--k_env_dashed', type=float, default=0.1)
    parser.add_argument('--sigma_env_dashed', type=float, default=0.3)
    parser.add_argument('--use_lane_lines', type=str2bool, default=True)

    # Device
    parser.add_argument('--gpu', type=int, default=0)

    # Visualization suffix
    parser.add_argument('--viz_suffix', type=str, default='default')

    # Test options - reconstruct (posterior)
    parser.add_argument('--test_recon_viz_multi', type=str2bool, default=False)
    parser.add_argument('--test_recon_coll_rate', type=str2bool, default=False)

    # Test options - sample (prior)
    parser.add_argument('--test_sample_viz_multi', type=str2bool, default=False)
    parser.add_argument('--test_sample_viz_rollout', type=str2bool, default=False)
    parser.add_argument('--test_sample_disp_err', type=str2bool, default=False)
    parser.add_argument('--test_sample_coll_rate', type=str2bool, default=False)
    parser.add_argument('--test_sample_num', type=int, default=3)
    parser.add_argument('--test_sample_future_len', type=int, default=None)

    # Latent analysis
    parser.add_argument('--test_latent_analysis', type=str2bool, default=False)

    # Video
    parser.add_argument('--make_video', type=str2bool, default=False)

    args = parser.parse_args()
    config_dict = vars(args)
    config = dict2obj(config_dict)
    return config, config_dict


def save_latent_analysis(scene_dir, z_global_mean_scene, z_global_var_scene,
                         ego_idx_in_scene, z_local_mean_scene, z_local_var_scene,
                         attn_weights_scene):
    """
    Save CSV and PNG latent analysis for a single scene.

    :param scene_dir: output directory for this scene (e.g. latent_analysis/scene_0000/)
    :param z_global_mean_scene: (num_agents, 32) z_global mean for all agents in scene
    :param z_global_var_scene: (num_agents, 32) z_global var for all agents in scene
    :param ego_idx_in_scene: index of ego agent within the scene (always 0)
    :param z_local_mean_scene: (FT, 32) z_local mean for ego across timesteps
    :param z_local_var_scene: (FT, 32) z_local var for ego across timesteps
    :param attn_weights_scene: list of FT numpy arrays, each (Q_len, KV_len) — head-averaged
    """
    os.makedirs(scene_dir, exist_ok=True)
    z_global_dim = z_global_mean_scene.shape[1]
    z_local_dim = z_local_mean_scene.shape[1]
    num_agents = z_global_mean_scene.shape[0]
    FT = z_local_mean_scene.shape[0]

    # ========== CSV: z_global ==========
    with open(os.path.join(scene_dir, 'z_global.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['agent_idx', 'agent_type']
        header += [f'mean_{d}' for d in range(z_global_dim)]
        header += [f'var_{d}' for d in range(z_global_dim)]
        writer.writerow(header)
        for a in range(num_agents):
            atype = 'ego' if a == ego_idx_in_scene else 'sur'
            row = [a, atype]
            row += z_global_mean_scene[a].tolist()
            row += z_global_var_scene[a].tolist()
            writer.writerow(row)

    # ========== CSV: z_local ==========
    with open(os.path.join(scene_dir, 'z_local.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['timestep']
        header += [f'mean_{d}' for d in range(z_local_dim)]
        if z_local_var_scene is not None:
            header += [f'var_{d}' for d in range(z_local_dim)]
        writer.writerow(header)
        for t in range(FT):
            row = [t]
            row += z_local_mean_scene[t].tolist()
            if z_local_var_scene is not None:
                row += z_local_var_scene[t].tolist()
            writer.writerow(row)

    # ========== CSV: attn_weights ==========
    # Each element: (Q_len, KV_len) head-averaged attention
    with open(os.path.join(scene_dir, 'attn_weights.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        kv_len = attn_weights_scene[0].shape[-1] if len(attn_weights_scene) > 0 else 0
        window_size_csv = 4  # z_local_window
        num_sur_csv = kv_len // window_size_csv
        header = ['timestep']
        if num_sur_csv * window_size_csv == kv_len and num_sur_csv > 0:
            for a in range(num_sur_csv):
                for w in range(window_size_csv):
                    header.append(f's{a}_w{w}')
        else:
            for kv in range(kv_len):
                header.append(f'kv{kv}')
        writer.writerow(header)
        for t in range(len(attn_weights_scene)):
            aw = attn_weights_scene[t]  # (Q_len, KV_len)
            # Average over Q_len dimension
            aw_avg = aw.mean(axis=0)  # (KV_len,)
            row = [t]
            row += aw_avg.tolist()
            writer.writerow(row)

    # ========== PNG: z_global_dist ==========
    fig, ax = plt.subplots(figsize=(14, 5))
    ego_mean = z_global_mean_scene[ego_idx_in_scene]
    x = np.arange(z_global_dim)
    width = 0.35
    ax.bar(x - width/2, ego_mean, width, label='ego', color='tab:blue', alpha=0.8)
    if num_agents > 1:
        sur_means = z_global_mean_scene[[a for a in range(num_agents) if a != ego_idx_in_scene]]
        sur_mean_avg = sur_means.mean(axis=0)
        ax.bar(x + width/2, sur_mean_avg, width, label='sur (avg)', color='tab:orange', alpha=0.8)
    ax.set_xlabel('z_global dimension')
    ax.set_ylabel('mean value')
    ax.set_title('z_global distribution (ego vs sur)')
    ax.legend()
    ax.set_xticks(x[::4])
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, 'z_global_dist.png'), dpi=100)
    plt.close()

    # ========== PNG: z_local_over_time ==========
    # Show top 8 dimensions by variance across time
    z_local_np = z_local_mean_scene  # (FT, 32)
    dim_variance = np.var(z_local_np, axis=0)
    top_dims = np.argsort(dim_variance)[-8:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for d in top_dims:
        ax.plot(range(FT), z_local_np[:, d], marker='o', markersize=3, label=f'dim {d}')
    ax.set_xlabel('timestep')
    ax.set_ylabel('z_local mean')
    ax.set_title('z_local mean over time (top 8 varying dims)')
    ax.legend(fontsize=7)
    ax.set_xticks(range(FT))
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, 'z_local_over_time.png'), dpi=100)
    plt.close()

    # ========== PNG: z_local_var_over_time ==========
    if z_local_var_scene is not None:
        z_local_var_np = z_local_var_scene  # (FT, 32)
        fig, ax = plt.subplots(figsize=(10, 5))
        for d in top_dims:
            ax.plot(range(FT), z_local_var_np[:, d], marker='o', markersize=3, label=f'dim {d}')
        ax.set_xlabel('timestep')
        ax.set_ylabel('z_local var')
        ax.set_title('z_local variance over time (top 8 varying dims)')
        ax.legend(fontsize=7)
        ax.set_xticks(range(FT))
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, 'z_local_var_over_time.png'), dpi=100)
        plt.close()

    # ========== PNG: z_local_magnitude ==========
    z_local_l2 = np.linalg.norm(z_local_np, axis=1)  # (FT,)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(FT), z_local_l2, marker='o', color='tab:red', linewidth=2)
    ax.set_xlabel('timestep')
    ax.set_ylabel('L2 norm')
    ax.set_title('z_local L2 magnitude over time')
    ax.set_xticks(range(FT))
    plt.tight_layout()
    plt.savefig(os.path.join(scene_dir, 'z_local_magnitude.png'), dpi=100)
    plt.close()

    # ========== PNG: attn_weights_heatmap ==========
    # Head-averaged attention: single heatmap (timestep x KV_index)
    if len(attn_weights_scene) > 0:
        kv_len = attn_weights_scene[0].shape[-1]
        heatmap_data = np.zeros((len(attn_weights_scene), kv_len))
        for t in range(len(attn_weights_scene)):
            aw = attn_weights_scene[t]  # (Q_len, KV_len)
            heatmap_data[t] = aw.mean(axis=0)  # avg over Q_len -> (KV_len,)

        # Build meaningful x-axis labels: sur{agent}_w{window}
        window_size = 4  # z_local_window
        num_sur = kv_len // window_size
        if num_sur * window_size == kv_len and num_sur > 0:
            xlabels = [f's{a}_w{w}' for a in range(num_sur) for w in range(window_size)]
        else:
            xlabels = [f'kv{i}' for i in range(kv_len)]

        num_t = len(attn_weights_scene)
        fig, ax = plt.subplots(figsize=(max(kv_len * 1.2, 6), 8))
        im = ax.pcolormesh(heatmap_data, cmap='hot', edgecolors='none', linewidth=0)
        ax.set_xlabel('KV index (sur_agent x window)')
        ax.set_ylabel('timestep')
        ax.set_title(f'Cross-attention weights (head-averaged, {num_sur} sur agent(s))')
        ax.set_xticks([i + 0.5 for i in range(kv_len)])
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(num_t)])
        ax.set_yticklabels([str(i) for i in range(num_t)])
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(scene_dir, 'attn_weights_heatmap.png'), dpi=100)
        plt.close()


def save_latent_pca_tsne(out_path, all_z_global, all_z_global_labels,
                         all_z_local):
    """
    Save PCA and t-SNE visualizations for z_global and z_local.
    Called in finally block so it works even on Ctrl+C.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if len(all_z_global) == 0 and len(all_z_local) == 0:
        print('[latent PCA/t-SNE] No data collected, skipping.')
        return

    # ========== z_global PCA / t-SNE ==========
    if len(all_z_global) >= 2:
        zg = np.array(all_z_global)       # (N, 32)
        labels = all_z_global_labels       # list of 'ego' or 'sur'
        is_ego = np.array([l == 'ego' for l in labels])

        for method_name, Reducer in [('pca', PCA), ('tsne', TSNE)]:
            if method_name == 'tsne' and len(zg) < 5:
                continue
            if method_name == 'pca':
                reducer = Reducer(n_components=2)
            else:
                perp = min(30, len(zg) - 1)
                reducer = Reducer(n_components=2, perplexity=perp, random_state=42)
            coords = reducer.fit_transform(zg)  # (N, 2)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(coords[is_ego, 0], coords[is_ego, 1],
                       c='tab:blue', label='ego', alpha=0.7, s=40)
            ax.scatter(coords[~is_ego, 0], coords[~is_ego, 1],
                       c='tab:orange', label='sur', alpha=0.7, s=40)
            ax.set_title(f'z_global {method_name.upper()} (n={len(zg)})')
            ax.set_xlabel(f'{method_name.upper()} dim 0')
            ax.set_ylabel(f'{method_name.upper()} dim 1')
            ax.legend()
            plt.tight_layout()
            save_path = os.path.join(out_path, f'z_global_{method_name}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f'[latent PCA/t-SNE] saved {save_path}')

    # ========== z_local PCA / t-SNE ==========
    if len(all_z_local) >= 2:
        zl = np.array([d['mean'] for d in all_z_local])         # (N, 32)
        timesteps = np.array([d['timestep'] for d in all_z_local])  # (N,)
        scene_ids = np.array([d['scene_idx'] for d in all_z_local])  # (N,)

        for method_name, Reducer in [('pca', PCA), ('tsne', TSNE)]:
            if method_name == 'tsne' and len(zl) < 5:
                continue
            if method_name == 'pca':
                reducer = Reducer(n_components=2)
            else:
                perp = min(30, len(zl) - 1)
                reducer = Reducer(n_components=2, perplexity=perp, random_state=42)
            coords = reducer.fit_transform(zl)  # (N, 2)

            # Color by timestep
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=timesteps, cmap='viridis', alpha=0.7, s=20)
            ax.set_title(f'z_local {method_name.upper()} colored by timestep (n={len(zl)})')
            ax.set_xlabel(f'{method_name.upper()} dim 0')
            ax.set_ylabel(f'{method_name.upper()} dim 1')
            plt.colorbar(sc, ax=ax, label='timestep')
            plt.tight_layout()
            save_path = os.path.join(out_path, f'z_local_{method_name}_by_timestep.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f'[latent PCA/t-SNE] saved {save_path}')

            # Color by scene
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=scene_ids, cmap='tab20', alpha=0.7, s=20)
            ax.set_title(f'z_local {method_name.upper()} colored by scene (n={len(zl)})')
            ax.set_xlabel(f'{method_name.upper()} dim 0')
            ax.set_ylabel(f'{method_name.upper()} dim 1')
            plt.colorbar(sc, ax=ax, label='scene index')
            plt.tight_layout()
            save_path = os.path.join(out_path, f'z_local_{method_name}_by_scene.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f'[latent PCA/t-SNE] saved {save_path}')


def run_one_epoch(data_loader, model, map_env, loss_fn, device, out_path,
                  viz_suffix='default',
                  test_recon_viz_multi=False,
                  test_recon_coll_rate=False,
                  test_sample_viz_multi=False,
                  test_sample_viz_rollout=False,
                  test_sample_disp_err=False,
                  test_sample_coll_rate=False,
                  test_sample_num=3,
                  test_sample_future_len=None,
                  make_video=False,
                  sur_gt_replay=False,
                  test_latent_analysis=False,
                  latent_accum=None
                  ):
    '''
    Run through test dataset and perform various desired evaluations.
    '''
    pbar = tqdm.tqdm(data_loader)

    if test_recon_viz_multi:
        recon_multi_agent_out_path = os.path.join(out_path, f'viz_recon_multi_{viz_suffix}')
        mkdir(recon_multi_agent_out_path)
    if test_sample_viz_multi:
        sample_multi_agent_out_path = os.path.join(out_path, f'viz_sample_multi_{viz_suffix}')
        mkdir(sample_multi_agent_out_path)
    if test_sample_viz_rollout:
        sample_rollout_out_path = os.path.join(out_path, f'viz_sample_rollout_{viz_suffix}')
        mkdir(sample_rollout_out_path)
    if test_latent_analysis:
        latent_analysis_path = os.path.join(out_path, 'latent_analysis')
        mkdir(latent_analysis_path)

    metrics = {}
    freq_metrics = {}
    data_idx = 0

    # Use external accumulators if provided (for Ctrl+C survival via try/finally in main)
    if latent_accum is not None:
        all_z_global = latent_accum['z_global']
        all_z_global_labels = latent_accum['z_global_labels']
        all_z_local = latent_accum['z_local']
    else:
        all_z_global = []
        all_z_global_labels = []
        all_z_local = []

    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        B = map_idx.size(0)

        pred = model(scene_graph, map_idx, map_env, use_post_mean=True, teacher_forcing=False)
        loss_dict = loss_fn(scene_graph, pred, use_teacher_forcing=False)
        loss = loss_dict['loss'][0]

        err_dict = loss_fn.compute_err(scene_graph, pred, model.get_normalizer())

        batch_metrics = {**loss_dict, **err_dict}
        batch_freq_metrics = {}

        #
        # Reconstruction-based Evaluations
        #
        recon_pred = None
        if test_recon_viz_multi or test_recon_coll_rate or test_latent_analysis:
            recon_pred = model.reconstruct(scene_graph, map_idx, map_env, sur_gt_replay=sur_gt_replay)

        if test_recon_viz_multi:
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_recon_multi_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(recon_multi_agent_out_path, pred_prefix)
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=recon_pred['future_pred'],
                                            viz_traj=True,
                                            make_video=make_video,
                                            viz_bounds=[-50.0, -50.0, 50.0, 50.0],
                                            show_gt=True)

        if test_recon_coll_rate:
            coll_pred = {k : v for k, v in recon_pred.items()}
            coll_pred['future_pred'] = coll_pred['future_pred'].unsqueeze(1)

            coll_rate_dict = compute_coll_rate_env(scene_graph, map_idx, coll_pred, map_env,
                                                       model.get_normalizer(), model.get_att_normalizer(),
                                                       ego_only=True)
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            coll_rate_dict = compute_coll_rate_veh(scene_graph, coll_pred,
                                                       model.get_normalizer(), model.get_att_normalizer())
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

        #
        # Latent Analysis
        #
        if test_latent_analysis and recon_pred is not None:
            z_global_mean, z_global_var = recon_pred['prior_out']
            # z_global_mean, z_global_var: (NA, 32) flattened across all scenes (prior = past only)
            z_local_mean = model.get_z_local_mean()  # (FT, num_ego, 32) — deterministic value
            z_local_var = model.get_z_local_var()     # None (deterministic z_local)
            attn_weights = model.get_attn_weights()   # list of tensors

            # Convert to numpy
            z_global_mean_np = z_global_mean.cpu().numpy()
            z_global_var_np = z_global_var.cpu().numpy()
            z_local_mean_np = z_local_mean.cpu().numpy() if z_local_mean is not None else None
            z_local_var_np = None  # deterministic z_local has no variance

            # Process attention weights: list of (1, num_heads, Q_len, KV_len) -> numpy
            attn_weights_np = None
            if attn_weights is not None:
                attn_weights_np = [aw.squeeze(0).cpu().numpy() for aw in attn_weights]
                # Each: (Q_len, KV_len) — head-averaged (PyTorch 1.9)

            # Split by scene using scene_graph.ptr
            ego_counter = 0  # tracks ego index across batch
            for bidx in range(B):
                scene_start = scene_graph.ptr[bidx].item()
                scene_end = scene_graph.ptr[bidx + 1].item()
                num_agents_in_scene = scene_end - scene_start

                scene_idx = data_idx + bidx
                scene_dir = os.path.join(latent_analysis_path, f'scene_{scene_idx:04d}')

                # z_global for this scene
                zg_mean = z_global_mean_np[scene_start:scene_end]  # (num_agents, 32)
                zg_var = z_global_var_np[scene_start:scene_end]

                # z_local for this scene's ego (ego_counter-th ego in batch)
                zl_mean = z_local_mean_np[:, ego_counter, :] if z_local_mean_np is not None else np.zeros((12, 32))
                zl_var = None  # deterministic z_local

                # attn_weights: FT timesteps per ego
                # In single-sample reconstruct, _compute_z_local_dist is called FT times
                # Each call loops over B scenes, appending B weights
                # So total attn_weights = FT * B, and for scene bidx at timestep t: index = t * B + bidx
                scene_attn = []
                if attn_weights_np is not None:
                    num_attn = len(attn_weights_np)
                    FT_model = num_attn // B if B > 0 else 0
                    for t in range(FT_model):
                        idx = t * B + bidx
                        if idx < num_attn:
                            scene_attn.append(attn_weights_np[idx])

                save_latent_analysis(
                    scene_dir, zg_mean, zg_var,
                    ego_idx_in_scene=0,  # ego is always first in scene
                    z_local_mean_scene=zl_mean,
                    z_local_var_scene=zl_var,
                    attn_weights_scene=scene_attn
                )

                # Accumulate for PCA/t-SNE
                for a in range(num_agents_in_scene):
                    all_z_global.append(zg_mean[a])
                    all_z_global_labels.append('ego' if a == 0 else 'sur')
                FT = zl_mean.shape[0]
                for t in range(FT):
                    all_z_local.append({
                        'mean': zl_mean[t],
                        'timestep': t,
                        'scene_idx': scene_idx
                    })

                ego_counter += 1

        #
        # Sampling-based Evaluations
        #
        sample_pred = None
        if test_sample_disp_err or test_sample_viz_multi or test_sample_coll_rate or test_sample_viz_rollout:
            sample_pred = model.sample_batched(scene_graph, map_idx, map_env, test_sample_num,
                                        include_mean=False, nfuture=test_sample_future_len,
                                        sur_gt_replay=sur_gt_replay)

        if test_sample_viz_multi:
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_sample_multi_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(sample_multi_agent_out_path, pred_prefix)
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=sample_pred['future_pred'],
                                            viz_traj=True,
                                            make_video=make_video,
                                            show_gt=False)

        if test_sample_viz_rollout:
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx
                pred_prefix = 'test_sample_rollout_%08d_pred' % (multi_agt_data_idx)
                pred_out_path = os.path.join(sample_rollout_out_path, pred_prefix)
                nutils.viz_scene_graph(scene_graph, map_idx, map_env, bidx, pred_out_path,
                                            model.get_normalizer(), model.get_att_normalizer(),
                                            future_pred=sample_pred['future_pred'],
                                            viz_traj=False,
                                            make_video=make_video,
                                            viz_bounds=[-50.0, -50.0, 50.0, 50.0],
                                            center_viz=True)

        if test_sample_disp_err:
            disp_err_dict = compute_disp_err(scene_graph, sample_pred, model.get_normalizer())
            batch_metrics = {**batch_metrics, **disp_err_dict}

        if test_sample_coll_rate:
            coll_rate_dict = compute_coll_rate_env(scene_graph, map_idx, sample_pred, map_env,
                                                       model.get_normalizer(), model.get_att_normalizer(),
                                                       ego_only=True)
            coll_rate_dict = {'sample_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            coll_rate_dict = compute_coll_rate_veh(scene_graph, sample_pred,
                                                       model.get_normalizer(), model.get_att_normalizer())
            coll_rate_dict = {'sample_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

        data_idx += B

        progress_bar_metrics = {}
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
            progress_bar_metrics[k] = torch.mean(v).item()

        freq_prefixes = ['recon_', 'sample_']
        freq_postfixes = ['_map', '_veh']
        freq_expts = [test_recon_coll_rate, test_sample_coll_rate]
        used_prefixes = [pref for pid, pref in enumerate(freq_prefixes) if freq_expts[pid]]
        used_postfixes = freq_postfixes
        for freq_pref in used_prefixes:
            for freq_post in used_postfixes:
                num_coll = freq_pref + 'num_coll' + freq_post
                num_traj = freq_pref + 'num_traj' + freq_post
                if num_coll not in freq_metrics:
                    freq_metrics[num_coll] = 0
                    freq_metrics[num_traj] = 0
                freq_metrics[num_coll] += float(batch_freq_metrics[num_coll])
                freq_metrics[num_traj] += float(batch_freq_metrics[num_traj])
                progress_bar_metrics[freq_pref + 'coll_freq' + freq_post] = float(batch_freq_metrics[num_coll]) / batch_freq_metrics[num_traj]

        pbar.set_postfix(progress_bar_metrics)

    epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = torch.cat(metrics[k])
        epoch_metrics["Test Mean " + k] = torch.mean(metrics[k]).item()

    for freq_pref in used_prefixes:
        for freq_post in used_postfixes:
            num_coll = freq_pref + 'num_coll' + freq_post
            num_traj = freq_pref + 'num_traj' + freq_post
            test_coll_freq = freq_metrics[num_coll] / freq_metrics[num_traj]
            epoch_metrics["Test (%s, %s) Collision Freq" % (freq_pref, freq_post)] = test_coll_freq

    Logger.log('Final ===================================== ')
    if len(epoch_metrics) > 0:
        for k, v in epoch_metrics.items():
            Logger.log('%s = %f' % (k, v))



def main():
    cfg, cfg_dict = parse_cfg()

    # Auto-set sur_gt_replay based on data_types (adv_sol → True, otherwise False)
    if cfg.data_types == 'adv_sol':
        cfg.sur_gt_replay = True
    else:
        cfg.sur_gt_replay = False

    print('=== Test Fine-tuned TrafficPlanner ===')
    print(f'ckpt: {cfg.ckpt}')
    print(f'scenario_dir: {cfg.scenario_dir}')
    print(f'data_types: {cfg.data_types}')
    print(f'test_split: {cfg.test_split}')
    print(f'sur_gt_replay: {cfg.sur_gt_replay} (auto-set from data_types)')

    base_out = cfg.out
    mkdir(base_out)
    log_path = os.path.join(base_out, 'test_log.txt')
    Logger.init(log_path)
    Logger.log('Args: ' + str(cfg_dict))

    # Device
    device = f'cuda:{cfg.gpu}'
    Logger.log('Using device %s...' % (str(device)))

    # Map environment
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(data_path,
                        bounds=cfg.map_obs_bounds,
                        L=cfg.map_obs_size_pix,
                        W=cfg.map_obs_size_pix,
                        layers=cfg.map_layers,
                        device=device)

    # Dataset - FinetuneTrafficPlannerDataset with data_types selection
    Logger.log('Creating test dataset (data_types=%s, split=%s)...' % (cfg.data_types, cfg.test_split))
    test_dataset = FinetuneTrafficPlannerDataset(
        scenario_path=cfg.scenario_dir,
        map_env=map_env,
        split=cfg.test_split,
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        dt=cfg.dt,
        data_types=cfg.data_types)

    test_loader = GraphDataLoader(test_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=cfg.shuffle_test,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  worker_init_fn=lambda _: np.random.seed())

    # Model
    Logger.log('Initializing TrafficPlanner Model...')
    model = TrafficPlannerModel(
        cfg.past_len,
        cfg.future_len,
        cfg.map_obs_size_pix,
        len(test_dataset.categories),
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
    ).to(device)

    # Loss (eval only)
    loss_weights = {
        'recon': 1.0,
        'kl': 1.0,
        'coll_veh_prior': 0.0,
        'coll_env_prior': 0.0,
        'potential_veh': 0.0,
        'potential_env': 0.0,
        'sparse': 0.0,
    }
    loss_fn = TrafficPlannerLoss(
        loss_weights,
        phase=1,
        use_potential_loss=False,
        use_sparse_loss=False,
        use_veh_potential=False
    ).to(device)

    # Load checkpoint
    if cfg.ckpt is not None:
        ckpt_epoch, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate!')

    Logger.log('Num model params: %d' % (count_params(model)))

    # Normalizers
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # Run test
    model.eval()
    latent_accum = {
        'z_global': [],
        'z_global_labels': [],
        'z_local': [],
    }
    with torch.no_grad():
        start_t = time.time()
        try:
            run_one_epoch(
                        test_loader, model, map_env, loss_fn, device, base_out,
                        viz_suffix=cfg.viz_suffix,
                        test_recon_viz_multi=cfg.test_recon_viz_multi,
                        test_recon_coll_rate=cfg.test_recon_coll_rate,
                        test_sample_viz_multi=cfg.test_sample_viz_multi,
                        test_sample_viz_rollout=cfg.test_sample_viz_rollout,
                        test_sample_disp_err=cfg.test_sample_disp_err,
                        test_sample_coll_rate=cfg.test_sample_coll_rate,
                        test_sample_num=cfg.test_sample_num,
                        test_sample_future_len=cfg.test_sample_future_len,
                        make_video=cfg.make_video,
                        sur_gt_replay=cfg.sur_gt_replay,
                        test_latent_analysis=cfg.test_latent_analysis,
                        latent_accum=latent_accum if cfg.test_latent_analysis else None
                        )
        except KeyboardInterrupt:
            print('\n[Ctrl+C] Interrupted. Generating PCA/t-SNE with data collected so far...')
        finally:
            Logger.log('Test time: %f s' % (time.time() - start_t))
            if cfg.test_latent_analysis:
                save_latent_pca_tsne(base_out,
                                     latent_accum['z_global'],
                                     latent_accum['z_global_labels'],
                                     latent_accum['z_local'])


if __name__ == "__main__":
    main()
