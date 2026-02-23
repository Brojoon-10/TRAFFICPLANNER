# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Test script for TrafficPlannerModel
# Based on test_fit_traffic_boston_trans.py

'''
Runs through dataset and runs various evaluations on the TrafficPlanner model.

By default, computes the same losses/errors as during training.
'''

import os, shutil
import time
import tqdm
import torch
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets import nuscenes_utils as nutils
from models.trafficplanner_model import TrafficPlannerModel
from losses.trafficplanner_loss import TrafficPlannerLoss, compute_disp_err, compute_coll_rate_env, compute_coll_rate_veh
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, load_state
from utils.config import get_parser, add_base_args
import argparse


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

def parse_cfg():
    '''
    Parse given config file into a config object.

    Returns: config object and config dict
    '''
    parser = get_parser('Test TrafficPlanner Model')
    parser = add_base_args(parser)

    # additional data args
    parser.add_argument('--test_on_val', type=str2bool, default=False,
                        help="If given, uses the validation dataset rather than test set for evaluation.")
    parser.add_argument('--shuffle_test', type=str2bool, default=False,
                        help="If given, shuffles test dataset.")
    parser.add_argument('--seq_interval', type=int, default=1,
                        help='Number of steps between sequences in the dataset.')

    # TrafficPlannerModel architecture parameters (past_feat_size, future_feat_size in base_args)
    parser.add_argument('--z_local_size', type=int, default=32,
                        help='Latent dimension for z_local (ego-only reactive)')
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
    parser.add_argument('--use_sur_z_local', type=str2bool, default=True)

    # Potential field parameters (ignored in test, but needed for config compatibility)
    parser.add_argument('--k_veh_repel', type=float, default=1.0)
    parser.add_argument('--sigma_veh', type=float, default=1.5)
    parser.add_argument('--k_env_boundary', type=float, default=1.0)
    parser.add_argument('--sigma_env_boundary', type=float, default=0.5)
    parser.add_argument('--k_env_solid', type=float, default=0.8)
    parser.add_argument('--sigma_env_solid', type=float, default=0.5)
    parser.add_argument('--k_env_dashed', type=float, default=0.1)
    parser.add_argument('--sigma_env_dashed', type=float, default=0.3)
    parser.add_argument('--use_lane_lines', type=str2bool, default=True)

    #
    # test options
    #

    # Visualization suffix - allows custom naming for viz folders
    parser.add_argument('--viz_suffix', type=str, default='default',
                        help='Suffix for visualization folder names (e.g., viz_recon_multi_<suffix>)')
    
    # Device options
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (e.g., 0, 1, 2)')

    # reconstruct (use posterior)
    parser.add_argument('--test_recon_viz_multi', type=str2bool, default=False,
                        help="Save all-agent visualization for reconstructing all test sequences.")
    parser.add_argument('--test_recon_coll_rate', type=str2bool, default=False,
                        help="Computes collision rate of reconstructed test trajectories")

    # sample (use prior)
    parser.add_argument('--test_sample_viz_multi', type=str2bool, default=False,
                        help="Save all-agent visualization for sampling all test sequences.")
    parser.add_argument('--test_sample_viz_rollout', type=str2bool, default=False,
                        help="Create videos of multiple sampled futures individually.")

    parser.add_argument('--test_sample_disp_err', type=str2bool, default=False,
                        help="Computes min displacement errors (ADE, FDE, and angle-based version) based on multiple samples.")
    parser.add_argument('--test_sample_coll_rate', type=str2bool, default=False,
                        help="Computes collision rate of N random samples.")

    parser.add_argument('--test_sample_num', type=int, default=3, help='Number of future traj to sample')
    parser.add_argument('--test_sample_future_len', type=int, default=None, help='If not None, samples this many steps into the future rather than future_len')

    # Video generation option
    parser.add_argument('--make_video', type=str2bool, default=False,
                        help="If given, creates videos for visualizations.")

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict


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
                  use_challenge_splits=False,
                  make_video=False
                  ):
    '''
    Run through test dataset and perform various desired evaluations.
    '''
    pbar = tqdm.tqdm(data_loader)

    # Create visualization directories with custom suffix
    if test_recon_viz_multi:
        recon_multi_agent_out_path = os.path.join(out_path, f'viz_recon_multi_{viz_suffix}')
        mkdir(recon_multi_agent_out_path)
    if test_sample_viz_multi:
        sample_multi_agent_out_path = os.path.join(out_path, f'viz_sample_multi_{viz_suffix}')
        mkdir(sample_multi_agent_out_path)
    if test_sample_viz_rollout:
        sample_rollout_out_path = os.path.join(out_path, f'viz_sample_rollout_{viz_suffix}')
        mkdir(sample_rollout_out_path)

    metrics = {}
    freq_metrics = {}
    data_idx = 0
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        B = map_idx.size(0)
        NA = scene_graph.past.size(0)

        # uses mean of posterior to compute recon errors
        # teacher_forcing=False for test (no GT reset during decoding)
        pred = model(scene_graph, map_idx, map_env, use_post_mean=True, teacher_forcing=False)
        loss_dict = loss_fn(scene_graph, pred, use_teacher_forcing=False)
        loss = loss_dict['loss'][0]

        # compute interpretable errors
        err_dict = loss_fn.compute_err(scene_graph, pred, model.get_normalizer())

        # metrics to save
        batch_metrics = {**loss_dict, **err_dict}
        batch_freq_metrics = {}

        #
        # Reconstruction-based Evaluations
        #
        recon_pred = None
        if test_recon_viz_multi or test_recon_coll_rate:
            recon_pred = model.reconstruct(scene_graph, map_idx, map_env)

        if test_recon_viz_multi:
            # Visualize all results for each agent jointly
            for bidx in range(B):
                multi_agt_data_idx = data_idx + bidx

                # ################################################################
                # # DEBUG: Check GT values before visualization
                # ################################################################
                # binds = scene_graph.batch == bidx
                # gt_future_debug = scene_graph.future_gt[binds]
                # print(f"\n{'#'*60}")
                # print(f"### DEBUG Scene {multi_agt_data_idx} ###")
                # print(f"{'#'*60}")
                # print(f"  GT shape: {gt_future_debug.shape}")
                # print(f"  GT[0] first 3 timesteps xy:\n    {gt_future_debug[0, :3, :2]}")
                # print(f"  GT has NaN: {torch.isnan(gt_future_debug).any()}")
                # print(f"  GT xy min={gt_future_debug[:,:,:2].min():.4f}, max={gt_future_debug[:,:,:2].max():.4f}")
                # print(f"  future_vis: {scene_graph.future_vis[binds]}")
                # print(f"{'#'*60}\n")
                # ################################################################

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

            # environment collisions
            coll_rate_dict = compute_coll_rate_env(scene_graph, map_idx, coll_pred, map_env,
                                                       model.get_normalizer(), model.get_att_normalizer(),
                                                       ego_only=True)
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

            # vehicle collisions
            coll_rate_dict = compute_coll_rate_veh(scene_graph, coll_pred,
                                                       model.get_normalizer(), model.get_att_normalizer())
            coll_rate_dict = {'recon_' + k : v for k, v in coll_rate_dict.items()}
            batch_freq_metrics = {**batch_freq_metrics, **coll_rate_dict}

        #
        # Sampling-based Evaluations
        #
        sample_pred = None
        if test_sample_disp_err or test_sample_viz_multi or test_sample_coll_rate or test_sample_viz_rollout:
            sample_pred = model.sample_batched(scene_graph, map_idx, map_env, test_sample_num,
                                        include_mean=False, nfuture=test_sample_future_len)

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

        # Metrics to log to the tqdm progress bar
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

    # create output directory
    base_out = cfg.out
    mkdir(base_out)
    log_path = os.path.join(base_out, 'test_log.txt')
    Logger.init(log_path)
    Logger.log('Args: ' + str(cfg_dict))

    # device setup
    device = f'cuda:{cfg.gpu}'
    Logger.log('Using device %s...' % (str(device)))

    # load dataset
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')

    map_env = FITMapEnv(data_path,
                            bounds=cfg.map_obs_bounds,
                            L=cfg.map_obs_size_pix,
                            W=cfg.map_obs_size_pix,
                            layers=cfg.map_layers,
                            device=device,
                            )
    test_dataset = FITDataset(data_path, map_env,
                            split='test' if not cfg.test_on_val else 'val',
                            categories=cfg.agent_types,
                            npast=cfg.past_len,
                            nfuture=cfg.future_len,
                            dt=cfg.dt,
                            reduce_cats=cfg.reduce_cats,
                            seq_interval=cfg.seq_interval,
                            )

    # create loaders
    test_loader = GraphDataLoader(test_dataset,
                                    batch_size=cfg.batch_size,
                                    shuffle=cfg.shuffle_test,
                                    num_workers=cfg.num_workers,
                                    pin_memory=False,
                                    worker_init_fn=lambda _: np.random.seed())

    # create model
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

    print(model)

    # Loss for evaluation (no potential loss needed for testing)
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
        use_veh_potential=False      # Disable vehicle potential computation
    ).to(device)

    # load model weights
    if cfg.ckpt is not None:
        ckpt_epoch, _, _ = load_state(cfg.ckpt, model, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    Logger.log('Num model params: %d' % (count_params(model)))

    # set normalizers
    model.set_normalizer(test_dataset.get_state_normalizer())
    model.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # run evaluations on test data
    model.eval()
    with torch.no_grad():
        start_t = time.time()
        run_one_epoch(test_loader, model, map_env, loss_fn, device, base_out,
                    viz_suffix=cfg.viz_suffix,
                    test_recon_viz_multi=cfg.test_recon_viz_multi,
                    test_recon_coll_rate=cfg.test_recon_coll_rate,
                    test_sample_viz_multi=cfg.test_sample_viz_multi,
                    test_sample_viz_rollout=cfg.test_sample_viz_rollout,
                    test_sample_disp_err=cfg.test_sample_disp_err,
                    test_sample_coll_rate=cfg.test_sample_coll_rate,
                    test_sample_num=cfg.test_sample_num,
                    test_sample_future_len=cfg.test_sample_future_len,
                    use_challenge_splits=cfg.use_challenge_splits,
                    make_video=cfg.make_video
                    )
        Logger.log('Test time: %f s' % (time.time() - start_t))


if __name__ == "__main__":
    main()
