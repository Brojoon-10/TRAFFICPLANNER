# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Fine-tuning script for TrafficPlannerModel
# Trains ego reactive ability using adversarial scenario + solution data.
# Freezes: z_global encoder + GCN + map + sur decoder
# Trains: z_local components + ego decoder
#
# Dataset: each scene produces 2 samples (normal + adv_sol), shuffled together.
# Same training loop structure as train_trafficplanner.py.

import os, argparse, time, csv

import gc
import tqdm
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader as GraphDataLoader

from models.trafficplanner_model import TrafficPlannerModel
from losses.trafficplanner_loss import TrafficPlannerLoss

from datasets.finetune_trafficplanner_dataset import FinetuneTrafficPlannerDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, save_state, load_state, c2c
from utils.config import get_parser, add_base_args
import matplotlib.pyplot as plt


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
    parser = get_parser('Fine-tune TrafficPlanner Model')
    parser = add_base_args(parser)

    # Fine-tuning specific
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Path to adv_sol_success/ JSON directory')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Pre-trained TrafficPlannerModel checkpoint')
    parser.add_argument('--data_types', type=str, default='both',
                        choices=['both', 'normal', 'adv_sol'],
                        help='Data types to train: both, normal, or adv_sol')

    # Training options
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=10)

    # Device
    parser.add_argument('--gpu', type=int, default=0)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # LR scheduler (cosine annealing)
    parser.add_argument('--use_lr_anneal', type=str2bool, default=False,
                        help='Enable cosine annealing LR scheduler')
    parser.add_argument('--lr_max', type=float, default=None,
                        help='Max LR for cosine annealing (default: use --lr)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Min LR for cosine annealing')
    parser.add_argument('--lr_anneal_epochs', type=int, default=None,
                        help='T_max for cosine annealing (default: use --epochs)')

    # TrafficPlannerModel architecture (must match pretrained)
    parser.add_argument('--z_local_size', type=int, default=32)
    parser.add_argument('--num_intents', type=int, default=8,
                        help='Number of intent codebook entries (K)')
    parser.add_argument('--sur_pred_dim', type=int, default=2,
                        help='Predicted surrounding agent delta dimension (dx, dy)')
    parser.add_argument('--map_recrop', type=str2bool, default=False,
                        help='Re-crop map tokens at each decode step')
    # Transformer decoder params (must match pre-trained model)
    parser.add_argument('--trans_num_layers', type=int, default=4)
    parser.add_argument('--trans_d_model', type=int, default=128)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--trans_ffn_dim', type=int, default=512)
    parser.add_argument('--trans_dropout', type=float, default=0.1)
    parser.add_argument('--use_ego_z_local', type=str2bool, default=True,
                        help='Enable ego z_local (IntentCodebook)')
    parser.add_argument('--use_sur_z_local', type=str2bool, default=False,
                        help='Enable sur z_local via separate IntentCodebook')

    # Loss weights
    parser.add_argument('--loss_recon', type=float, default=1.0)
    parser.add_argument('--loss_kl', type=float, default=0.0,
                        help='KL loss weight (0 for fine-tuning since z_global is frozen)')
    parser.add_argument('--loss_veh_coll_prior', type=float, default=0.05)
    parser.add_argument('--loss_env_coll_prior', type=float, default=0.1)

    # Auxiliary losses (Redesign)
    parser.add_argument('--loss_sur_pred', type=float, default=0.1)
    parser.add_argument('--loss_ego_pred', type=float, default=0.0)
    parser.add_argument('--loss_intent_ce', type=float, default=0.1)
    parser.add_argument('--loss_map_attn', type=float, default=0.1)

    # Potential loss (disabled by default)
    parser.add_argument('--use_potential_loss', type=str2bool, default=False)
    parser.add_argument('--use_veh_potential', type=str2bool, default=False)
    parser.add_argument('--loss_potential_veh', type=float, default=0.0)
    parser.add_argument('--loss_potential_env', type=float, default=0.0)

    # Potential field parameters (needed for config compatibility)
    parser.add_argument('--k_veh_repel', type=float, default=1.0)
    parser.add_argument('--sigma_veh', type=float, default=1.5)
    parser.add_argument('--k_env_boundary', type=float, default=1.0)
    parser.add_argument('--sigma_env_boundary', type=float, default=0.5)
    parser.add_argument('--k_env_solid', type=float, default=0.8)
    parser.add_argument('--sigma_env_solid', type=float, default=0.5)
    parser.add_argument('--k_env_dashed', type=float, default=0.1)
    parser.add_argument('--sigma_env_dashed', type=float, default=0.3)
    parser.add_argument('--use_lane_lines', type=str2bool, default=True)
    parser.add_argument('--env_loss_ego_only', type=str2bool, default=True)

    # Sparsity loss (disabled)
    parser.add_argument('--use_sparse_loss', type=str2bool, default=False)
    parser.add_argument('--loss_sparse', type=float, default=0.0)
    parser.add_argument('--target_sparsity', type=float, default=0.1)

    # Teacher forcing
    parser.add_argument('--use_teacher_forcing', type=str2bool, default=True)

    # Loss plot
    parser.add_argument('--loss_plot_suffix', type=str, default='finetune_trafficplanner')

    args = parser.parse_args()
    config_dict = vars(args)
    config = dict2obj(config_dict)
    return config, config_dict


def run_one_epoch(data_loader, model, map_env, loss_fn, device,
                  train=True,
                  optimizer=None,
                  use_teacher_forcing=False,
                  current_epoch=0,
                  tb_writer=None,
                  global_step=0):
    """Same structure as train_trafficplanner.py run_one_epoch."""
    if train and optimizer is None:
        throw_err('Must give optimizer to train!')
    prefix = "Train" if train else "Eval"

    pbar = tqdm.tqdm(data_loader)
    metrics = {}
    empty_cache = False

    for i, data in enumerate(pbar):
        scene_graph, map_idx = data

        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()

        try:
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)

            do_sample = loss_fn.loss_weights.get('coll_veh_prior', 0.0) > 0.0 or \
                        loss_fn.loss_weights.get('coll_env_prior', 0.0) > 0.0

            pred = model(scene_graph, map_idx, map_env,
                         future_sample=do_sample,
                         teacher_forcing=use_teacher_forcing if train else False,
                         current_epoch=current_epoch)

            loss_dict = loss_fn(scene_graph, pred,
                                map_idx=map_idx,
                                map_env=map_env,
                                model=model,
                                use_teacher_forcing=use_teacher_forcing if train else False)

            loss = loss_dict['loss'][0]

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            err_dict = loss_fn.compute_err(scene_graph, pred, model.get_normalizer())

        except RuntimeError as e:
            import traceback
            Logger.log('Caught error in batch %d: %s' % (i, str(e)))
            traceback.print_exc()
            Logger.log('Skipping batch')
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad
            empty_cache = True
            continue

        batch_metrics = {**loss_dict, **err_dict}

        progress_bar_metrics = {}
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(c2c(v))
            progress_bar_metrics[k] = torch.mean(v).item()
        pbar.set_postfix(progress_bar_metrics)

        # TensorBoard: per-batch logging (train only, every 100 batches)
        if tb_writer is not None and train:
            global_step += 1
            if global_step % 10 == 0:
                for k, v in progress_bar_metrics.items():
                    tb_writer.add_scalar(f'batch/{k}', v, global_step)

    epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = np.concatenate(metrics[k])
        epoch_metrics[k] = np.mean(metrics[k])

    mean_epoch_loss = epoch_metrics.get('loss', float('inf'))
    return mean_epoch_loss, epoch_metrics, global_step


def main():
    cfg, cfg_dict = parse_cfg()

    print('=== TrafficPlanner Fine-tuning ===')
    print(f'pretrained_ckpt: {cfg.pretrained_ckpt}')
    print(f'scenario_dir: {cfg.scenario_dir}')
    print(f'data_types: {cfg.data_types}')
    print(f'use_teacher_forcing: {cfg.use_teacher_forcing}')

    # Create output directory and logging
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'train_log.txt')
    Logger.init(log_path)
    Logger.log('Args: ' + str(cfg_dict))

    # Device
    device = f'cuda:{cfg.gpu}'
    Logger.log('Using device %s...' % str(device))

    # Map environment
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(data_path,
                        bounds=cfg.map_obs_bounds,
                        L=cfg.map_obs_size_pix,
                        W=cfg.map_obs_size_pix,
                        layers=cfg.map_layers,
                        device=device)

    # Dataset (flat: each scene → 2 samples, shuffled together)
    Logger.log('Creating fine-tuning dataset...')
    train_dataset = FinetuneTrafficPlannerDataset(
        scenario_path=cfg.scenario_dir,
        map_env=map_env,
        split='train',
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        dt=cfg.dt,
        data_types=cfg.data_types)
    val_dataset = FinetuneTrafficPlannerDataset(
        scenario_path=cfg.scenario_dir,
        map_env=map_env,
        split='val',
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        dt=cfg.dt,
        data_types=cfg.data_types)

    train_loader = GraphDataLoader(train_dataset,
                                   batch_size=cfg.batch_size,
                                   shuffle=True,
                                   num_workers=cfg.num_workers,
                                   pin_memory=False,
                                   worker_init_fn=lambda _: np.random.seed())
    val_loader = GraphDataLoader(val_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.num_workers,
                                 pin_memory=False,
                                 worker_init_fn=lambda _: np.random.seed())

    # Initialize model
    Logger.log('Initializing TrafficPlanner Model...')
    model = TrafficPlannerModel(
        cfg.past_len,
        cfg.future_len,
        cfg.map_obs_size_pix,
        len(train_dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        z_local_size=cfg.z_local_size,
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
    ).to(device)

    # Load pre-trained checkpoint
    Logger.log('Loading pre-trained checkpoint: %s' % cfg.pretrained_ckpt)
    ckpt_epoch, _, _ = load_state(cfg.pretrained_ckpt, model, optimizer=None, map_location=device)
    Logger.log('Loaded checkpoint from epoch %d' % ckpt_epoch)

    # Freeze for fine-tuning (z_global + GCN + map + sur decoder)
    model.freeze_for_finetuning()

    # Create optimizer with only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    Logger.log('Optimizer created with %d trainable parameter groups' % len(trainable_params))
    Logger.log('Total model params: %d' % count_params(model))

    # LR scheduler (cosine annealing)
    scheduler = None
    if cfg.use_lr_anneal:
        lr_max = cfg.lr_max if cfg.lr_max is not None else cfg.lr
        lr_min = cfg.lr_min
        T_max = cfg.lr_anneal_epochs if cfg.lr_anneal_epochs is not None else cfg.epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr_max
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min)
        Logger.log(f'LR Cosine Annealing: max={lr_max}, min={lr_min}, T_max={T_max}')

    # Set normalizers
    model.set_normalizer(train_dataset.get_state_normalizer())
    model.set_att_normalizer(train_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # Loss function
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
    }

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

    sparsity_cfg = {
        'target_sparsity': cfg.target_sparsity,
        'kl_weight': 0.1,
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights,
        train_dataset.get_state_normalizer(),
        train_dataset.get_att_normalizer(),
        phase=2,  # No z_global KL
        use_sparse_loss=cfg.use_sparse_loss,
        use_potential_loss=cfg.use_potential_loss,
        use_veh_potential=cfg.use_veh_potential,
        potential_cfg=potential_cfg,
        sparsity_cfg=sparsity_cfg,
        ego_only_recon=True
    ).to(device)

    # Print applied configuration
    Logger.log('=' * 80)
    Logger.log('APPLIED CONFIGURATION (Fine-tuning)')
    Logger.log('=' * 80)

    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.log(f'\n[Freeze/Train]')
    Logger.log(f'  Frozen params: {num_frozen}')
    Logger.log(f'  Trainable params: {num_trainable}')

    Logger.log(f'\n[Loss Weights]')
    for key, value in loss_fn.loss_weights.items():
        Logger.log(f'  {key}: {value}')

    Logger.log(f'\n[Teacher Forcing]')
    Logger.log(f'  use_teacher_forcing: {cfg.use_teacher_forcing}')

    Logger.log(f'\n[Optimizer]')
    Logger.log(f'  lr: {cfg.lr}')
    Logger.log(f'  weight_decay: {cfg.weight_decay}')
    Logger.log('=' * 80)

    # Training loop
    ckpts_path = os.path.join(cfg.out, 'checkpoints_finetune')
    mkdir(ckpts_path)
    min_eval_loss = float('inf')

    train_loss = []
    valid_loss = []
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # TensorBoard
    tb_log_dir = os.path.join(cfg.out, 'tb_logs')
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    Logger.log(f'TensorBoard logs: {tb_log_dir}')

    # CSV logging
    csv_dir = os.path.join(cfg.out, 'csv_logs')
    mkdir(csv_dir)
    csv_train_path = os.path.join(csv_dir, 'train_losses.csv')
    csv_val_path = os.path.join(csv_dir, 'val_losses.csv')
    csv_train_file = open(csv_train_path, 'w', newline='')
    csv_val_file = open(csv_val_path, 'w', newline='')
    csv_train_writer = None
    csv_val_writer = None
    Logger.log(f'CSV logs: {csv_dir}')

    global_step = 0  # batch-level step counter for TensorBoard

    for epoch in range(cfg.epochs):
        Logger.log('Starting epoch %d...' % epoch)

        # Train
        start_t = time.time()
        model.train()
        mean_train_loss, train_epoch_metrics, global_step = run_one_epoch(
            train_loader, model, map_env, loss_fn, device,
            train=True,
            optimizer=optimizer,
            use_teacher_forcing=cfg.use_teacher_forcing,
            current_epoch=epoch,
            tb_writer=tb_writer,
            global_step=global_step)
        train_loss.append(mean_train_loss)

        # TensorBoard: log train metrics
        if tb_writer is not None:
            for k, v in train_epoch_metrics.items():
                tb_writer.add_scalar(f'train/{k}', v, epoch)
            if scheduler is not None:
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

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
        ax1.set_title('Train Loss (Fine-tune)')
        plt.savefig(os.path.join(cfg.out, f'loss_{cfg.loss_plot_suffix}.png'), format='png')

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()
        Logger.log('Epoch %d - Train loss: %.6f - Time: %.1fs' % (epoch, mean_train_loss, time.time() - start_t))

        # Validate
        if epoch % cfg.val_every == 0:
            Logger.log('Validating...')
            with torch.no_grad():
                model.eval()
                mean_eval_loss, val_epoch_metrics, _ = run_one_epoch(
                    val_loader, model, map_env, loss_fn, device,
                    train=False,
                    use_teacher_forcing=False,
                    current_epoch=epoch)
                valid_loss.append(mean_eval_loss)

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

                ax2.clear()
                ax2.plot(valid_loss)
                ax2.set_title('Validation Loss (Fine-tune)')
                plt.savefig(os.path.join(cfg.out, f'loss_{cfg.loss_plot_suffix}.png'), format='png')

                Logger.log('Epoch %d - Eval loss: %.6f (best: %.6f)' % (epoch, mean_eval_loss, min_eval_loss))

                if mean_eval_loss < min_eval_loss:
                    Logger.log('New best! Saving checkpoint...')
                    min_eval_loss = mean_eval_loss
                    save_file = os.path.join(ckpts_path, 'best_finetune_model.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)

                torch.cuda.empty_cache()

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % epoch)
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            save_file = os.path.join(ckpts_path, 'latest_finetune_model.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)

    # Close loggers
    if tb_writer is not None:
        tb_writer.close()
    csv_train_file.close()
    csv_val_file.close()
    Logger.log(f'Fine-tuning complete. TensorBoard: tensorboard --logdir {tb_log_dir}')


if __name__ == '__main__':
    main()
