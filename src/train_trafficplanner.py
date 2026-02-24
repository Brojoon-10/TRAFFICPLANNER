# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Training script for TrafficPlannerModel
# Based on train_fit_traffic_boston_trans.py with modifications for:
# - TrafficPlannerModel with z_global + z_local architecture
# - Phase-based training (Phase 1: pretrain, Phase 2: finetune)
# - Potential-based collision avoidance loss

import os, argparse, time

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

    # LR scheduler (cosine annealing)
    parser.add_argument('--use_lr_anneal', type=str2bool, default=False,
                        help='Enable cosine annealing LR scheduler')
    parser.add_argument('--lr_max', type=float, default=None,
                        help='Max LR for cosine annealing (default: use --lr)')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Min LR for cosine annealing')
    parser.add_argument('--lr_anneal_epochs', type=int, default=None,
                        help='(deprecated) T_max for epoch-based cosine annealing')
    parser.add_argument('--lr_warmup_steps', type=int, default=0,
                        help='Number of warmup steps for step-based LR scheduler')
    parser.add_argument('--lr_total_steps', type=int, default=None,
                        help='Total steps for LR scheduler (auto: epochs * steps_per_epoch)')

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
    parser.add_argument('--loss_kl', type=float, default=0.004, help='KL loss weight')
    parser.add_argument('--kl_anneal_end', type=int, default=20, help='If given, uses KL loss annealing and will reach full weight at this epoch.')
    parser.add_argument('--loss_recon', type=float, default=1.0, help='Reconstruction loss weight')
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
    parser.add_argument('--tf_init_segment_len', type=int, default=1,
                        help='Initial segment length for teacher forcing (anneals to future_len)')
    parser.add_argument('--tf_max_annealing_epoch', type=int, default=200,
                        help='Epoch to reach full segment length (future_len) via cosine annealing')

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
    parser.add_argument('--hist_attn_nhead', type=int, default=4,
                        help='Number of heads for decoder history attention')
    parser.add_argument('--map_attn_nhead', type=int, default=4,
                        help='Number of heads for decoder map cross-attention')
    parser.add_argument('--sur_pred_dim', type=int, default=2,
                        help='Predicted surrounding agent delta dimension (dx, dy)')
    parser.add_argument('--map_recrop', type=str2bool, default=False,
                        help='Re-crop map tokens at each decode step')
    parser.add_argument('--use_ego_intent', type=str2bool, default=True,
                        help='Enable ego intent codebook')
    parser.add_argument('--use_sur_intent', type=str2bool, default=False,
                        help='Enable sur intent codebook')

    # Redesign: auxiliary loss weights
    parser.add_argument('--loss_sur_pred', type=float, default=0.1,
                        help='Sur prediction auxiliary loss weight')
    parser.add_argument('--loss_ego_pred', type=float, default=0.1,
                        help='Ego prediction auxiliary loss weight (Phase 1 only)')
    parser.add_argument('--loss_intent_ce', type=float, default=0.0,
                        help='Intent classification CE loss weight (Phase 2 only)')
    parser.add_argument('--loss_map_attn', type=float, default=0.0,
                        help='Map attention guidance loss weight')
    parser.add_argument('--map_gt_steps', type=int, default=6,
                        help='Number of future GT steps for map attention soft label')
    parser.add_argument('--map_gt_decay_lambda', type=float, default=0.3,
                        help='Exponential decay lambda for map attention GT weights: w_t = exp(-lambda*t) / sum')
    parser.add_argument('--map_gauss_sigma_d', type=float, default=0.8,
                        help='Lateral Gaussian sigma for map attention soft label (grid cells)')
    parser.add_argument('--map_attn_anneal', type=str2bool, default=False,
                        help='Enable epoch-based cosine annealing for map attn loss (full → 0)')
    parser.add_argument('--map_attn_anneal_epochs', type=int, default=0,
                        help='Number of epochs over which map attn weight decays to 0 (0=use total epochs)')

    args = parser.parse_args()
    config_dict = vars(args)
    # Config dict to object
    config = dict2obj(config_dict)

    return config, config_dict


def run_one_epoch(data_loader, model, map_env, loss_fn, device, out_path,
                  train=True,
                  optimizer=None,
                  step_counter=0,
                  use_wandb=False,
                  use_teacher_forcing=False,
                  current_epoch=0,
                  tb_writer=None,
                  global_step=0,
                  scheduler=None):
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
            scene_graph = scene_graph.to(device)
            map_idx = map_idx.to(device)
            B = map_idx.size(0)

            do_sample = loss_fn.loss_weights.get('coll_veh_prior', 0.0) > 0.0 or \
                        loss_fn.loss_weights.get('coll_env_prior', 0.0) > 0.0

            pred = model(scene_graph, map_idx, map_env, future_sample=do_sample,
                         teacher_forcing=use_teacher_forcing if train else False)

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
                recon_val = torch.mean(loss_dict['recon_loss']).item()
                kl_val = torch.mean(loss_dict['kl_loss']).item()

                # Retrieve potential loss values if available
                pot_veh_val = loss_dict.get('potential_veh_loss', torch.tensor(0.0)).item()
                pot_env_val = loss_dict.get('potential_env_loss', torch.tensor(0.0)).item()
                sparse_val = loss_dict.get('sparse_loss', torch.tensor(0.0)).item()

                # Formatted print statement for clear and readable output.
                print(
                    f"Total Loss: {loss.item():.4f} | "
                    f"Recon: {recon_val:.4f} | "
                    f"KL: {kl_val:.4f} | "
                    f"PotVeh: {pot_veh_val:.4f} | "
                    f"PotEnv: {pot_env_val:.4f} | "
                    f"Sparse: {sparse_val:.4f}"
                )
            # --- END: ROBUST NAN DEBUGGING BLOCK ---

            if train:
                # training step for generator
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
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
    print(f'Phase: {cfg.phase}')
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
        hist_attn_nhead=cfg.hist_attn_nhead,
        map_attn_nhead=cfg.map_attn_nhead,
        sur_pred_dim=cfg.sur_pred_dim,
        map_recrop=cfg.map_recrop,
        use_ego_intent=cfg.use_ego_intent,
        use_sur_intent=cfg.use_sur_intent,
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
        'map_gauss_sigma_d': getattr(cfg, 'map_gauss_sigma_d', 0.8),
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
        aux_cfg=aux_cfg
    ).to(device)

    Logger.log('Num model params: %d' % (count_params(model)))

    # create optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # load model weights & optimizer to start from, if given
    ckpt_epoch = 0
    ckpt_eval_loss = float('inf')

    # Phase 2: Load Phase 1 checkpoint
    if cfg.phase == 2 and cfg.phase1_ckpt is not None:
        Logger.log('Loading Phase 1 checkpoint for Phase 2 training...')
        ckpt_epoch, ckpt_eval_loss, _ = load_state(cfg.phase1_ckpt, model,
                                                   optimizer=None,  # Don't load optimizer for Phase 2
                                                   map_location=device)
        Logger.log('Loaded Phase 1 checkpoint from epoch %d' % (ckpt_epoch))
        ckpt_epoch = 0  # Reset epoch counter for Phase 2
        ckpt_eval_loss = float('inf')  # Reset eval loss tracking

        # Freeze z_global encoder for Phase 2
        model.freeze_z_global()
        Logger.log('Frozen z_global encoder for Phase 2 training')

        # Recreate optimizer with only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params,
                               lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        Logger.log('Created optimizer with %d trainable parameters' % len(trainable_params))
    elif cfg.ckpt is not None:
        ckpt_epoch, ckpt_eval_loss, global_step_loaded = load_state(cfg.ckpt, model,
                                                                     optimizer=optimizer,
                                                                     map_location=device)
        Logger.log('Loaded checkpoint from epoch %d with validation loss %f...' % (ckpt_epoch, ckpt_eval_loss))

    # LR scheduler (step-based warmup + cosine decay)
    scheduler = None
    global_step = 0
    if cfg.use_lr_anneal:
        lr_max = cfg.lr_max if cfg.lr_max is not None else cfg.lr
        lr_min = cfg.lr_min
        warmup_steps = cfg.lr_warmup_steps

        # Estimate total steps
        est_steps_per_epoch = len(train_loader)
        if cfg.lr_total_steps is not None:
            total_steps = cfg.lr_total_steps
        else:
            total_steps = est_steps_per_epoch * cfg.epochs
        Logger.log(f'LR: estimated {est_steps_per_epoch} steps/epoch, {total_steps} total steps')

        # Step-based warmup + cosine decay
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

        # If resuming from checkpoint, advance scheduler
        if cfg.ckpt is not None and 'global_step_loaded' in dir():
            global_step = global_step_loaded
            for _ in range(global_step):
                scheduler.step()
            Logger.log(f'Restored LR scheduler to global_step={global_step}')

        actual_lr = optimizer.param_groups[0]['lr']
        Logger.log(f'LR Step-based: max={lr_max}, min={lr_min}, warmup={warmup_steps}, total={total_steps}')
        Logger.log(f'LR (actual): {actual_lr}')

    # Freeze z_global if requested (for custom training scenarios)
    if cfg.freeze_z_global and cfg.phase == 1:
        model.freeze_z_global()
        Logger.log('Frozen z_global encoder (manual override)')

    # so can unnormalize as needed
    model.set_normalizer(train_dataset.get_state_normalizer())
    model.set_att_normalizer(train_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

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

    # Loss function parameters
    Logger.log('\n[Loss Function]')
    Logger.log(f'  phase: {loss_fn.phase}')
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

    # Teacher forcing
    Logger.log('\n[Teacher Forcing]')
    Logger.log(f'  use_teacher_forcing: {cfg.use_teacher_forcing}')
    Logger.log(f'  mode: GT-cached independent 1-step prediction (no AR during training)')

    # Optimizer
    Logger.log('\n[Optimizer]')
    actual_init_lr = optimizer.param_groups[0]['lr']
    Logger.log(f'  lr: {actual_init_lr}')
    if cfg.use_lr_anneal:
        Logger.log(f'  lr_max: {cfg.lr_max}, lr_min: {cfg.lr_min}')
        Logger.log(f'  warmup_steps: {cfg.lr_warmup_steps}')
    Logger.log(f'  weight_decay: {cfg.weight_decay}')

    Logger.log('=' * 80)
    Logger.log('')

    # KL loss annealing (Phase 1 only)
    use_kl_anneal = cfg.kl_anneal_end is not None and cfg.phase == 1
    if use_kl_anneal:
        assert cfg.kl_anneal_end > 0
        Logger.log('Using KL annealing...')

    # TensorBoard
    tb_log_dir = os.path.join(cfg.out, 'tb_logs')
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    Logger.log(f'TensorBoard logs: {tb_log_dir}')

    # run training
    ckpts_path = os.path.join(cfg.out, 'checkpoints_trafficplanner')
    mkdir(ckpts_path)
    step_counter = 0
    min_eval_loss = ckpt_eval_loss

    # Matplotlib for loss visualization
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for epoch in range(ckpt_epoch, cfg.epochs):
        Logger.log('Starting epoch %d (Phase %d)...' % (epoch, cfg.phase))

        # compute loss weights with KL annealing (Phase 1 only)
        if use_kl_anneal:
            cur_beta = compute_kl_weight(epoch, cfg.kl_anneal_end, cfg.loss_kl)
            loss_fn.loss_weights['kl'] = cur_beta
            Logger.log('KL weight %f...' % (loss_fn.loss_weights['kl']))
            if use_wandb:
                wandb.log({'kl_weight': loss_fn.loss_weights['kl']}, step=step_counter)
            if epoch == cfg.kl_anneal_end:
                Logger.log('KL ANNEALING FINISHED: resetting val loss tracking...')
                min_eval_loss = float('inf')

        # Map attention loss annealing: cosine decay from full weight to 0
        if getattr(cfg, 'map_attn_anneal', False) and cfg.loss_map_attn > 0:
            anneal_total = cfg.map_attn_anneal_epochs if cfg.map_attn_anneal_epochs > 0 else cfg.epochs
            progress = min(epoch / max(anneal_total, 1), 1.0)
            map_attn_w = cfg.loss_map_attn * 0.5 * (1 + math.cos(math.pi * progress))
            loss_fn.loss_weights['map_attn'] = map_attn_w
            Logger.log('Map attn weight %.6f...' % map_attn_w)
            if use_wandb:
                wandb.log({'map_attn_weight': map_attn_w}, step=step_counter)

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
                                        scheduler=scheduler)
        train_loss.append(mean_train_loss)

        # TensorBoard: epoch-level train metrics
        if tb_writer is not None:
            for k, v in train_epoch_metrics.items():
                tb_writer.add_scalar(f'train/{k}', v, epoch)
            if scheduler is not None:
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        ax1.clear()
        ax1.plot(train_loss)
        ax1.set_title(f"Train Loss (Phase {cfg.phase})")
        plt.savefig(f'loss_{cfg.loss_plot_suffix}.jpg', format='jpeg')

        # Log LR after epoch
        if scheduler is not None:
            if use_wandb:
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=step_counter)

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
                valid_loss.append(mean_eval_loss)

                # TensorBoard: epoch-level val metrics
                if tb_writer is not None:
                    for k, v in val_epoch_metrics.items():
                        tb_writer.add_scalar(f'val/{k}', v, epoch)

                print(f'min_eval_loss = ', min_eval_loss)
                print(f'mean_eval_loss = ', mean_eval_loss)
                ax2.clear()
                ax2.plot(valid_loss)
                ax2.set_title(f"Validation Loss (Phase {cfg.phase})")
                plt.savefig(f'loss_{cfg.loss_plot_suffix}.jpg', format='jpeg')

                if mean_eval_loss < min_eval_loss:
                    Logger.log('Lowest eval loss so far! Saving checkpoint...')
                    min_eval_loss = mean_eval_loss
                    save_file = os.path.join(ckpts_path, f'best_eval_model_phase{cfg.phase}.pth')
                    save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=step_counter)
                    if use_wandb:
                        wandb.save(save_file)
                torch.cuda.empty_cache()

        # save checkpoint if desired
        if epoch % cfg.save_every == 0:
            Logger.log('Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=step_counter)
            save_file = os.path.join(ckpts_path, f'latest_model_phase{cfg.phase}.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss, global_step=step_counter)
            if use_wandb:
                wandb.save(save_file)

    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
    Logger.log(f'Training complete. TensorBoard: tensorboard --logdir {tb_log_dir}')

    if use_wandb:
        # save full log after training
        wandb.save(log_path)

if __name__ == "__main__":
    main()
