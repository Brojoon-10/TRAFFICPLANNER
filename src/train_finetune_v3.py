# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fine-tuning script for Agent Cross-Attention Training

Key differences from train_traffic.py:
1. Uses FinetuneDataset (JSON pairs: Normal + Adv)
2. Encoder frozen, only Decoder trained
3. Dual forward pass (Normal + Adv pairs)
4. ext_future injection for Adv pair (Sur attack trajectory)
5. No KL loss (z sampled from Prior only)
6. Separate loss weights for Normal vs Adv
"""

import os, argparse, time

import gc
import tqdm
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models.transCVAE_agent_attention import FITTrafficModel as TrafficModel

from losses.finetune_loss import FinetuneLoss, compute_finetune_loss
from datasets.finetune_dataset import FinetuneDataset, finetune_collate_fn
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device, count_params, save_state, load_state, c2c
from utils.config import get_parser, add_base_args


def parse_cfg():
    '''
    Parse given config file into a config object.
    '''
    parser = get_parser('Fine-tune Agent Cross-Attention model')
    parser = add_base_args(parser)

    # Dataset options
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Path to JSON scenario directory (adv_gen output)')

    # Training options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--val_every', type=int, default=5, help='Number of epochs between validations.')
    parser.add_argument('--save_every', type=int, default=5, help='Number of epochs between saving model checkpoint.')
    parser.add_argument('--print_every', type=int, default=10, help='Number of batches between printing stats.')

    # Optimizer options
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for ADAM')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay on params.')

    # GNN-Transformer model parameters (same as train_traffic_trans_decoder_changed.py)
    parser.add_argument('--gcn_hidden_dim', type=int, default=64,
                        help='Hidden dimension for GCN, Transformer (d_model), and GRU')
    parser.add_argument('--step_feat_dim', type=int, default=128,
                        help='Feature dimension for the input state MLP')
    parser.add_argument('--gcn_message_dim', type=int, default=128,
                        help='Hidden dimension during GCN message passing')
    parser.add_argument('--transformer_nhead', type=int, default=8,
                        help='Number of heads in Transformer Encoder')
    parser.add_argument('--transformer_nlayer', type=int, default=3,
                        help='Number of layers in Transformer Encoder')
    parser.add_argument('--decoder_gru_layers', type=int, default=3,
                        help='Number of layers in Decoder GRU')
    parser.add_argument('--decoder_attn_heads', type=int, default=8,
                        help='Number of heads in Decoder Cross-Attention')

    # Normal pair loss weights
    parser.add_argument('--normal_loss_recon', type=float, default=1.0, help='Normal pair: recon loss weight')
    parser.add_argument('--normal_loss_coll_veh', type=float, default=0.05, help='Normal pair: vehicle collision loss weight')
    parser.add_argument('--normal_loss_coll_env', type=float, default=1.0, help='Normal pair: env collision loss weight')

    # Adv pair loss weights
    parser.add_argument('--adv_loss_recon', type=float, default=0.0, help='Adv pair: recon loss weight (usually 0)')
    parser.add_argument('--adv_loss_coll_veh', type=float, default=1.0, help='Adv pair: vehicle collision loss weight')
    parser.add_argument('--adv_loss_coll_env', type=float, default=1.0, help='Adv pair: env collision loss weight')

    # Overall loss weighting
    parser.add_argument('--normal_weight', type=float, default=1.0, help='Overall weight for Normal pair loss')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Overall weight for Adv pair loss')

    args = parser.parse_args()
    config_dict = vars(args)
    config = dict2obj(config_dict)

    return config, config_dict


def freeze_encoder(model):
    """Freeze encoder parameters (Prior + Posterior networks)."""
    # Freeze Prior network
    for param in model.latent_prior_net.parameters():
        param.requires_grad = False

    # Freeze Posterior network (encoder)
    for param in model.latent_posterior_net.parameters():
        param.requires_grad = False

    # Freeze temporal GCN encoder
    for param in model.temporal_gcn_encoder.parameters():
        param.requires_grad = False

    # Freeze step feature extractor
    for param in model.step_feature_extractor.parameters():
        param.requires_grad = False

    # Freeze transformer encoder
    for param in model.transformer_encoder.parameters():
        param.requires_grad = False

    # Freeze positional encoding
    for param in model.positional_encoding.parameters():
        param.requires_grad = False

    # Freeze map encoder
    for param in model.map_conv.parameters():
        param.requires_grad = False
    for param in model.map_feature.parameters():
        param.requires_grad = False

    Logger.log('Encoder frozen. Only decoder will be trained.')


def run_one_epoch(data_loader, model, map_env, normal_loss_fn, adv_loss_fn,
                  device, out_path, cfg,
                  train=True,
                  optimizer=None,
                  step_counter=0,
                  use_wandb=False):
    '''
    Run through dataset for a single epoch.
    '''
    if use_wandb:
        import wandb
    if train and optimizer is None:
        throw_err('Must give optimizer to train!')
    prefix = "Train" if train else "Eval"

    pbar = tqdm.tqdm(data_loader)
    metrics = {}

    empty_cache = False
    for i, batch_data in enumerate(pbar):
        pred_normal = pred_adv = loss_dict = None
        if empty_cache:
            empty_cache = False
            gc.collect()
            torch.cuda.empty_cache()

        try:
            # Unpack batch
            normal_graph = batch_data['normal_graph'].to(device)
            adv_graph = batch_data['adv_graph'].to(device)
            map_idx = batch_data['map_idx'].to(device)
            ext_future = batch_data['ext_future'].to(device)  # [NA, FT, 4]

            B = map_idx.size(0)

            # =============================================================
            # Normal pair forward (no ext_future injection)
            # =============================================================
            pred_normal = model(normal_graph, map_idx, map_env,
                               use_post_mean=False,
                               future_sample=False)

            # =============================================================
            # Adv pair forward (with ext_future injection for Sur)
            # Ego predicts, Sur trajectory is injected
            # =============================================================
            # Create ext_future_mask: Sur agents get injected (not Ego)
            # In batched graph, ego_inds = adv_graph.ptr[:-1]
            NA = adv_graph.num_nodes
            ego_inds = adv_graph.ptr[:-1]
            ext_future_mask = torch.ones(NA, dtype=torch.bool, device=device)
            ext_future_mask[ego_inds] = False  # Ego predicts, Sur injected

            # Use embed + decode_embedding (same pattern as adv_gen)
            embed_out = model.embed(adv_graph, map_idx, map_env)
            prior_mu, prior_var = embed_out['prior_out']
            # Sample z from prior
            z_samp = model.rsample(prior_mu, prior_var)
            # Decode with ext_future injection
            pred_adv = model.decode_embedding(
                z_samp, embed_out, adv_graph, map_idx, map_env,
                ext_future=ext_future,
                ext_future_mask=ext_future_mask
            )

            # =============================================================
            # Compute combined loss
            # =============================================================
            loss_dict = compute_finetune_loss(
                normal_graph, pred_normal,
                adv_graph, pred_adv,
                normal_loss_fn, adv_loss_fn,
                map_idx, map_env,
                normal_weight=cfg.normal_weight,
                adv_weight=cfg.adv_weight
            )
            loss = loss_dict['loss'][0]

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        except RuntimeError as e:
            Logger.log('Caught error in training batch: %s' % (str(e)))
            Logger.log('Skipping')
            if pred_normal is not None:
                del pred_normal
            if pred_adv is not None:
                del pred_adv
            if loss_dict is not None:
                del loss_dict
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad
            empty_cache = True
            continue

        # Metrics to log
        batch_metrics = {k: v for k, v in loss_dict.items()}

        if use_wandb:
            wandb_batch_metrics = {}
            if train:
                step_counter += B
                for k, v in batch_metrics.items():
                    if v is not None and torch.is_tensor(v):
                        wandb_batch_metrics[prefix + " Batch Mean " + k] = torch.mean(v).item()
            if len(wandb_batch_metrics) > 0:
                wandb.log(wandb_batch_metrics, step=step_counter)

        # Progress bar metrics
        progress_bar_metrics = {}
        for k, v in batch_metrics.items():
            if v is None:
                continue
            if not torch.is_tensor(v):
                continue
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(c2c(v))
            progress_bar_metrics[k] = torch.mean(v).item()
        pbar.set_postfix(progress_bar_metrics)

    # Epoch metrics
    wandb_epoch_metrics = {}
    for k, v in metrics.items():
        metrics[k] = np.concatenate([np.atleast_1d(x) for x in v])
        wandb_epoch_metrics[prefix + " Epoch Mean " + k] = np.mean(metrics[k])

    mean_epoch_loss = wandb_epoch_metrics.get(prefix + " Epoch Mean loss", float('inf'))

    if use_wandb and len(wandb_epoch_metrics) > 0:
        wandb.log(wandb_epoch_metrics, step=step_counter)

    return step_counter, mean_epoch_loss


def main():
    cfg, cfg_dict = parse_cfg()

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=cfg_dict,
                   mode='offline' if cfg.wandb_offline else 'online',
                   name=cfg.wandb_name)

    # Create output directory
    mkdir(cfg.out)
    log_path = os.path.join(cfg.out, 'finetune_log.txt')
    Logger.init(log_path)
    Logger.log('Args: ' + str(cfg_dict))

    # Device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # Load map environment (same as train_traffic_trans_decoder_changed.py)
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(data_path,
                        bounds=cfg.map_obs_bounds,
                        L=cfg.map_obs_size_pix,
                        W=cfg.map_obs_size_pix,
                        layers=cfg.map_layers,
                        device=device)

    # Create dataset
    Logger.log('Loading finetune dataset from %s...' % cfg.scenario_dir)
    train_dataset = FinetuneDataset(
        data_path=data_path,
        map_env=map_env,
        split='train',
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        dt=cfg.dt,
        scenario_path=cfg.scenario_dir
    )
    val_dataset = FinetuneDataset(
        data_path=data_path,
        map_env=map_env,
        split='val',
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        dt=cfg.dt,
        scenario_path=cfg.scenario_dir
    )

    Logger.log('Train scenes: %d, Val scenes: %d' % (len(train_dataset), len(val_dataset)))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=finetune_collate_fn,
        pin_memory=False,
        worker_init_fn=lambda _: np.random.seed()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=finetune_collate_fn,
        pin_memory=False,
        worker_init_fn=lambda _: np.random.seed()
    )

    # Create model (same as train_traffic_trans_decoder_changed.py)
    model = TrafficModel(
        cfg.past_len, cfg.future_len, cfg.map_obs_size_pix,
        len(train_dataset.categories),
        map_feat_size=cfg.map_feat_size,
        gcn_hidden_dim=cfg.gcn_hidden_dim,
        latent_size=cfg.latent_size,
        output_bicycle=cfg.model_output_bicycle,
        dt=cfg.dt,
        # GNN-Transformer args
        step_feat_dim=cfg.step_feat_dim,
        gcn_message_dim=cfg.gcn_message_dim,
        transformer_nhead=cfg.transformer_nhead,
        transformer_nlayer=cfg.transformer_nlayer,
        # Decoder args
        decoder_gru_layers=cfg.decoder_gru_layers,
        decoder_attn_heads=cfg.decoder_attn_heads,
        # Map Conv args
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list
    ).to(device)

    Logger.log('Model created.')
    Logger.log('Num model params: %d' % (count_params(model)))

    # Load pretrained checkpoint (REQUIRED for fine-tuning)
    if cfg.ckpt is None:
        throw_err('Must provide pretrained checkpoint for fine-tuning!')

    ckpt_epoch, ckpt_eval_loss, _ = load_state(cfg.ckpt, model, optimizer=None, map_location=device)
    Logger.log('Loaded pretrained checkpoint from epoch %d with loss %f' % (ckpt_epoch, ckpt_eval_loss))

    # Set normalizers
    model.set_normalizer(train_dataset.get_state_normalizer())
    model.set_att_normalizer(train_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # Freeze encoder
    freeze_encoder(model)

    # Create loss functions
    normal_loss_weights = {
        'recon': cfg.normal_loss_recon,
        'coll_veh': cfg.normal_loss_coll_veh,
        'coll_env': cfg.normal_loss_coll_env
    }
    adv_loss_weights = {
        'recon': cfg.adv_loss_recon,
        'coll_veh': cfg.adv_loss_coll_veh,
        'coll_env': cfg.adv_loss_coll_env
    }

    normal_loss_fn = FinetuneLoss(
        normal_loss_weights,
        train_dataset.get_state_normalizer(),
        train_dataset.get_att_normalizer()
    ).to(device)

    adv_loss_fn = FinetuneLoss(
        adv_loss_weights,
        train_dataset.get_state_normalizer(),
        train_dataset.get_att_normalizer()
    ).to(device)

    Logger.log('Normal loss weights: %s' % str(normal_loss_weights))
    Logger.log('Adv loss weights: %s' % str(adv_loss_weights))

    # Create optimizer (only for trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    Logger.log('Num trainable params: %d' % sum(p.numel() for p in trainable_params))

    optimizer = optim.Adam(trainable_params,
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # Training loop
    ckpts_path = os.path.join(cfg.out, 'checkpoints')
    mkdir(ckpts_path)
    step_counter = 0
    min_eval_loss = float('inf')

    # Loss history for plotting (same style as train_traffic_trans_decoder_changed.py)
    train_loss_history = []
    val_loss_history = []
    fig, (ax1, ax2) = plt.subplots(1, 2)
    loss_plot_path = os.path.join(cfg.out, 'loss_finetune_v3.jpg')

    for epoch in range(cfg.epochs):
        Logger.log('Starting epoch %d...' % (epoch))

        start_t = time.time()
        model.train()

        step_counter, train_loss = run_one_epoch(
            train_loader, model, map_env, normal_loss_fn, adv_loss_fn,
            device, cfg.out, cfg,
            train=True,
            optimizer=optimizer,
            step_counter=step_counter,
            use_wandb=use_wandb
        )

        # Update train loss plot
        train_loss_history.append(train_loss)
        ax1.clear()
        ax1.plot(train_loss_history)
        ax1.set_title("Train Loss")
        plt.savefig(loss_plot_path, format='jpeg')

        torch.cuda.empty_cache()
        Logger.log('Epoch %d time: %f, train loss: %f' % (epoch, time.time() - start_t, train_loss))

        # Validation
        if epoch % cfg.val_every == 0 and len(val_dataset) > 0:
            Logger.log('Validating...')
            with torch.no_grad():
                model.eval()
                step_counter, val_loss = run_one_epoch(
                    val_loader, model, map_env, normal_loss_fn, adv_loss_fn,
                    device, cfg.out, cfg,
                    train=False,
                    step_counter=step_counter,
                    use_wandb=use_wandb
                )
            Logger.log('Epoch %d val loss: %f' % (epoch, val_loss))

            # Update val loss plot
            val_loss_history.append(val_loss)
            ax2.clear()
            ax2.plot(val_loss_history)
            ax2.set_title("Validation Loss")
            plt.savefig(loss_plot_path, format='jpeg')

            # Track best model based on val loss
            if val_loss < min_eval_loss:
                Logger.log('Best val loss so far! Saving best model...')
                min_eval_loss = val_loss
                save_file = os.path.join(ckpts_path, 'best_model_v3.pth')
                save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)

            torch.cuda.empty_cache()

        # Save checkpoint
        if epoch % cfg.save_every == 0:
            Logger.log('Saving checkpoint...')
            save_file = os.path.join(ckpts_path, 'epoch_%08d_model_v3.pth' % (epoch))
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            save_file = os.path.join(ckpts_path, 'latest_model_v3.pth')
            save_state(save_file, model, optimizer, cur_epoch=epoch, min_val_loss=min_eval_loss)
            if use_wandb:
                import wandb
                wandb.save(save_file)

    Logger.log('Training complete!')

    if use_wandb:
        import wandb
        wandb.save(log_path)


if __name__ == "__main__":
    main()
