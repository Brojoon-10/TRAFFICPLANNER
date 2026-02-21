# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fine-tuning Loss for Agent Cross-Attention Training

Based on TrafficModelLoss, but:
1. Only computes loss for Ego (index 0 in each batch)
2. No KL loss (z is sampled from Prior, not Posterior)
3. Separate loss weights for Normal vs Adv pairs (from cfg):
   - Normal: recon + coll_veh + coll_env
   - Adv: coll_veh + coll_env only (recon=0)
"""

import torch
from torch import nn

from losses.common import log_normal
from losses.traffic_model import VehCollLoss, EnvCollLoss


class FinetuneLoss(nn.Module):
    """
    Fine-tuning loss for Ego-only training.
    Loss weights are set per-instance from cfg.
    """
    def __init__(self, loss_weights,
                 state_normalizer=None,
                 att_normalizer=None):
        '''
        :param loss_weights: dict with 'recon', 'coll_veh', 'coll_env' weights
        :param state_normalizer: normalization object for kinematic state
        :param att_normalizer: normalization object for length/width
        '''
        super(FinetuneLoss, self).__init__()
        self.loss_weights = loss_weights
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer

    def forward(self, scene_graph, pred,
                map_idx=None,
                map_env=None):
        '''
        :param scene_graph: containing input and GT data
        :param pred: dict of model predictions
        :param map_idx, map_env: for env collision losses
        '''
        ego_inds = scene_graph.ptr[:-1]
        device = pred['future_pred'].device
        loss = torch.tensor(0.0, device=device)
        loss_out = {}

        # ============================================================
        # Reconstruction loss (Ego only)
        # ============================================================
        if self.loss_weights.get('recon', 0.0) > 0.0:
            gt_future = scene_graph.future_gt[ego_inds]  # B x FT x 6
            pred_future = pred['future_pred'][ego_inds]  # B x FT x 4
            future_vis = scene_graph.future_vis[ego_inds]  # B x FT

            # Apply visibility mask
            vis_mask = future_vis == 1.0
            gt_future_masked = gt_future[vis_mask]
            pred_future_masked = pred_future[vis_mask]

            # Reconstruction loss (MSE, only first 4 dims: x, y, hcos, hsin)
            recon_loss = -log_normal(pred_future_masked, gt_future_masked[:, :4],
                                      torch.ones_like(pred_future_masked))
            loss = loss + self.loss_weights['recon'] * recon_loss.mean()
            loss_out['recon'] = recon_loss.mean().view((1,))

        # ============================================================
        # Vehicle collision loss (Ego vs Sur) - all agents needed
        # ============================================================
        if self.loss_weights.get('coll_veh', 0.0) > 0.0:
            if self.state_normalizer is None or self.att_normalizer is None:
                raise ValueError('Must have normalizers to compute collision loss!')

            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            veh_coll_loss = VehCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)

            pred_traj = self.state_normalizer.unnormalize(pred['future_pred'])
            coll_pens, na_sqr = veh_coll_loss(pred_traj)
            coll_veh_loss = torch.sum(coll_pens) / na_sqr
            loss = loss + self.loss_weights['coll_veh'] * coll_veh_loss
            loss_out['coll_veh'] = coll_veh_loss.view((1,))

        # ============================================================
        # Environment collision loss (Ego only)
        # ============================================================
        if self.loss_weights.get('coll_env', 0.0) > 0.0:
            if map_idx is None or map_env is None:
                raise ValueError('Must have map_idx and map_env for env collision loss!')
            if self.state_normalizer is None or self.att_normalizer is None:
                raise ValueError('Must have normalizers to compute collision loss!')

            veh_att = self.att_normalizer.unnormalize(scene_graph.lw[ego_inds])
            env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, pred['future_pred'].size(1))

            ego_pred_traj = self.state_normalizer.unnormalize(pred['future_pred'][ego_inds])
            coll_env_loss = env_coll_loss(ego_pred_traj)
            loss = loss + self.loss_weights['coll_env'] * coll_env_loss.mean()
            loss_out['coll_env'] = coll_env_loss.mean().view((1,))

        loss_out['loss'] = loss.view((1,))

        return loss_out


def compute_finetune_loss(normal_graph, normal_pred,
                          adv_graph, adv_pred,
                          normal_loss_fn, adv_loss_fn,
                          map_idx, map_env,
                          normal_weight=1.0, adv_weight=1.0):
    """
    Compute combined loss for Normal and Adv pairs.

    :param normal_graph, normal_pred: Normal pair data
    :param adv_graph, adv_pred: Adv pair data
    :param normal_loss_fn: FinetuneLoss with Normal weights (recon + coll)
    :param adv_loss_fn: FinetuneLoss with Adv weights (coll only, recon=0)
    :param map_idx, map_env: Map data
    :param normal_weight, adv_weight: Overall scaling factors

    :return: Combined loss dict
    """
    # Normal pair loss
    normal_loss_out = normal_loss_fn(normal_graph, normal_pred, map_idx, map_env)

    # Adv pair loss
    adv_loss_out = adv_loss_fn(adv_graph, adv_pred, map_idx, map_env)

    # Combine
    total_loss = normal_weight * normal_loss_out['loss'] + adv_weight * adv_loss_out['loss']

    return {
        'loss': total_loss,
        # Adv losses (first for terminal display)
        'adv_loss': adv_loss_out['loss'],
        'adv_coll_veh': adv_loss_out.get('coll_veh', torch.tensor(0.0)),
        'adv_coll_env': adv_loss_out.get('coll_env', torch.tensor(0.0)),
        # Normal losses
        'normal_loss': normal_loss_out['loss'],
        'normal_recon': normal_loss_out.get('recon', torch.tensor(0.0)),
        'normal_coll_veh': normal_loss_out.get('coll_veh', torch.tensor(0.0)),
        'normal_coll_env': normal_loss_out.get('coll_env', torch.tensor(0.0)),
    }
