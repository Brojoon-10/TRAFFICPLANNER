# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Loss functions for TrafficPlannerModel
# Extended from TrafficModelLoss with:
# - Phase-based training (Phase 1: pretrain, Phase 2: finetune)
# - z_local sparsity loss
# - Potential-based collision avoidance loss

import time

import torch
from torch import nn

import numpy as np

from losses.common import kl_normal, log_normal
from utils.transforms import transform2frame
from utils.torch import c2c
import datasets.nuscenes_utils as nutils

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)


#
# Potential-based Loss Classes
#

class VehPotentialLoss(nn.Module):
    """
    Vehicle repulsion potential loss for ego vehicle only.

    Ego receives exponential repulsion from other agents.
    This prevents ego from getting too close to other vehicles
    while not affecting other agents' behavior.

    Potential: U(d) = k * exp(-d / sigma)
    """
    def __init__(self, state_normalizer=None, k_repel=2.0, sigma=3.0, min_dist=1.0):
        """
        :param state_normalizer: normalizer for kinematic state
        :param k_repel: repulsion strength
        :param sigma: repulsion decay rate
        :param min_dist: minimum distance clamp
        """
        super(VehPotentialLoss, self).__init__()
        self.state_normalizer = state_normalizer
        self.k_repel = k_repel
        self.sigma = sigma
        self.min_dist = min_dist

    def forward(self, pred_traj, scene_graph):
        """
        :param pred_traj: (NA, FT, 4) predicted trajectories (normalized)
        :param scene_graph: agent information with ptr for batch structure
        :return: scalar potential loss
        """
        NA, FT, _ = pred_traj.size()
        device = pred_traj.device

        if NA <= 1:
            return torch.tensor(0.0, device=device)

        # Unnormalize for distance computation
        if self.state_normalizer is not None:
            pred_traj_unnorm = self.state_normalizer.unnormalize(pred_traj)
        else:
            pred_traj_unnorm = pred_traj

        # Get ego indices and batch info
        ego_inds = scene_graph.ptr[:-1]  # (B,) indices of ego in flattened NA
        B = len(ego_inds)

        total_repulsion = torch.tensor(0.0, device=device)
        num_ego_pairs = 0

        for t in range(FT):
            positions_t = pred_traj_unnorm[:, t, :2]  # (NA, 2)

            # For each batch, compute ego's repulsion from other agents
            for b in range(B):
                scene_start = scene_graph.ptr[b].item()
                scene_end = scene_graph.ptr[b + 1].item()
                ego_idx = ego_inds[b].item()

                # Ego position
                ego_pos = positions_t[ego_idx:ego_idx + 1]  # (1, 2)

                # Other agents' positions in this scene
                other_mask = torch.ones(scene_end - scene_start, dtype=torch.bool, device=device)
                other_mask[0] = False  # ego is always first in scene
                scene_positions = positions_t[scene_start:scene_end]
                other_positions = scene_positions[other_mask]  # (num_other, 2)

                if other_positions.size(0) == 0:
                    continue

                # Distance from ego to each other agent
                distances = torch.norm(ego_pos - other_positions, dim=-1)  # (num_other,)
                distances = torch.clamp(distances, min=self.min_dist)

                # Exponential repulsion
                repulsion = self.k_repel * torch.exp(-distances / self.sigma)

                total_repulsion = total_repulsion + repulsion.sum()
                num_ego_pairs += other_positions.size(0)

        if num_ego_pairs > 0:
            return total_repulsion / (num_ego_pairs * FT)
        else:
            return torch.tensor(0.0, device=device)


class EnvPotentialLoss(nn.Module):
    """
    Environment boundary repulsion potential loss for ego vehicle only.

    Ego is repelled from:
    1. Non-drivable area boundaries (strongest repulsion)
    2. Solid lane lines (strong repulsion - lane change prohibited)
    3. Dashed lane lines (weak repulsion - lane change allowed)

    Map raster channels (from fit_utils.py):
    - Channel 0: drivable_area
    - Channel 1: solid_line (lane_change=no)
    - Channel 2: dashed_line (lane_change=yes)

    Potential: U(d) = k * exp(-d / sigma)
    """
    def __init__(self, state_normalizer=None, att_normalizer=None,
                 k_repel_boundary=2.0, sigma_boundary=2.0,
                 k_repel_solid=1.5, sigma_solid=1.5,
                 k_repel_dashed=0.3, sigma_dashed=1.0,
                 use_lane_lines=True,
                 ego_only=False):
        """
        :param state_normalizer: normalizer for kinematic state
        :param att_normalizer: normalizer for vehicle attributes (length, width)
        :param k_repel_boundary: repulsion strength for non-drivable boundary
        :param sigma_boundary: repulsion decay rate for boundary
        :param k_repel_solid: repulsion strength for solid lane lines
        :param sigma_solid: repulsion decay rate for solid lines
        :param k_repel_dashed: repulsion strength for dashed lane lines (weaker)
        :param sigma_dashed: repulsion decay rate for dashed lines
        :param use_lane_lines: whether to use lane line repulsion (requires 3-channel map)
        :param ego_only: if True, only apply to ego; if False, apply to all agents
        """
        super(EnvPotentialLoss, self).__init__()
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer

        # Boundary (non-drivable area) parameters
        self.k_repel_boundary = k_repel_boundary
        self.sigma_boundary = sigma_boundary

        # Solid line parameters (strong repulsion)
        self.k_repel_solid = k_repel_solid
        self.sigma_solid = sigma_solid

        # Dashed line parameters (weak repulsion)
        self.k_repel_dashed = k_repel_dashed
        self.sigma_dashed = sigma_dashed

        self.use_lane_lines = use_lane_lines
        self.ego_only = ego_only

    def forward(self, pred_traj, scene_graph, map_idx, map_env):
        """
        :param pred_traj: (NA, FT, 4) predicted trajectories (normalized)
        :param scene_graph: agent information
        :param map_idx: (B,) map index for each batch
        :param map_env: map environment object with raster data
        :return: scalar potential loss
        """
        NA, FT, _ = pred_traj.size()
        device = pred_traj.device

        # Get ego indices and batch info
        ego_inds = scene_graph.ptr[:-1]
        B = len(ego_inds)

        if B == 0:
            return torch.tensor(0.0, device=device)

        # Unnormalize trajectories
        if self.state_normalizer is not None:
            pred_traj_unnorm = self.state_normalizer.unnormalize(pred_traj)
        else:
            pred_traj_unnorm = pred_traj

        if self.ego_only:
            # Get ego trajectories only
            agent_traj = pred_traj_unnorm[ego_inds]  # (B, FT, 4)
            # Unnormalize vehicle attributes for ego
            if self.att_normalizer is not None:
                veh_att = self.att_normalizer.unnormalize(scene_graph.lw[ego_inds])  # (B, 2)
            else:
                veh_att = scene_graph.lw[ego_inds]
            # map_idx stays as (B,)
            agent_map_idx = map_idx
        else:
            # Use all agents
            agent_traj = pred_traj_unnorm  # (NA, FT, 4)
            # Unnormalize vehicle attributes for all
            if self.att_normalizer is not None:
                veh_att = self.att_normalizer.unnormalize(scene_graph.lw)  # (NA, 2)
            else:
                veh_att = scene_graph.lw
            # Expand map_idx to match all agents: each agent gets its batch's map_idx
            ptr_cpu = scene_graph.ptr.cpu().numpy()
            agent_map_idx = torch.zeros(NA, dtype=map_idx.dtype, device=device)
            for b in range(B):
                agent_map_idx[ptr_cpu[b]:ptr_cpu[b+1]] = map_idx[b]

        # Get raster channels
        # Channel 0: drivable_area, Channel 1: solid_line, Channel 2: dashed_line
        drivable_raster = map_env.nusc_raster[:, 0]

        # Check if lane line rasters are available (3-channel map)
        has_lane_lines = self.use_lane_lines and map_env.nusc_raster.size(1) >= 3
        if has_lane_lines:
            # Invert lane line rasters: get_coll_point finds 0-valued regions,
            # but lane lines are 1 where line exists. Invert so it finds line pixels.
            solid_line_raster = 1 - map_env.nusc_raster[:, 1]
            dashed_line_raster = 1 - map_env.nusc_raster[:, 2]

        total_repulsion = torch.tensor(0.0, device=device)
        num_valid_points = 0
        num_agents = agent_traj.size(0)  # B if ego_only, NA otherwise

        for t in range(FT):
            agent_state_t = agent_traj[:, t, :]  # (num_agents, 4)

            # 1. Boundary repulsion (non-drivable area)
            coll_pt_boundary = nutils.get_coll_point(
                drivable_raster,
                map_env.nusc_dx,
                agent_state_t.detach(),
                veh_att,
                agent_map_idx
            )  # (num_agents, 2) - closest non-drivable point

            valid_boundary = ~torch.isnan(torch.sum(coll_pt_boundary, dim=1))

            if torch.sum(valid_boundary) > 0:
                agent_pos = agent_state_t[valid_boundary, :2]
                coll_pt_valid = coll_pt_boundary[valid_boundary]
                distances = torch.norm(agent_pos - coll_pt_valid, dim=-1)
                repulsion = self.k_repel_boundary * torch.exp(-distances / self.sigma_boundary)
                total_repulsion = total_repulsion + repulsion.sum()
                num_valid_points += valid_boundary.sum().item()

            # 2. Solid line repulsion (strong - lane change prohibited)
            if has_lane_lines:
                coll_pt_solid = nutils.get_coll_point(
                    solid_line_raster,
                    map_env.nusc_dx,
                    agent_state_t.detach(),
                    veh_att,
                    agent_map_idx
                )

                valid_solid = ~torch.isnan(torch.sum(coll_pt_solid, dim=1))

                if torch.sum(valid_solid) > 0:
                    agent_pos = agent_state_t[valid_solid, :2]
                    coll_pt_valid = coll_pt_solid[valid_solid]
                    distances = torch.norm(agent_pos - coll_pt_valid, dim=-1)
                    repulsion = self.k_repel_solid * torch.exp(-distances / self.sigma_solid)
                    total_repulsion = total_repulsion + repulsion.sum()
                    num_valid_points += valid_solid.sum().item()

            # 3. Dashed line repulsion (weak - lane change allowed)
            if has_lane_lines:
                coll_pt_dashed = nutils.get_coll_point(
                    dashed_line_raster,
                    map_env.nusc_dx,
                    agent_state_t.detach(),
                    veh_att,
                    agent_map_idx
                )

                valid_dashed = ~torch.isnan(torch.sum(coll_pt_dashed, dim=1))

                if torch.sum(valid_dashed) > 0:
                    agent_pos = agent_state_t[valid_dashed, :2]
                    coll_pt_valid = coll_pt_dashed[valid_dashed]
                    distances = torch.norm(agent_pos - coll_pt_valid, dim=-1)
                    repulsion = self.k_repel_dashed * torch.exp(-distances / self.sigma_dashed)
                    total_repulsion = total_repulsion + repulsion.sum()
                    num_valid_points += valid_dashed.sum().item()

        if num_valid_points > 0:
            return total_repulsion / num_valid_points
        else:
            return torch.tensor(0.0, device=device)


class SparsityLoss(nn.Module):
    """
    Sparsity loss for z_local to encourage minimal deviation from z_global.

    z_local should only activate when necessary (reactive maneuvers),
    otherwise it should be close to zero (relying on z_global).

    Combines L1 sparsity with optional KL divergence for target sparsity level.
    """
    def __init__(self, target_sparsity=0.1, kl_weight=0.1):
        """
        :param target_sparsity: target fraction of active z_local dimensions
        :param kl_weight: weight for KL divergence term
        """
        super(SparsityLoss, self).__init__()
        self.target_sparsity = target_sparsity
        self.kl_weight = kl_weight

    def forward(self, z_local):
        """
        :param z_local: (FT, num_ego, z_local_size) or (FT, num_ego, NS, z_local_size)
        :return: scalar sparsity loss
        """
        if z_local is None:
            return torch.tensor(0.0)

        # L1 sparsity: encourage z_local to be sparse
        l1_loss = torch.mean(torch.abs(z_local))

        # KL divergence to encourage specific sparsity level
        mean_activation = torch.mean(torch.abs(z_local), dim=-1)
        sparsity_target = torch.full_like(mean_activation, self.target_sparsity)

        # KL divergence between Bernoulli distributions
        kl_loss = torch.mean(
            sparsity_target * torch.log(sparsity_target / (mean_activation + 1e-8) + 1e-8) +
            (1 - sparsity_target) * torch.log((1 - sparsity_target) / (1 - mean_activation + 1e-8) + 1e-8)
        )

        return l1_loss + self.kl_weight * kl_loss

class TrafficPlannerLoss(nn.Module):
    """
    Loss function for TrafficPlannerModel (Redesign).

    Phase 1 (Pre-training):
        L = L_recon + λ_kl * L_kl + λ_sur_pred * L_sur_pred + λ_ego_pred * L_ego_pred + λ_map_attn * L_map_attn

    Phase 2 (Fine-tuning):
        L = L_recon + λ_sur_pred * L_sur_pred + λ_intent_ce * L_intent_ce + λ_map_attn * L_map_attn
        (z_global frozen, ego_pred disabled, intent_ce active)

    Auxiliary Losses:
        (A) Intent CE: z_local → 9-class (3acc × 3yaw) — Phase 2 only
        (B) Sur Pred: ego_hist_ctx → sur delta — Phase 1+2
        (C) Ego Pred: sur_hist_ctx → ego delta — Phase 1 only
        (D) Map Attn Guidance: attention weight vs GT position KL — Phase 1+2
    """
    def __init__(self, loss_weights,
                    state_normalizer=None,
                    att_normalizer=None,
                    phase=1,
                    use_sparse_loss=False,
                    use_potential_loss=True,
                    use_veh_potential=False,
                    potential_cfg=None,
                    sparsity_cfg=None,
                    ego_only_recon=False,
                    aux_cfg=None):
        """
        :param loss_weights: dict of weightings for loss terms
        :param aux_cfg: dict of auxiliary loss config (optional)
            - sur_pred_dim: 2 (dx, dy)
            - map_gt_steps: 4
            - map_gt_weights: [0.4, 0.3, 0.2, 0.1]
            - acc_bins: [-1.0, 1.0]  (3 bins: decel, maintain, accel)
            - yaw_bins: [-0.1, 0.1]  (3 bins: left, straight, right)
        """
        super(TrafficPlannerLoss, self).__init__()
        self.loss_weights = loss_weights
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer
        self.phase = phase
        self.use_sparse_loss = use_sparse_loss
        self.use_potential_loss = use_potential_loss
        self.use_veh_potential = use_veh_potential
        self.ego_only_recon = ego_only_recon

        # Auxiliary loss config
        default_aux_cfg = {
            'sur_pred_dim': 2,
            'map_gt_steps': 6,
            'map_gt_weights': [0.3, 0.25, 0.2, 0.15, 0.07, 0.03],
            'acc_bins': [-1.0, 1.0],   # normalized thresholds
            'yaw_bins': [-0.1, 0.1],
        }
        if aux_cfg is not None:
            default_aux_cfg.update(aux_cfg)
        self.aux_cfg = default_aux_cfg

        # Potential field parameters (configurable via cfg)
        # Tuned for lane width 3.5m (center-to-boundary 1.75m)
        default_potential_cfg = {
            # Vehicle repulsion
            'k_veh_repel': 1.0,      # vehicle repulsion strength
            'sigma_veh': 1.5,        # vehicle repulsion decay rate
            'min_dist_veh': 1.0,     # minimum vehicle distance
            # Environment boundary repulsion (non-drivable area) - strongest
            'k_env_boundary': 1.0,   # boundary repulsion strength
            'sigma_env_boundary': 0.5,  # boundary repulsion decay rate
            # Solid lane line repulsion (lane change prohibited) - slightly weaker than boundary
            'k_env_solid': 0.8,      # solid line repulsion strength
            'sigma_env_solid': 0.5,  # solid line repulsion decay rate
            # Dashed lane line repulsion (lane change allowed) - weak
            'k_env_dashed': 0.1,     # dashed line repulsion strength (weaker)
            'sigma_env_dashed': 0.3, # dashed line repulsion decay rate
            # Lane line usage flag
            'use_lane_lines': True,  # whether to use lane line repulsion
        }
        if potential_cfg is not None:
            default_potential_cfg.update(potential_cfg)
        self.potential_cfg = default_potential_cfg

        # Sparsity loss parameters
        default_sparsity_cfg = {
            'target_sparsity': 0.1,
            'kl_weight': 0.1,
        }
        if sparsity_cfg is not None:
            default_sparsity_cfg.update(sparsity_cfg)
        self.sparsity_cfg = default_sparsity_cfg

        # Initialize loss modules
        self.veh_potential_loss = VehPotentialLoss(
            state_normalizer=state_normalizer,
            k_repel=self.potential_cfg['k_veh_repel'],
            sigma=self.potential_cfg['sigma_veh'],
            min_dist=self.potential_cfg['min_dist_veh']
        )

        self.env_potential_loss = EnvPotentialLoss(
            state_normalizer=state_normalizer,
            att_normalizer=att_normalizer,
            k_repel_boundary=self.potential_cfg['k_env_boundary'],
            sigma_boundary=self.potential_cfg['sigma_env_boundary'],
            k_repel_solid=self.potential_cfg['k_env_solid'],
            sigma_solid=self.potential_cfg['sigma_env_solid'],
            k_repel_dashed=self.potential_cfg['k_env_dashed'],
            sigma_dashed=self.potential_cfg['sigma_env_dashed'],
            use_lane_lines=self.potential_cfg['use_lane_lines'],
            ego_only=self.potential_cfg.get('env_loss_ego_only', True)
        )

        self.sparsity_loss = SparsityLoss(
            target_sparsity=self.sparsity_cfg['target_sparsity'],
            kl_weight=self.sparsity_cfg['kl_weight']
        )

    def set_phase(self, phase):
        '''Switch between Phase 1 (pre-training) and Phase 2 (fine-tuning).'''
        self.phase = phase

    def set_sparse_loss(self, enable):
        '''Enable or disable z_local sparsity loss.'''
        self.use_sparse_loss = enable

    def set_potential_loss(self, enable):
        '''Enable or disable potential-based repulsion loss.'''
        self.use_potential_loss = enable

    def _compute_auxiliary_losses(self, model, scene_graph, loss_out_dict):
        """
        Compute auxiliary losses from model's stored outputs (Redesign).

        (B) Sur Pred Loss: MSE between predicted sur delta and GT sur delta
        (C) Ego Pred Loss: MSE between predicted ego delta and GT ego delta (Phase 1 only)
        (D) Map Attn Guidance: KL between attn weight and GT future position (disabled for now — needs map_env)
        (A) Intent CE: CE between intent classification and GT acc/yaw class (Phase 2 only)

        :param model: TrafficPlannerModel with stored outputs
        :param scene_graph: scene graph with GT
        :param loss_out_dict: dict to store individual loss values
        :return: total auxiliary loss (scalar)
        """
        device = scene_graph.past.device
        aux_loss = torch.tensor(0.0, device=device)

        if model is None:
            return aux_loss

        gt_future = scene_graph.future_gt  # (NA, FT, 6)
        ego_mask = torch.zeros(gt_future.size(0), dtype=torch.bool, device=device)
        ego_inds = scene_graph.ptr[:-1]
        ego_mask[ego_inds] = True
        FT = gt_future.size(1)

        # ---- (B) Sur Pred Loss: ego_hist_ctx → predicted sur delta vs GT ----
        sur_pred_w = self.loss_weights.get('sur_pred', 0.0)
        if sur_pred_w > 0.0:
            sur_pred_list = model.get_sur_pred_outputs()
            if sur_pred_list is not None and len(sur_pred_list) > 0:
                sur_pred_loss = torch.tensor(0.0, device=device)
                count = 0
                for t, pred_delta in enumerate(sur_pred_list):
                    if t >= FT:
                        break
                    # GT sur delta: gt_future[sur_idx, t, :2] - gt_future[sur_idx, t-1, :2]
                    sur_gt_state = gt_future[~ego_mask]  # (num_sur, FT, 6)
                    if t == 0:
                        sur_prev = scene_graph.past[:, -1, :2][~ego_mask]  # (num_sur, 2)
                    else:
                        sur_prev = sur_gt_state[:, t - 1, :2]
                    gt_sur_delta = sur_gt_state[:, t, :2] - sur_prev  # (num_sur, 2)
                    # pred_delta is (num_ego, 2) — each ego predicts its paired sur's delta
                    # For simplicity, use mean across all sur agents if multiple
                    gt_delta_per_ego = gt_sur_delta.mean(dim=0, keepdim=True).expand_as(pred_delta)
                    sur_pred_loss = sur_pred_loss + nn.functional.mse_loss(pred_delta, gt_delta_per_ego)
                    count += 1
                if count > 0:
                    sur_pred_loss = sur_pred_loss / count
                    aux_loss = aux_loss + sur_pred_w * sur_pred_loss
                    loss_out_dict['sur_pred_loss'] = sur_pred_loss.detach().view((1,))

        # ---- (C) Ego Pred Loss: sur_hist_ctx → predicted ego delta (Phase 1 only) ----
        ego_pred_w = self.loss_weights.get('ego_pred', 0.0)
        if ego_pred_w > 0.0 and self.phase == 1:
            ego_pred_list = model.get_ego_pred_outputs()
            if ego_pred_list is not None and len(ego_pred_list) > 0:
                ego_pred_loss = torch.tensor(0.0, device=device)
                count = 0
                for t, pred_delta in enumerate(ego_pred_list):
                    if t >= FT:
                        break
                    ego_gt_state = gt_future[ego_mask]  # (num_ego, FT, 6)
                    if t == 0:
                        ego_prev = scene_graph.past[:, -1, :2][ego_mask]
                    else:
                        ego_prev = ego_gt_state[:, t - 1, :2]
                    gt_ego_delta = ego_gt_state[:, t, :2] - ego_prev
                    gt_delta_per_sur = gt_ego_delta.mean(dim=0, keepdim=True).expand_as(pred_delta)
                    ego_pred_loss = ego_pred_loss + nn.functional.mse_loss(pred_delta, gt_delta_per_sur)
                    count += 1
                if count > 0:
                    ego_pred_loss = ego_pred_loss / count
                    aux_loss = aux_loss + ego_pred_w * ego_pred_loss
                    loss_out_dict['ego_pred_loss'] = ego_pred_loss.detach().view((1,))

        # ---- (A) Intent CE Loss (Phase 2 only) ----
        intent_ce_w = self.loss_weights.get('intent_ce', 0.0)
        if intent_ce_w > 0.0 and self.phase == 2 and model is not None:
            z_local = model.get_z_local()  # (FT, num_ego, intent_dim)
            if z_local is not None and hasattr(model, 'intent_ce_head'):
                # Compute GT acc/yaw classes from GT future
                ego_gt = gt_future[ego_mask]  # (num_ego, FT, 6)
                # speed at each step, compute acceleration
                ego_speed = ego_gt[:, :, 4]  # (num_ego, FT)
                ego_prev_speed = torch.cat([scene_graph.past[:, -1, 4:5][ego_mask], ego_speed[:, :-1]], dim=1)
                ego_acc = (ego_speed - ego_prev_speed) / 0.5  # (num_ego, FT)
                ego_yaw_rate = ego_gt[:, :, 5]  # (num_ego, FT)

                acc_bins = self.aux_cfg['acc_bins']
                yaw_bins = self.aux_cfg['yaw_bins']

                # Bin: 0=decel, 1=maintain, 2=accel
                acc_class = torch.zeros_like(ego_acc, dtype=torch.long)
                acc_class[ego_acc > acc_bins[1]] = 2
                acc_class[(ego_acc >= acc_bins[0]) & (ego_acc <= acc_bins[1])] = 1

                # Bin: 0=left, 1=straight, 2=right
                yaw_class = torch.zeros_like(ego_yaw_rate, dtype=torch.long)
                yaw_class[ego_yaw_rate > yaw_bins[1]] = 2
                yaw_class[(ego_yaw_rate >= yaw_bins[0]) & (ego_yaw_rate <= yaw_bins[1])] = 1

                # Combined class: 3*acc + yaw → 9 classes
                gt_class = acc_class * 3 + yaw_class  # (num_ego, FT)

                intent_ce_loss = torch.tensor(0.0, device=device)
                num_steps = min(z_local.size(0), FT)
                for t in range(num_steps):
                    z_t = z_local[t]  # (num_ego, intent_dim)
                    logits = model.intent_ce_head(z_t)  # (num_ego, 9)
                    intent_ce_loss = intent_ce_loss + nn.functional.cross_entropy(logits, gt_class[:, t])

                if num_steps > 0:
                    intent_ce_loss = intent_ce_loss / num_steps
                    aux_loss = aux_loss + intent_ce_w * intent_ce_loss
                    loss_out_dict['intent_ce_loss'] = intent_ce_loss.detach().view((1,))

        # ---- (D) Map Attn Guidance Loss: attn_weights vs GT future position ----
        # Phase 1: ego + sur, Phase 2: ego only (sur frozen)
        map_attn_w = self.loss_weights.get('map_attn', 0.0)
        if map_attn_w > 0.0 and model is not None:
            map_attn_loss = self._compute_map_attn_guidance_loss(
                model, scene_graph, ego_mask, gt_future, FT)
            if map_attn_loss is not None:
                aux_loss = aux_loss + map_attn_w * map_attn_loss
                loss_out_dict['map_attn_loss'] = map_attn_loss.detach().view((1,))

        return aux_loss

    def _compute_map_attn_guidance_loss(self, model, scene_graph, ego_mask, gt_future, FT):
        """
        Map Attention Guidance: attention이 전방 4스텝 GT 위치의 map token에 집중하도록 KL loss.

        GT 위치 → ego/sur local frame → pixel → conv3 grid index → soft label
        soft label vs attn_weights → KL divergence

        :return: scalar loss or None
        """
        device = gt_future.device
        normalizer = self.state_normalizer

        ego_attn_list = model.get_ego_map_attn_weights()
        sur_attn_list = model.get_sur_map_attn_weights()

        if ego_attn_list is None or len(ego_attn_list) == 0:
            return None

        map_gt_steps = self.aux_cfg['map_gt_steps']
        map_gt_weights = self.aux_cfg['map_gt_weights']
        grid_size = model.map_token_spatial  # 29
        num_tokens = grid_size * grid_size  # 841

        # map_obs_bounds: [low_l, low_w, high_l, high_w] in meters
        bounds = [-17.0, -38.5, 60.0, 38.5]
        pix_size = model.map_obs_size_pix  # 256
        # meters per pixel
        m2pix_l = pix_size / (bounds[2] - bounds[0])  # 256 / 77
        m2pix_w = pix_size / (bounds[3] - bounds[1])  # 256 / 77
        # pixel to grid (conv3 output)
        pix2grid_l = grid_size / pix_size  # 29 / 256
        pix2grid_w = grid_size / pix_size

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        num_steps = min(len(ego_attn_list), FT)

        for t in range(num_steps):
            remaining = min(map_gt_steps, FT - t - 1)
            if remaining <= 0:
                continue

            # --- Ego map attn guidance ---
            ego_attn_w = ego_attn_list[t]  # (num_ego, 1, num_tokens)
            ego_attn_dist = ego_attn_w.squeeze(1)  # (num_ego, num_tokens)

            # Current ego position (frame for local transform)
            if t == 0:
                ego_frame = scene_graph.past[:, -1, :4][ego_mask]  # (num_ego, 4) normalized
            else:
                ego_frame = gt_future[ego_mask][:, t - 1, :4]

            ego_soft_label = self._make_soft_label(
                ego_frame, gt_future[ego_mask], t, remaining,
                map_gt_weights, normalizer, bounds, m2pix_l, m2pix_w,
                pix2grid_l, pix2grid_w, grid_size, num_tokens, device)

            if ego_soft_label is not None:
                # KL(soft_label || attn_dist) — soft_label is target
                ego_kl = self._kl_with_epsilon(ego_soft_label, ego_attn_dist)
                total_loss = total_loss + ego_kl
                count += 1

            # --- Sur map attn guidance (Phase 1 only) ---
            if self.phase == 1 and sur_attn_list is not None and t < len(sur_attn_list):
                sur_attn_w = sur_attn_list[t]  # (num_sur, 1, num_tokens)
                sur_attn_dist = sur_attn_w.squeeze(1)

                if t == 0:
                    sur_frame = scene_graph.past[:, -1, :4][~ego_mask]
                else:
                    sur_frame = gt_future[~ego_mask][:, t - 1, :4]

                sur_soft_label = self._make_soft_label(
                    sur_frame, gt_future[~ego_mask], t, remaining,
                    map_gt_weights, normalizer, bounds, m2pix_l, m2pix_w,
                    pix2grid_l, pix2grid_w, grid_size, num_tokens, device)

                if sur_soft_label is not None:
                    sur_kl = self._kl_with_epsilon(sur_soft_label, sur_attn_dist)
                    total_loss = total_loss + sur_kl
                    count += 1

        if count > 0:
            return total_loss / count
        return None

    def _make_soft_label(self, frame, gt_future_agent, t, remaining,
                         weights, normalizer, bounds, m2pix_l, m2pix_w,
                         pix2grid_l, pix2grid_w, grid_size, num_tokens, device):
        """
        GT future 위치 → agent local frame → grid index → soft label 생성.

        :param frame: (N, 4) current agent position (normalized, x,y,hx,hy)
        :param gt_future_agent: (N, FT, 6) GT future for these agents (normalized)
        :param t: current timestep
        :param remaining: number of future steps to look ahead (1~4)
        :return: (N, num_tokens) soft label or None
        """
        N = frame.size(0)
        if N == 0:
            return None

        # Unnormalize frame and future positions
        frame_unnorm = normalizer.unnormalize(frame)  # (N, 4)

        soft_label = torch.zeros(N, num_tokens, device=device)

        for k in range(remaining):
            future_t = t + 1 + k
            if future_t >= gt_future_agent.size(1):
                break

            gt_pos = gt_future_agent[:, future_t, :4]  # (N, 4) normalized
            gt_pos_unnorm = normalizer.unnormalize(gt_pos)  # (N, 4)

            # Global → agent local frame: transform2frame expects (B, N, 4)
            local_pos = transform2frame(
                frame_unnorm, gt_pos_unnorm.unsqueeze(1))  # (N, 1, 4)
            local_xy = local_pos[:, 0, :2]  # (N, 2) — local (l, w)

            # Local meters → pixel coordinates
            pix_l = (local_xy[:, 0] - bounds[0]) * m2pix_l  # offset + scale
            pix_w = (local_xy[:, 1] - bounds[1]) * m2pix_w

            # Pixel → conv3 grid index
            gi = (pix_l * pix2grid_l).long().clamp(0, grid_size - 1)
            gj = (pix_w * pix2grid_w).long().clamp(0, grid_size - 1)

            # Flatten index
            flat_idx = gi * grid_size + gj  # (N,)

            # Add weighted contribution
            w = weights[k]
            soft_label.scatter_add_(1, flat_idx.unsqueeze(1), torch.full((N, 1), w, device=device))

        # Normalize to probability distribution
        label_sum = soft_label.sum(dim=1, keepdim=True)
        if (label_sum == 0).any():
            return None
        soft_label = soft_label / label_sum

        return soft_label

    def _kl_with_epsilon(self, target, pred, eps=1e-8):
        """KL(target || pred) with epsilon smoothing to avoid log(0)."""
        pred_smooth = pred + eps
        pred_smooth = pred_smooth / pred_smooth.sum(dim=-1, keepdim=True)
        target_smooth = target + eps
        target_smooth = target_smooth / target_smooth.sum(dim=-1, keepdim=True)
        kl = (target_smooth * (target_smooth.log() - pred_smooth.log())).sum(dim=-1)
        return kl.mean()

    def forward(self, scene_graph, pred,
                 map_idx=None,
                 map_env=None,
                 model=None,
                 use_teacher_forcing=False):
        '''
        Computes loss (Redesign — with auxiliary losses).

        :param scene_graph: containing input and GT data
        :param pred: dict of model predictions.
        :param model: TrafficPlannerModel instance (for auxiliary loss computation)
        :param use_teacher_forcing: if True, compute loss over TF segments
        '''

        # Handle teacher forcing mode
        if use_teacher_forcing:
            return self._forward_teacher_forcing(scene_graph, pred, map_idx, map_env, model)

        # reconstruction loss
        gt_future = scene_graph.future_gt
        pred_future = pred['future_pred']

        if self.ego_only_recon:
            ego_inds = scene_graph.ptr[:-1]
            gt_future_ego = gt_future[ego_inds]
            pred_future_ego = pred_future[ego_inds]
            future_vis_ego = scene_graph.future_vis[ego_inds]
            gt_future_valid = gt_future_ego[future_vis_ego == 1.0]
            pred_future_valid = pred_future_ego[future_vis_ego == 1.0]
        else:
            gt_future_valid = gt_future[scene_graph.future_vis == 1.0]
            pred_future_valid = pred_future[scene_graph.future_vis == 1.0]

        recon_loss = -log_normal(pred_future_valid, gt_future_valid[:, :4], torch.ones_like(pred_future_valid))

        # KL divergence loss
        pm, pv = pred['prior_out']
        qm, qv = pred['posterior_out']
        kl_loss = kl_normal(qm, qv, pm, pv)

        # total weighted loss
        loss = self.loss_weights['recon'] * recon_loss.mean()

        # KL loss: Phase 1 only
        if self.phase == 1 and self.loss_weights.get('kl', 0.0) > 0.0:
            loss = loss + self.loss_weights['kl'] * kl_loss.mean()

        # z_local sparsity loss
        sparse_loss = None
        if self.use_sparse_loss and self.loss_weights.get('sparse', 0.0) > 0.0 and model is not None:
            z_local = model.get_z_local()
            if z_local is not None:
                sparse_loss = self.sparsity_loss(z_local)
                loss = loss + self.loss_weights['sparse'] * sparse_loss

        # Potential-based losses
        potential_veh_loss = None
        if self.use_potential_loss and self.use_veh_potential and self.loss_weights.get('potential_veh', 0.0) > 0.0:
            potential_veh_loss = self.veh_potential_loss(pred_future, scene_graph)
            loss = loss + self.loss_weights['potential_veh'] * potential_veh_loss

        potential_env_loss = None
        if self.use_potential_loss and self.loss_weights.get('potential_env', 0.0) > 0.0:
            if map_idx is not None and map_env is not None:
                potential_env_loss = self.env_potential_loss(pred_future, scene_graph, map_idx, map_env)
                loss = loss + self.loss_weights['potential_env'] * potential_env_loss

        prior_coll_loss = None
        if self.loss_weights.get('coll_veh_prior', 0.0) > 0.0:
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collision loss!')
                exit()
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            veh_coll_loss = VehCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
            if self.loss_weights['coll_veh_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'])
                prior_coll_pens, na_sqr = veh_coll_loss(prior_traj)
                prior_coll_loss = torch.sum(prior_coll_pens) / na_sqr
                loss = loss + self.loss_weights['coll_veh_prior'] * prior_coll_loss

        prior_coll_env_loss = None
        if self.loss_weights.get('coll_env_prior', 0.0) > 0.0:
            assert(map_idx is not None and map_env is not None)
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collision loss!')
                exit()
            ego_inds = scene_graph.ptr[:-1]
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw[ego_inds])
            env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, pred['future_pred'].size(1))
            if self.loss_weights['coll_env_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'][ego_inds])
                prior_coll_env_loss = env_coll_loss(prior_traj)
                loss = loss + self.loss_weights['coll_env_prior'] * prior_coll_env_loss.mean()

        loss_out = {
            'loss': loss.view((1,)),
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

        # Auxiliary losses (Redesign)
        aux_loss = self._compute_auxiliary_losses(model, scene_graph, loss_out)
        if aux_loss.item() > 0:
            loss = loss + aux_loss
            loss_out['loss'] = loss.view((1,))

        if sparse_loss is not None:
            loss_out['sparse_loss'] = sparse_loss.view((1,))
        if potential_veh_loss is not None:
            loss_out['potential_veh_loss'] = potential_veh_loss.view((1,))
        if potential_env_loss is not None:
            loss_out['potential_env_loss'] = potential_env_loss.view((1,))
        if prior_coll_loss is not None:
            loss_out['coll_veh_prior'] = prior_coll_loss.view((1,))
        if prior_coll_env_loss is not None:
            loss_out['coll_env_prior'] = prior_coll_env_loss.view(-1)

        return loss_out

    def _forward_teacher_forcing(self, scene_graph, pred, map_idx, map_env, model):
        '''
        Compute loss for teacher forcing mode.

        Each segment has up to tf_segment_len predictions starting from GT-initialized state.
        Loss is computed for each segment and summed.

        :param pred: dict with 'future_pred' as list of 12 segments
            - all_segment_preds[seg_idx] = [pred_t, pred_t+1, ...] each (NA, 4)
        '''
        all_segment_preds = pred['future_pred']  # List of 12 segments
        gt_future = scene_graph.future_gt  # (NA, FT, 6)
        FT = gt_future.size(1)
        NA = gt_future.size(0)
        device = gt_future.device

        total_recon_loss = torch.tensor(0.0, device=device)
        total_potential_veh_loss = torch.tensor(0.0, device=device)
        total_potential_env_loss = torch.tensor(0.0, device=device)
        num_segments = len(all_segment_preds)
        num_valid_segments = 0

        for seg_idx, segment_preds in enumerate(all_segment_preds):
            if len(segment_preds) == 0:
                continue

            # Stack segment predictions: list of (NA, 4) -> (NA, seg_len, 4)
            seg_traj = torch.stack(segment_preds, dim=1)
            seg_len = seg_traj.size(1)

            # Get corresponding GT segment
            gt_start = seg_idx
            gt_end = min(seg_idx + seg_len, FT)
            actual_len = gt_end - gt_start

            if actual_len <= 0:
                continue

            # Trim predictions if needed
            seg_traj = seg_traj[:, :actual_len, :]
            gt_seg = gt_future[:, gt_start:gt_end, :4]  # (NA, actual_len, 4)

            # Only compute loss for valid timesteps
            future_vis_seg = scene_graph.future_vis[:, gt_start:gt_end]  # (NA, actual_len)
            valid_mask = future_vis_seg == 1.0

            if valid_mask.sum() == 0:
                continue

            # Reconstruction loss (MSE)
            if self.ego_only_recon:
                ego_inds = scene_graph.ptr[:-1]
                ego_mask = valid_mask[ego_inds]  # (B, actual_len)
                pred_valid = seg_traj[ego_inds][ego_mask]
                gt_valid = gt_seg[ego_inds][ego_mask]
            else:
                pred_valid = seg_traj[valid_mask]
                gt_valid = gt_seg[valid_mask]
            seg_recon_loss = -log_normal(pred_valid, gt_valid, torch.ones_like(pred_valid))
            total_recon_loss = total_recon_loss + seg_recon_loss.mean()

            # Potential-based loss (vehicle repulsion) - ego only
            if self.use_potential_loss and self.use_veh_potential and self.loss_weights.get('potential_veh', 0.0) > 0.0:
                seg_potential_veh = self.veh_potential_loss(seg_traj, scene_graph)
                total_potential_veh_loss = total_potential_veh_loss + seg_potential_veh

            # Potential-based loss (environment boundary) - ego only
            if self.use_potential_loss and self.loss_weights.get('potential_env', 0.0) > 0.0:
                if map_idx is not None and map_env is not None:
                    seg_potential_env = self.env_potential_loss(seg_traj, scene_graph, map_idx, map_env)
                    total_potential_env_loss = total_potential_env_loss + seg_potential_env

            num_valid_segments += 1

        # Average over segments
        if num_valid_segments > 0:
            total_recon_loss = total_recon_loss / num_valid_segments
            total_potential_veh_loss = total_potential_veh_loss / num_valid_segments
            total_potential_env_loss = total_potential_env_loss / num_valid_segments

        # KL loss (still computed on z_global)
        pm, pv = pred['prior_out']
        qm, qv = pred['posterior_out']
        kl_loss = kl_normal(qm, qv, pm, pv)

        # Total weighted loss
        loss = self.loss_weights['recon'] * total_recon_loss

        # KL loss: only in Phase 1
        if self.phase == 1 and self.loss_weights.get('kl', 0.0) > 0.0:
            loss = loss + self.loss_weights['kl'] * kl_loss.mean()

        # z_local sparsity loss
        sparse_loss = None
        if self.use_sparse_loss and self.loss_weights.get('sparse', 0.0) > 0.0 and model is not None:
            z_local = model.get_z_local()
            if z_local is not None:
                sparse_loss = self.sparsity_loss(z_local)
                loss = loss + self.loss_weights['sparse'] * sparse_loss

        # Potential losses
        if self.use_potential_loss and self.loss_weights.get('potential_veh', 0.0) > 0.0:
            loss = loss + self.loss_weights['potential_veh'] * total_potential_veh_loss

        if self.use_potential_loss and self.loss_weights.get('potential_env', 0.0) > 0.0:
            loss = loss + self.loss_weights['potential_env'] * total_potential_env_loss

        loss_out = {
            'loss': loss.view((1,)),
            'recon_loss': total_recon_loss.view((1,)),
            'kl_loss': kl_loss,
            'num_segments': torch.tensor([num_valid_segments], device=device, dtype=torch.float32),
        }

        # Auxiliary losses (Redesign)
        aux_loss = self._compute_auxiliary_losses(model, scene_graph, loss_out)
        if aux_loss.item() > 0:
            loss = loss + aux_loss
            loss_out['loss'] = loss.view((1,))

        if sparse_loss is not None:
            loss_out['sparse_loss'] = sparse_loss.view((1,))
        if self.use_potential_loss and self.loss_weights.get('potential_veh', 0.0) > 0.0:
            loss_out['potential_veh_loss'] = total_potential_veh_loss.view((1,))
        if self.use_potential_loss and self.loss_weights.get('potential_env', 0.0) > 0.0:
            loss_out['potential_env_loss'] = total_potential_env_loss.view((1,))

        return loss_out

    def compute_err(self, scene_graph, pred, normalizer):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

        # Skip error computation for teacher forcing mode (pred_future is a list of segments)
        if isinstance(pred_future, list):
            return {}

        NA, FT, _ = gt_future.size()

        gt_future = normalizer.unnormalize(gt_future)
        pred_future = normalizer.unnormalize(pred_future)
        # print("gtfuture")
        # print(gt_future)
        # print("pred_future")
        # print(pred_future)
        # only want to compute errors for timesteps we have GT data
        gt_future = gt_future[scene_graph.future_vis == 1.0]
        pred_future = pred_future[scene_graph.future_vis == 1.0]

        # positional distance error
        gt_pos = gt_future[:,:2]
        pred_pos = pred_future[:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)
        # angle distance error
        gt_h = gt_future[:,2:4]
        gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
        pred_h = pred_future[:,2:4]
        pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
        dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
        ang_diff = torch.acos(dotprod)
        ang_err = torch.rad2deg(ang_diff)

        # NLL of posterior mean under the prior
        post_mean = pred['posterior_out'][0]
        z_logprob = log_normal(post_mean, pred['prior_out'][0], pred['prior_out'][1])
        z_mdist =  torch.norm((post_mean - pred['prior_out'][0]) / torch.sqrt(pred['prior_out'][1]), dim=-1)

        err_out = {
            'pos_err' : pos_err, # (num_valid_frames, )
            'ang_err' : ang_err, # (num_valid_frames, )
            'z_logprob' : z_logprob, # NA
            'z_mdist' : z_mdist
        }

        return err_out

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att, batch, ptr,
                       num_circ=5,
                       buffer_dist=0.0):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param batch: from the scene graph
        :param ptr: from the scene_graph
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.batch = batch
        self.ptr = ptr

        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.num_pairs = torch.sum(self.graph_sizes*self.graph_sizes - self.graph_sizes)

        NA = self.veh_att.size(0)
        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2. # radius of the discs for each vehicle assuming length > width
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist
        # need a mask to ignore "self" collisions and "collisions" from other scene graphs in the batch
        off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True

        self.valid_mask = torch.logical_and(off_diag_mask, batch_mask)

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss, number of "interactions" that could have caused a collision, i.e. number of valid vehicle pairs
        '''
        NA, T, _ = traj.size()
        cur_valid_mask = self.valid_mask.view(1, NA, NA).expand(T, NA, NA)

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_cent = self.centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        
        
        # pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)
        
#-----------------------------------------------------HJ changed------------------------------------------------
        
        pair_dists = torch.cdist(
            cur_cent1.contiguous(),
            cur_cent2.contiguous()
        ).view(T*NA*NA, self.num_circ*self.num_circ)
        
#-----------------------------------------------------HJ changed------------------------------------------------
       
        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)
        cur_penalty_dists = self.penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        is_colliding_mask = torch.logical_and(is_colliding_mask,cur_valid_mask)
        # compute penalties
        cur_penalties = torch.where(is_colliding_mask, 1.0 - (min_pair_dists / cur_penalty_dists), torch.zeros_like(cur_penalty_dists))
        cur_penalties = cur_penalties[cur_valid_mask]

        return cur_penalties, self.num_pairs

class EnvCollLoss(nn.Module):
    '''
    Penalizes overlap with non-drivable area.
    '''
    def __init__(self, veh_att, mapixes, map_env, T):
        '''
        :param veh_att: (NA, 2) UNNORMALIZED
        :param mapixes: (NA, )
        :param map_env: 
        :param T: number of steps in trajectories that loss will be computed on
        '''
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        # loss will be applied on all timesteps, so update info accordingly
        NA = veh_att.size(0)
        assert(NA == mapixes.size(0))
        self.mapixes = mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.penalty_dists = torch.sqrt((veh_att[:, 0]**2 / 4.0) + (veh_att[:, 1]**2 / 4.0)) # max dist from center to corner of each vehicle
        self.penalty_dists = self.penalty_dists.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.veh_att = veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)
        self.T = T
        

    def forward(self, traj):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss
        '''
        NA = traj.size(0)
        T = self.T
        assert(T == traj.size(1))
        assert(NA*T == self.veh_att.size(0))
        traj = traj.view(NA*T, 4)

        all_penalties = torch.zeros((NA*T)).to(traj.device)
        # get collisions w/ non-drivable (first layer)
        drivable_raster = self.map_env.nusc_raster[:, 0]
        coll_pt = nutils.get_coll_point(drivable_raster,
                                        self.map_env.nusc_dx,
                                        traj.detach(),
                                        self.veh_att,
                                        self.mapixes)
        valid = ~torch.isnan(torch.sum(coll_pt, axis=1))
        if torch.sum(valid) == 0:
            return all_penalties.view(NA, T)

        # compute penalties
        traj_cent = traj[:,:2][valid]
        cur_dists = torch.norm(traj_cent - coll_pt[valid], dim=1)
        cur_pen_dists = self.penalty_dists[valid]
        val_penalties = 1.0 - (cur_dists / cur_pen_dists)
        # return a penalty for every time step, if not colliding is just 0
        all_penalties[valid] = val_penalties

        return all_penalties.view(NA, T)
    
def compute_disp_err(scene_graph, pred, normalizer):
    '''
    Computes sample-based displacement errors.

    ONLY computes for ego vehicle since have guaranteed full past/future motion so the statistics
    will be correct.
    '''
    gt_future = scene_graph.future_gt # NA x FT x 6
    pred_future = pred['future_pred'] # NA x NS x FT x 4

    NA, FT, _ = gt_future.size()
    NS = pred_future.size(1)

    # make sure same length
    FT = pred_future.size(2) if pred_future.size(2) < FT else FT
    pred_future = pred_future[:, :, :FT] # if prediction is longer, make sure only compare the steps we have
    gt_future = gt_future[:, :FT]

    gt_future = normalizer.unnormalize(gt_future).view(NA, 1, FT, 6)
    pred_future = normalizer.unnormalize(pred_future)

    # find index of first agent in each batch
    ego_inds = scene_graph.ptr[:-1]

    # ego-only data
    gt_future = gt_future[ego_inds] # B x 1 x FT x 6
    pred_future = pred_future[ego_inds] # B x NS x FT x 4
    B = gt_future.size(0)

    # positional ADE
    gt_pos = gt_future[:,:,:,:2]
    pred_pos = pred_future[:,:,:,:2]
    diff = torch.norm(gt_pos - pred_pos, dim=-1) # B x NS x FT
    ade = diff.mean(dim=-1) # B x NS
    min_ade = torch.min(ade, dim=1)[0] # B

    # positional APD
    pred_pairwise_pos = pred_pos.view(B, NS, 1, FT, 2).expand(B, NS, NS, FT, 2)
    pairwise_diff = torch.norm(pred_pairwise_pos - pred_pairwise_pos.transpose(1, 2), dim=-1) # B x NS x NS x FT
    all_sum = torch.sum(pairwise_diff, dim=[1, 2]).sum(dim=-1)
    apd = all_sum / (NS*(NS-1)*FT) # don't want to include diagonal elmnts

    # positional FDE
    fde = diff[:,:,-1]
    min_fde = torch.min(fde, dim=1)[0] # B 

    # angular ADE
    gt_h = gt_future[:,:,:,2:4]
    gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
    pred_h = pred_future[:,:,:,2:4]
    pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
    dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
    ang_diff = torch.rad2deg(torch.acos(dotprod)) # B x NS x FT
    ang_ade = ang_diff.mean(dim=-1)
    ang_min_ade = torch.min(ang_ade, dim=1)[0] # B

    # angular FDE
    ang_fde = ang_diff[:,:,-1]
    ang_min_fde = torch.min(ang_fde, dim=1)[0]

    disp_err_dict = {
        'pos_minADE' : min_ade,
        'pos_minFDE' : min_fde,
        'ang_minADE' : ang_min_ade,
        'ang_minFDE' : ang_min_fde,
        'APD' : apd
    }
    return disp_err_dict

def compute_coll_rate_env(scene_graph, map_idx, pred, map_env, state_normalizer, att_normalizer,
                        ego_only=False):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from datasets.utils import get_ego_inds

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw
    mapixes = map_idx[scene_graph.batch]

    if ego_only:
        ego_inds = get_ego_inds(scene_graph)
        pred_future = pred_future[ego_inds]
        veh_att = veh_att[ego_inds]
        mapixes = mapixes[ego_inds]
        NA = pred_future.size(0)

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future).view(NA*NS*FT, 4)
    veh_att = att_normalizer.unnormalize(veh_att).view(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    # check if on drivable (first layer)
    drivable_raster = map_env.nusc_raster[:, 0]
    mapixes = mapixes.view(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    # don't check for nan futures
    valid_frames = ~torch.isnan(pred_future.sum(-1))
    drivable_frac = nutils.check_on_layer(drivable_raster,
                                            map_env.nusc_dx,
                                            pred_future[valid_frames],
                                            veh_att[valid_frames],
                                            mapixes[valid_frames])
    final_drivable_frac = torch.ones(NA*NS*FT).to(drivable_frac) # by default, nan values are not considered colliding
    final_drivable_frac[valid_frames] = drivable_frac
    final_drivable_frac = final_drivable_frac.view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_env_from_traj(pred_future, veh_att, mapixes, map_env):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    :param pred_future: NA x NS x FT x 4 UNNORMALIZED
    :param veh_att:
    :param mapixes: (NA) index of the map for each agent
    :param map_env:

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils

    NA, NS, FT, _ = pred_future.size()

    # unnorm preds and attribs
    pred_future = pred_future.reshape(NA*NS*FT, 4)
    veh_att = veh_att.reshape(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    # check if on drivable (first layer)
    drivable_raster = map_env.nusc_raster[:, 0]
    mapixes = mapixes.reshape(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    # don't check for nan futures
    valid_frames = ~torch.isnan(pred_future.sum(-1))
    drivable_frac = nutils.check_on_layer(drivable_raster,
                                            map_env.nusc_dx,
                                            pred_future[valid_frames],
                                            veh_att[valid_frames],
                                            mapixes[valid_frames])
    final_drivable_frac = torch.ones(NA*NS*FT).to(drivable_frac) # by default, nan values are not considered colliding
    final_drivable_frac[valid_frames] = drivable_frac
    final_drivable_frac = final_drivable_frac.view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_veh(scene_graph, pred, state_normalizer, att_normalizer):
    '''
    Computes number of rollouts that collided with other agents for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    WARNING: this function assumes the scene graph edges connect all vehicle pairs that
    need to be checked for collisions. Also assumes all edges are bidirectional, i.e.
    if (3, 5) is a pair then (5, 3) is also one. We ONLY check one of the two.

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future)
    veh_att = att_normalizer.unnormalize(veh_att)

    # all the vehicle pairs to go over
    pred_future = pred_future.cpu().numpy()
    veh_att = veh_att.cpu().numpy()
    pairs = scene_graph.edge_index.cpu().numpy().T

    veh_coll = np.zeros((NA, NS), dtype=np.bool_)
    poly_cache = dict()    
    # loop over every timestep in every sample for this combination
    coll_count = 0
    for s in range(NS):
        for veh_pair in pairs:
            aj, ai = veh_pair
            if aj <= ai:
                continue # don't double count
            if veh_coll[ai, s]:
                continue # already determined there has been a collision for this agent at this sample, move on
            for t in range(FT):
                # compute iou
                if (ai, s, t) not in poly_cache:
                    ai_state = pred_future[ai, s, t, :]
                    if np.sum(np.isnan(ai_state)) > 0:
                        poly_cache[(ai, s, t)] = None
                        continue # don't have data for this step
                    ai_corners = nutils.get_corners(ai_state, veh_att[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, s, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, s, t)]
                    if ai_poly is None:
                        continue
                if (aj, s, t) not in poly_cache:
                    aj_state = pred_future[aj, s, t, :]
                    if np.sum(np.isnan(aj_state)) > 0:
                        poly_cache[(aj, s, t)] = None
                        continue # don't have data for this step
                    aj_corners = nutils.get_corners(aj_state, veh_att[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, s, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, s, t)]
                    if aj_poly is None:
                        continue
                cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                if cur_iou > VEH_COLL_THRESH:
                    coll_count += 1
                    veh_coll[ai, s] = True
                    break # don't need to check rest of sequence

    coll_dict = {
        'num_coll_veh' : float(coll_count),
        'num_traj_veh' : float(NS*NA),
        'did_collide' : veh_coll
    }

    return coll_dict