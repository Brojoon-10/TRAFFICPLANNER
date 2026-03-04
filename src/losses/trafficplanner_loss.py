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
from datasets.utils import NUSC_NORM_STATS, CARLA_NORM_STATS

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
                    aux_cfg=None,
                    recon_pos_weight=1.0,
                    kl_free_bits=0.0):
        """
        :param loss_weights: dict of weightings for loss terms
        :param aux_cfg: dict of auxiliary loss config (optional)
            - sur_pred_dim: 2 (dx, dy)
            - map_gt_steps: 6
            - map_gt_decay_lambda: 0.3 (exponential decay rate for GT step weights)
            - acc_bins: [-1.0, 1.0]  (3 bins: decel, maintain, accel)
            - yaw_bins: [-0.1, 0.1]  (3 bins: left, straight, right)
        :param kl_free_bits: per-dim free nats for KL (0=off)
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
        self.kl_free_bits = kl_free_bits
        self.recon_pos_weight = recon_pos_weight

        # Auxiliary loss config
        default_aux_cfg = {
            'sur_pred_dim': 2,
            'map_gt_steps': 6,
            'map_gt_decay_lambda': 0.3,  # exponential decay: w_t = exp(-λt) / Σexp(-λt)
            'intent_sigma': 0.5,         # Gaussian sigma for intent soft label
            'intent_range': 2.0,         # prototype grid range in std-scaled units
            'map_gauss_sigma_d': 0.8,    # lateral Gaussian sigma (grid cells, ~2.1m)
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
        Compute auxiliary losses from model's stored outputs (Transformer Decoder).

        Outputs can be:
        - Training: tensor (FT, num_agent, dim) from parallel forward
        - Inference: list of FT tensors from AR forward

        (B) Sur Pred Loss: ego A2A output → predicted sur delta vs GT
        (C) Ego Pred Loss: sur A2A output → predicted ego delta vs GT (Phase 1 only)
        (D) Map Attn Guidance: attn weight vs GT future position KL
        (A) Intent CE: z_local → soft label KL (Phase 2 only)

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

        # ---- (B) Sur Pred Loss: ego A2A output → predicted sur delta vs GT ----
        sur_pred_w = self.loss_weights.get('sur_pred', 0.0)
        if sur_pred_w > 0.0:
            sur_pred_raw = model.get_sur_pred_outputs()
            if sur_pred_raw is not None:
                sur_gt_state = gt_future[~ego_mask]  # (num_sur, FT, 6)

                # Normalize to tensor: (FT, num_ego, 2)
                if isinstance(sur_pred_raw, torch.Tensor):
                    sur_pred_tensor = sur_pred_raw  # already (FT, num_ego, 2)
                else:
                    sur_pred_tensor = torch.stack(sur_pred_raw, dim=0)

                sur_pred_loss = torch.tensor(0.0, device=device)
                num_steps = min(sur_pred_tensor.size(0), FT)
                for t in range(num_steps):
                    if t == 0:
                        sur_prev = scene_graph.past[:, -1, :2][~ego_mask]
                    else:
                        sur_prev = sur_gt_state[:, t - 1, :2]
                    gt_sur_delta = sur_gt_state[:, t, :2] - sur_prev
                    pred_delta = sur_pred_tensor[t]  # (num_ego, 2)
                    gt_delta_per_ego = gt_sur_delta.mean(dim=0, keepdim=True).expand_as(pred_delta)
                    sur_pred_loss = sur_pred_loss + nn.functional.mse_loss(pred_delta, gt_delta_per_ego)

                if num_steps > 0:
                    sur_pred_loss = sur_pred_loss / num_steps
                    aux_loss = aux_loss + sur_pred_w * sur_pred_loss
                    loss_out_dict['sur_pred_loss'] = sur_pred_loss.detach().view((1,))

        # ---- (C) Ego Pred Loss: sur A2A output → predicted ego delta (Phase 1 only) ----
        ego_pred_w = self.loss_weights.get('ego_pred', 0.0)
        if ego_pred_w > 0.0 and self.phase == 1:
            ego_pred_raw = model.get_ego_pred_outputs()
            if ego_pred_raw is not None:
                ego_gt_state = gt_future[ego_mask]  # (num_ego, FT, 6)

                if isinstance(ego_pred_raw, torch.Tensor):
                    ego_pred_tensor = ego_pred_raw
                else:
                    ego_pred_tensor = torch.stack(ego_pred_raw, dim=0)

                ego_pred_loss = torch.tensor(0.0, device=device)
                num_steps = min(ego_pred_tensor.size(0), FT)
                for t in range(num_steps):
                    if t == 0:
                        ego_prev = scene_graph.past[:, -1, :2][ego_mask]
                    else:
                        ego_prev = ego_gt_state[:, t - 1, :2]
                    gt_ego_delta = ego_gt_state[:, t, :2] - ego_prev
                    pred_delta = ego_pred_tensor[t]  # (num_sur, 2)
                    gt_delta_per_sur = gt_ego_delta.mean(dim=0, keepdim=True).expand_as(pred_delta)
                    ego_pred_loss = ego_pred_loss + nn.functional.mse_loss(pred_delta, gt_delta_per_sur)

                if num_steps > 0:
                    ego_pred_loss = ego_pred_loss / num_steps
                    aux_loss = aux_loss + ego_pred_w * ego_pred_loss
                    loss_out_dict['ego_pred_loss'] = ego_pred_loss.detach().view((1,))

        # ---- (A) Intent Soft Label Loss (Phase 2 only) ----
        # 9 prototypes in normalized (acc, yaw) space: 3x3 grid
        # Soft label via Gaussian distance, loss via KL divergence
        intent_ce_w = self.loss_weights.get('intent_ce', 0.0)
        if intent_ce_w > 0.0 and model is not None:
            z_local_raw = model.get_z_local()
            if z_local_raw is not None and hasattr(model, 'intent_ce_head'):
                num_intents = model.num_intents
                n_acc = int(np.sqrt(num_intents))
                n_yaw = num_intents // n_acc
                intent_range = self.aux_cfg.get('intent_range', 1.0)
                acc_vals = torch.linspace(-intent_range, intent_range, n_acc, device=device)
                yaw_vals = torch.linspace(-intent_range, intent_range, n_yaw, device=device)
                prototypes = torch.stack(torch.meshgrid(acc_vals, yaw_vals), dim=-1).reshape(-1, 2)

                sigma = self.aux_cfg['intent_sigma']

                ego_gt = gt_future[ego_mask]  # (num_ego, FT, 6)
                normalizer = model.get_normalizer()
                s_mean, s_std = normalizer.mean_vals[4].to(device), normalizer.std_vals[4].to(device)
                dt = model.dt

                ego_speed_raw = ego_gt[:, :, 4] * s_std + s_mean
                past_speed_norm = scene_graph.past[:, -1, 4:5][ego_mask]
                ego_prev_speed_raw = torch.cat([past_speed_norm * s_std + s_mean, ego_speed_raw[:, :-1]], dim=1)
                raw_acc = (ego_speed_raw - ego_prev_speed_raw) / dt
                # ninfo = NUSC_NORM_STATS[('car', 'truck')]
                ninfo = CARLA_NORM_STATS[('car', 'truck')]
                a_std = ninfo['a'][1]
                ego_acc = raw_acc / a_std

                hdot_mean = normalizer.mean_vals[5].to(device)
                hdot_std = normalizer.std_vals[5].to(device)
                raw_yaw_rate = ego_gt[:, :, 5] * hdot_std + hdot_mean
                ego_yaw_rate = raw_yaw_rate / ninfo['hdot'][1]

                gt_xy = torch.stack([ego_acc, ego_yaw_rate], dim=-1)  # (num_ego, FT, 2)
                dist_sq = ((gt_xy.unsqueeze(-2) - prototypes.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
                gt_soft_label = torch.softmax(-dist_sq / (2 * sigma ** 2), dim=-1)  # (num_ego, FT, num_intents)

                # Normalize z_local to tensor: (FT, num_ego, intent_dim)
                if isinstance(z_local_raw, torch.Tensor):
                    z_local_tensor = z_local_raw  # already (FT, num_ego, intent_dim)
                else:
                    z_local_tensor = torch.stack(z_local_raw, dim=0)

                intent_loss = torch.tensor(0.0, device=device)
                num_steps = min(z_local_tensor.size(0), FT)
                for t in range(num_steps):
                    z_t = z_local_tensor[t]  # (num_ego, intent_dim)
                    logits = model.intent_ce_head(z_t)  # (num_ego, num_intents)
                    log_pred = nn.functional.log_softmax(logits, dim=-1)
                    target = gt_soft_label[:, t, :]
                    intent_loss = intent_loss + nn.functional.kl_div(log_pred, target, reduction='batchmean')

                if num_steps > 0:
                    intent_loss = intent_loss / num_steps
                    aux_loss = aux_loss + intent_ce_w * intent_loss
                    loss_out_dict['intent_ce_loss'] = intent_loss.detach().view((1,))

        # ---- (D) Map Attn Guidance Loss: attn_weights vs GT future position ----
        # Phase 1: ego + sur, Phase 2: ego only (sur frozen)
        map_attn_w = self.loss_weights.get('map_attn', 0.0)
        if map_attn_w > 0.0 and model is not None:
            map_attn_loss = self._compute_map_attn_guidance_loss(
                model, scene_graph, ego_mask, gt_future, FT)
            if map_attn_loss is not None:
                aux_loss = aux_loss + map_attn_w * map_attn_loss
                loss_out_dict['map_attn_loss'] = map_attn_loss.detach().view((1,))

        # ---- (E) z_global Auxiliary Decoder Loss: z-only action prediction ----
        z_aux_w = self.loss_weights.get('z_aux', 0.0)
        if z_aux_w > 0.0 and model is not None:
            z_aux_traj = getattr(model, '_z_aux_traj', None)
            if z_aux_traj is not None:
                gt_actions = model._compute_gt_actions(scene_graph, ego_only=True)  # (N_ego, FT, 2)
                z_aux_loss = nn.functional.mse_loss(z_aux_traj, gt_actions)
                aux_loss = aux_loss + z_aux_w * z_aux_loss
                loss_out_dict['z_aux_loss'] = z_aux_loss.detach().view((1,))

        return aux_loss

    def _compute_map_attn_guidance_loss(self, model, scene_graph, ego_mask, gt_future, FT):
        """
        Map Attention Guidance: attention이 전방 GT 위치의 map token에 집중하도록 KL loss.

        Transformer training: ego_attn is (B*T_total, N_ego, num_tokens) from Layer 0 A2S
        We only use future timesteps (PT:PT+FT) for loss.

        When pooled raster is available, soft labels are weighted by drivable area
        potential: non-drivable and lane-line tokens get reduced weight, encouraging
        attention to focus on safe, drivable regions.

        :return: scalar loss or None
        """
        device = gt_future.device
        normalizer = self.state_normalizer

        ego_attn_raw = model.get_ego_map_attn_weights()
        sur_attn_raw = model.get_sur_map_attn_weights()

        if ego_attn_raw is None:
            return None

        map_gt_steps = self.aux_cfg['map_gt_steps']
        decay_lambda = self.aux_cfg['map_gt_decay_lambda']
        t_arr = np.arange(map_gt_steps, dtype=np.float64)
        map_gt_weights = np.exp(-decay_lambda * t_arr)
        map_gt_weights = (map_gt_weights / map_gt_weights.sum()).tolist()
        grid_size = model.map_token_spatial
        num_tokens = grid_size * grid_size

        bounds = [-17.0, -38.5, 60.0, 38.5]
        pix_size = model.map_obs_size_pix  # 256
        rf_stride = model.map_rf_stride
        rf_offset = model.map_rf_offset

        # Compute per-agent drivable potential weight for soft labels
        # Per-step raster: (T, NA, C, 29, 29) or initial: (NA, C, 29, 29)
        raster_pooled_raw = model.get_map_raster_pooled()
        raster_per_step = (raster_pooled_raw is not None and raster_pooled_raw.dim() == 5)
        k_solid = 0.0  # disabled: avg pool dilutes thin lines too much
        k_dashed = 0.0

        PT = model.PT if hasattr(model, 'PT') else 4

        def _get_drivable_weight(agent_mask, step_idx):
            """Get flattened drivable weight (N_sel, 841) for given agents at given step."""
            if raster_pooled_raw is None:
                return None
            if raster_per_step:
                # (T_total, NA, C, 29, 29) — T_total = PT + FT, step_idx is future index
                rp = raster_pooled_raw[PT + step_idx]  # (NA, C, 29, 29)
            else:
                # (NA, C, 29, 29) — same for all steps
                rp = raster_pooled_raw
            rp_sel = rp[agent_mask]  # (N_sel, C, 29, 29)
            drivable = rp_sel[:, 0]
            solid = rp_sel[:, 1] if rp_sel.size(1) > 1 else torch.zeros_like(drivable)
            dashed = rp_sel[:, 2] if rp_sel.size(1) > 2 else torch.zeros_like(drivable)
            w = drivable * (1.0 - k_solid * solid) * (1.0 - k_dashed * dashed)
            w = w.clamp(min=0.01)
            return w.reshape(rp_sel.size(0), -1)  # (N_sel, 841)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        # ego_attn_raw:
        #   TF: (B*T_total, N_ego, num_tokens) — need shifted slicing [PT-1:-1]
        #   AR: (FT, N_ego, num_tokens) — already last-token-only, no slicing needed
        if isinstance(ego_attn_raw, torch.Tensor):
            PT = model.PT
            T_total = PT + FT
            num_ego = int(ego_mask.sum())

            if ego_attn_raw.size(0) == T_total * num_ego or ego_attn_raw.size(0) == T_total:
                # TF mode: (B*T_total, N_ego, num_tokens) — shifted slicing
                ego_attn_all = ego_attn_raw.view(T_total, num_ego, num_tokens)
                ego_attn_future = ego_attn_all[PT-1:-1]  # (FT, N_ego, num_tokens)
            else:
                # AR mode: (FT, N_ego, num_tokens) — already future-only
                ego_attn_future = ego_attn_raw  # (FT, N_ego, num_tokens)

            max_t = FT - map_gt_steps  # only steps with full lookahead
            for t in range(max_t):
                remaining = map_gt_steps

                ego_attn_dist = ego_attn_future[t]  # (N_ego, num_tokens)
                if t == 0:
                    ego_frame = scene_graph.past[:, -1, :4][ego_mask]
                else:
                    ego_frame = gt_future[ego_mask][:, t - 1, :4]

                ego_soft_label = self._make_soft_label(
                    ego_frame, gt_future[ego_mask], t, remaining,
                    map_gt_weights, normalizer, bounds, pix_size,
                    rf_stride, rf_offset, grid_size, num_tokens, device)

                if ego_soft_label is not None:
                    # Apply drivable potential weighting per step
                    ego_dw = _get_drivable_weight(ego_mask, t)
                    if ego_dw is not None:
                        ego_soft_label = ego_soft_label * ego_dw
                        ego_soft_label = ego_soft_label / (ego_soft_label.sum(dim=1, keepdim=True) + 1e-8)
                    ego_kl = self._kl_with_epsilon(ego_soft_label, ego_attn_dist)
                    total_loss = total_loss + ego_kl
                    count += 1

            # Sur attn guidance (Phase 1 only)
            if self.phase == 1 and sur_attn_raw is not None and isinstance(sur_attn_raw, torch.Tensor):
                num_sur = int((~ego_mask).sum())
                if sur_attn_raw.size(0) == T_total * num_sur or sur_attn_raw.size(0) == T_total:
                    # TF mode
                    sur_attn_all = sur_attn_raw.view(T_total, num_sur, num_tokens)
                    sur_attn_future = sur_attn_all[PT-1:-1]  # (FT, N_sur, num_tokens)
                else:
                    # AR mode
                    sur_attn_future = sur_attn_raw  # (FT, N_sur, num_tokens)

                for t in range(max_t):
                    remaining = map_gt_steps
                    sur_attn_dist = sur_attn_future[t]
                    if t == 0:
                        sur_frame = scene_graph.past[:, -1, :4][~ego_mask]
                    else:
                        sur_frame = gt_future[~ego_mask][:, t - 1, :4]
                    sur_soft_label = self._make_soft_label(
                        sur_frame, gt_future[~ego_mask], t, remaining,
                        map_gt_weights, normalizer, bounds, pix_size,
                        rf_stride, rf_offset, grid_size, num_tokens, device)
                    if sur_soft_label is not None:
                        # Apply drivable potential weighting per step
                        sur_dw = _get_drivable_weight(~ego_mask, t)
                        if sur_dw is not None:
                            sur_soft_label = sur_soft_label * sur_dw
                            sur_soft_label = sur_soft_label / (sur_soft_label.sum(dim=1, keepdim=True) + 1e-8)
                        sur_kl = self._kl_with_epsilon(sur_soft_label, sur_attn_dist)
                        total_loss = total_loss + sur_kl
                        count += 1
        else:
            # Inference mode: list or None — skip map attn guidance
            pass

        if count > 0:
            return total_loss / count
        return None

    def _make_soft_label(self, frame, gt_future_agent, t, remaining,
                         weights, normalizer, bounds, pix_size,
                         rf_stride, rf_offset, grid_size, num_tokens, device):
        """
        Trajectory-based soft label with uniform lateral width.

        For each grid cell:
        1. Find nearest point on GT future polyline, get perpendicular distance (d)
           and arc length (s) at that point
        2. Lateral: exp(-0.5 * d² / σ_d²) — uniform-width Gaussian band along trajectory
        3. Longitudinal: additive decay bonus on centerline — nearer = brighter center
        4. Normalize to probability distribution

        Coordinates use CNN receptive field centers:
          token[m] RF center pixel = m * rf_stride + rf_offset
          meters_to_token: pixel = (meters - bounds_min) / (bounds_max - bounds_min) * pix_size
                           token = (pixel - rf_offset) / rf_stride

        :param frame: (N, 4) current agent position (normalized, x,y,hx,hy)
        :param gt_future_agent: (N, FT, 6) GT future for these agents (normalized)
        :param t: current timestep
        :param remaining: number of future steps to look ahead
        :return: (N, num_tokens) soft label or None
        """
        N = frame.size(0)
        if N == 0:
            return None

        sigma_d = self.aux_cfg.get('map_gauss_sigma_d', 1.0)
        decay_lambda = self.aux_cfg.get('map_gt_decay_lambda', 0.3)

        frame_unnorm = normalizer.unnormalize(frame)  # (N, 4)

        def _m2token_l(meters):
            """meters (longitudinal) → token index using RF centers."""
            pixel = (meters - bounds[0]) / (bounds[2] - bounds[0]) * pix_size
            return (pixel - rf_offset) / rf_stride

        def _m2token_w(meters):
            """meters (lateral) → token index using RF centers."""
            pixel = (meters - bounds[1]) / (bounds[3] - bounds[1]) * pix_size
            return (pixel - rf_offset) / rf_stride

        # Collect GT waypoints in grid coordinates: agent pos + K future points
        # Agent's current position in its own local frame is always (0, 0)
        agent_gi = _m2token_l(torch.zeros(N, device=device))  # (N,)
        agent_gj = _m2token_w(torch.zeros(N, device=device))  # (N,)

        waypoints_i = [agent_gi]
        waypoints_j = [agent_gj]

        num_pts = 0
        for k in range(remaining):
            future_t = t + 1 + k
            if future_t >= gt_future_agent.size(1):
                break
            gt_pos = gt_future_agent[:, future_t, :4]
            gt_pos_unnorm = normalizer.unnormalize(gt_pos)
            local_pos = transform2frame(frame_unnorm, gt_pos_unnorm.unsqueeze(1))
            local_xy = local_pos[:, 0, :2]
            gi_f = _m2token_l(local_xy[:, 0])
            gj_f = _m2token_w(local_xy[:, 1])
            waypoints_i.append(gi_f)
            waypoints_j.append(gj_f)
            num_pts += 1

        if num_pts == 0:
            return None

        # Stack waypoints: (N, K+1) where K+1 = agent + future points
        wp_i = torch.stack(waypoints_i, dim=1)  # (N, K+1)
        wp_j = torch.stack(waypoints_j, dim=1)  # (N, K+1)

        # Compute cumulative arc length along polyline (grid cell units)
        seg_di = wp_i[:, 1:] - wp_i[:, :-1]  # (N, K)
        seg_dj = wp_j[:, 1:] - wp_j[:, :-1]  # (N, K)
        seg_len = torch.sqrt(seg_di ** 2 + seg_dj ** 2 + 1e-8)  # (N, K)
        cum_len = torch.cat([torch.zeros(N, 1, device=device),
                             torch.cumsum(seg_len, dim=1)], dim=1)  # (N, K+1)
        total_len = cum_len[:, -1:]  # (N, 1)

        # Grid coordinates: (G*G, 2)
        gi_range = torch.arange(grid_size, device=device, dtype=torch.float32)
        gj_range = torch.arange(grid_size, device=device, dtype=torch.float32)
        grid_i, grid_j = torch.meshgrid(gi_range, gj_range)
        g_i = grid_i.reshape(-1)  # (G*G,)
        g_j = grid_j.reshape(-1)  # (G*G,)
        G2 = g_i.size(0)

        # For each grid cell, find nearest point on polyline
        soft_label = torch.zeros(N, G2, device=device)
        inv_sigma_d_sq = 1.0 / (sigma_d ** 2)

        for n in range(N):
            wi = wp_i[n]
            wj = wp_j[n]
            cl = cum_len[n]
            tl = total_len[n, 0]

            K = num_pts
            seg_start_i = wi[:K]
            seg_start_j = wj[:K]
            s_len = seg_len[n]

            # Vector from segment start to each grid cell: (G2, K)
            to_grid_i = g_i.unsqueeze(1) - seg_start_i.unsqueeze(0)
            to_grid_j = g_j.unsqueeze(1) - seg_start_j.unsqueeze(0)

            # Segment direction unit vectors: (K,)
            dir_i = seg_di[n] / (s_len + 1e-8)
            dir_j = seg_dj[n] / (s_len + 1e-8)

            # Project grid-to-start onto segment direction: (G2, K)
            proj = to_grid_i * dir_i.unsqueeze(0) + to_grid_j * dir_j.unsqueeze(0)
            proj_clamped = proj.clamp(min=0).clamp(max=s_len.unsqueeze(0))

            # Nearest point on segment
            near_i = seg_start_i.unsqueeze(0) + proj_clamped * dir_i.unsqueeze(0)
            near_j = seg_start_j.unsqueeze(0) + proj_clamped * dir_j.unsqueeze(0)

            # Perpendicular distance: (G2, K)
            d_sq = (g_i.unsqueeze(1) - near_i) ** 2 + (g_j.unsqueeze(1) - near_j) ** 2

            # Arc length at projection point: (G2, K)
            arc_at_proj = cl[:K].unsqueeze(0) + proj_clamped

            # Find nearest segment for each grid cell
            nearest_seg = d_sq.argmin(dim=1)
            arange_idx = torch.arange(G2, device=device)
            min_d_sq = d_sq[arange_idx, nearest_seg]
            min_arc = arc_at_proj[arange_idx, nearest_seg]

            # Mask out cells behind the agent
            raw_proj_seg0 = proj[:, 0]
            behind_mask = (nearest_seg == 0) & (raw_proj_seg0 < 0)

            # Lateral Gaussian: exp(-0.5 * d² / σ_d²)
            lateral = torch.exp(-0.5 * min_d_sq * inv_sigma_d_sq)

            # Longitudinal decay: exp(-λ * s)
            arc_normalized = min_arc / (tl + 1e-8) * num_pts
            longitudinal = torch.exp(-decay_lambda * arc_normalized)

            cell_value = longitudinal * lateral
            cell_value[behind_mask] = 0.0
            soft_label[n] = cell_value

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
        Computes loss (Transformer Decoder).

        Transformer decoder always returns (NA, FT, 4) tensor for both
        training (parallel) and inference (autoregressive).

        :param scene_graph: containing input and GT data
        :param pred: dict of model predictions.
        :param model: TrafficPlannerModel instance (for auxiliary loss computation)
        :param use_teacher_forcing: ignored (kept for API compatibility)
        '''

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

        # Reconstruction loss: log_normal (all 4 dims) + position MSE boost
        recon_loss = -log_normal(pred_future_valid, gt_future_valid[:, :4], torch.ones_like(pred_future_valid))
        pos_mse = ((pred_future_valid[:, :2] - gt_future_valid[:, :2]) ** 2).sum(dim=-1)
        head_mse = ((pred_future_valid[:, 2:4] - gt_future_valid[:, 2:4]) ** 2).sum(dim=-1)
        if self.recon_pos_weight > 0.0:
            recon_loss = recon_loss + self.recon_pos_weight * pos_mse
        # monitoring: MSE for each component
        pos_loss = pos_mse
        head_loss = head_mse

        # KL divergence loss
        pm, pv = pred['prior_out']
        qm, qv = pred['posterior_out']
        kl_loss = kl_normal(qm, qv, pm, pv, free_bits=self.kl_free_bits)

        # total weighted loss
        loss = self.loss_weights['recon'] * recon_loss.mean()

        # KL loss: Phase 1 only
        if self.phase == 1 and self.loss_weights.get('kl', 0.0) > 0.0:
            loss = loss + self.loss_weights['kl'] * kl_loss.mean()

        # z_local sparsity loss
        sparse_loss = None
        if self.use_sparse_loss and self.loss_weights.get('sparse', 0.0) > 0.0 and model is not None:
            z_local_stacked = model.get_z_local_stacked()
            if z_local_stacked is not None:
                sparse_loss = self.sparsity_loss(z_local_stacked)
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
            'pos_loss': pos_loss,
            'head_loss': head_loss,
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

    def compute_err(self, scene_graph, pred, normalizer):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

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