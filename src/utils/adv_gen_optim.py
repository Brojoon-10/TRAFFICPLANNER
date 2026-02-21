# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from losses.traffic_model import compute_coll_rate_env
from losses.adv_gen_nusc import interp_traj

from utils.transforms import transform2frame
from utils.logger import Logger, throw_err
from utils.scenario_gen import log_metric, log_freq_stat, viz_optim_results
import copy

def collate_tgt_other_z(scene_graph, tgt_z, other_z):
    '''
    Combines latents into the correct full scene graph structure with tgt at the 0 index of
    each scene graph.
    :param tgt_z: (B, NS, D) or (B, D)
    :param other_z: (NA-B, NS, D) or (NA-B, D)
    '''
    B = tgt_z.size(0)
    if len(tgt_z.size()) == 3:
        cur_z = torch.empty((0, other_z.size(1), other_z.size(2))).to(other_z)
    elif len(tgt_z.size()) == 2:
        cur_z = torch.empty((0, other_z.size(1))).to(other_z)
    prev_idx = 0
    for bidx in range(B):
        cur_size = scene_graph.ptr[bidx+1] - scene_graph.ptr[bidx] - 1
        cur_z = torch.cat([cur_z, tgt_z[bidx:bidx+1], other_z[prev_idx:prev_idx+cur_size]], dim=0)
        prev_idx += cur_size
    return cur_z


def run_adv_gen_optim(cur_z, lr, loss_weights, model, scene_graph, map_env, map_idx,
                    num_iters, embed_info, planner_name, tgt_prior_distrib, other_prior_distrib,
                    feasibility_time, feasibility_infront_min,
                    planner=None, planner_viz_out=None,
                    attack_agt_idx=None,
                    future_len=None,
                    veh_coll_buffer=0.1):
    '''
    :param attack_agt_idx: list of attackers LOCAL to each scene in the batched graph.
    :param future_len: if given, rolls out scenario future this many steps rather than the default of the model
    '''
    B = map_idx.size(0)
    NA = cur_z.size(0)
    ego_inds = scene_graph.ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True

    if attack_agt_idx is not None:
        attack_agt_idx = torch.tensor(attack_agt_idx).to(scene_graph.ptr)
        attack_agt_idx = attack_agt_idx + scene_graph.ptr[:-1]

    if future_len is None:
        future_len = model.FT

    # set up optimization
    tgt_z = cur_z[ego_mask].clone().detach() # external agent
    other_z_all = cur_z[~ego_mask].clone().detach() # all agents are attacking
    other_z_all.requires_grad = True

    # For ego_pred: use prior z (fixed), don't optimize ego
    if planner_name == 'ego_pred':
        tgt_z = tgt_prior_distrib[0].clone().detach()  # prior z for ego
        tgt_z.requires_grad = False
        optim_z = [other_z_all]  # only optimize adversaries
    else:
        tgt_z.requires_grad = True
        optim_z = [tgt_z, other_z_all]

    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all)

    adv_optim = optim.Adam(optim_z, lr=lr)

    # create loss functions
    from losses.adv_gen_nusc import TgtMatchingLoss, AdvGenLoss       
    tgt_loss = TgtMatchingLoss(loss_weights)
    adv_loss = AdvGenLoss(loss_weights,
                            model.get_att_normalizer().unnormalize(scene_graph.lw),
                            map_idx[scene_graph.batch],
                            map_env,
                            cur_z[~ego_mask].clone().detach(),
                            scene_graph.ptr,
                            veh_coll_buffer=veh_coll_buffer,
                            crash_loss_min_time=feasibility_time,
                            crash_loss_min_infront=feasibility_infront_min)

    ############### MODIFIED: ego_pred also doesn't need planner reset ###############
    if planner_name not in ['ego', 'ego_pred']:
    ####################################################################################
        # for real planners need to set initial state for rollouts
        all_init_state = model.get_normalizer().unnormalize(scene_graph.past_gt[:, -1, :])
        all_init_veh_att = model.get_att_normalizer().unnormalize(scene_graph.lw)
        planner.reset(all_init_state, all_init_veh_att, scene_graph.batch, B, map_idx)

    # default to open-loop
    planner_inject_traj = True # whether to use observed planner future rather than internal prediction during decoder rollout
    adv_use_own_pred = False   # whether to use our differentiable approx of planner rather than real observations to compute loss
    # if using hardcode, is closed-loop
    if planner_name == 'hardcode':
        Logger.log('NOTE: Operating in closed-loop for adv gen optimization!')
        planner_inject_traj = False
        adv_use_own_pred = True

        cur_agt_ptr = scene_graph.ptr - torch.arange(B+1)
        plan_t = np.linspace(model.dt, model.dt*future_len, future_len)

    ############### MODIFIED: Support ego_pred planner ###############
    if planner_name == 'ego_pred':
        Logger.log('NOTE: Using ego_pred planner (model prediction, not GT)!')
    ##################################################################

    # run optim
    pbar_optim = tqdm.tqdm(range(num_iters))
    for oidx in pbar_optim:
        def closure():
            adv_optim.zero_grad()

            # decode to get current future and compute loss
            loss = loss_dict = planner_fut = None
            if planner_name == 'ego':
                # in open-loop operation: already rolled out planner a single time, just try to attack.
                # for ego planner, take GT future traj
                planner_fut = scene_graph.future_gt[ego_mask][:, :, :4]
                # if we're injecting external, must be the same length as rollout
                assert planner_inject_traj is False or planner_fut.size(1) == future_len
            elif planner_name == 'ego_pred':
                ############### MODIFIED: Use model prediction as target ###############
                # Get current ego prediction from optimized tgt_z
                temp_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all.clone().detach())
                with torch.no_grad():
                    temp_out = model.decode_embedding(temp_z, embed_info, scene_graph, map_idx, map_env, nfuture=future_len)
                    planner_fut = temp_out['future_pred'][ego_mask]
                assert planner_inject_traj is False or planner_fut.size(1) == future_len
                ########################################################################

            other_z = other_z_all
            tgt_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all.clone().detach())
            other_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z.clone().detach(), other_z_all)

            # forward pass to compute target-specific loss
            tgt_decoder_out = model.decode_embedding(tgt_loss_input_z, embed_info, scene_graph, map_idx, map_env,
                                                ext_future=planner_fut if planner_inject_traj else None,
                                                nfuture=future_len)
            # forward pass to compute controlled agent loss
            other_decoder_out = model.decode_embedding(other_loss_input_z, embed_info, scene_graph, map_idx, map_env,
                                                ext_future=planner_fut if planner_inject_traj else None,
                                                nfuture=future_len)

            # rollout planner allowing it to react to the current model rollout
            if planner_name == 'hardcode':
                cur_agt_pred = tgt_decoder_out['future_pred'][~ego_mask]
                cur_agt_pred = model.normalizer.unnormalize(cur_agt_pred).detach().cpu().numpy()
                planner_fut = planner.rollout(cur_agt_pred, plan_t, cur_agt_ptr.cpu().numpy(), plan_t,
                                                # viz=True,
                                                control_all=False).to(scene_graph.future_gt)
                planner_fut = model.get_normalizer().normalize(planner_fut)

            # compute each loss
            loss_dict = dict()
            tgt_match_loss_dict = tgt_loss(model.get_normalizer().unnormalize(tgt_decoder_out['future_pred'][ego_mask]),
                                            model.get_normalizer().unnormalize(planner_fut),
                                            tgt_z,
                                            tgt_prior_distrib)
            loss_dict = {'tgt_match_' + k : v for k, v in tgt_match_loss_dict.items()}

            tgt_traj = planner_fut if not adv_use_own_pred else other_decoder_out['future_pred'][ego_mask]
            adv_loss_dict = adv_loss(model.get_normalizer().unnormalize(other_decoder_out['future_pred']),
                                        model.get_normalizer().unnormalize(tgt_traj),
                                        other_z,
                                        other_prior_distrib,
                                        attack_agt_idx=attack_agt_idx)
            adv_loss_dict = {'adv_' + k : v for k, v in adv_loss_dict.items()}
            loss_dict = {**loss_dict, **adv_loss_dict}

            # add together to get single loss
            loss = loss_dict['tgt_match_loss'] + loss_dict['adv_loss']

            progress_bar_metrics = {}
            for k, v in loss_dict.items():
                if v is None:
                    continue
                progress_bar_metrics[k] = torch.mean(v).item()
                print('%s = %f' % (k, progress_bar_metrics[k]))
            pbar_optim.set_postfix(progress_bar_metrics)

            # backprop
            loss.backward()
            return loss

        # update
        closure()
        adv_optim.step()

    # get final results
    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all)
    with torch.no_grad():
        final_decoder_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env, nfuture=future_len)
    final_result_traj = final_decoder_out['future_pred'].unsqueeze(1).clone().detach()
    if planner_name == 'ego':
        # replace the ego (planner) traj with GT
        final_result_traj[ego_inds, torch.zeros_like(ego_inds)] = scene_graph.future_gt[ego_mask][:, :, :4]
    elif planner_name == 'hardcode':
        # replace ego (planner) traj with actual output of planner
        cur_agt_pred = final_decoder_out['future_pred'][~ego_mask]
        cur_agt_pred = model.normalizer.unnormalize(cur_agt_pred).detach().cpu().numpy()
        planner_fut = planner.rollout(cur_agt_pred, plan_t, cur_agt_ptr.cpu().numpy(), plan_t,
                                        viz=planner_viz_out,
                                        control_all=False).to(scene_graph.future_gt)
        planner_fut = model.get_normalizer().normalize(planner_fut)
        final_result_traj[ego_inds, torch.zeros_like(ego_inds)] = planner_fut
    elif planner_name == 'ego_pred':
        # ego_pred: ego trajectory is already from model prediction (prior z)
        # just use the final_decoder_out which already has ego prediction
        pass  # final_result_traj already contains ego prediction from final_decoder_out

    # compute loss one more time to get final min agt/t
    tgt_traj = final_result_traj[ego_inds, torch.zeros_like(ego_inds)] # the true planner rollout
    adv_loss_dict = adv_loss(model.get_normalizer().unnormalize(final_decoder_out['future_pred']),
                                model.get_normalizer().unnormalize(tgt_traj),
                                cur_z[~ego_mask].clone().detach(),
                                other_prior_distrib,
                                return_mins=True)
    cur_min_agt = cur_min_t = None
    if 'min_agt' in adv_loss_dict:
        # returned as index within each batch
        cur_min_agt = adv_loss_dict['min_agt'] + scene_graph.ptr[:-1].cpu().numpy()
        print('Final min agt: ' + str(cur_min_agt))
    if 'min_t' in adv_loss_dict:
        cur_min_t = adv_loss_dict['min_t']
        print('Final min t: ' + str(cur_min_t))

    return cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t


def compute_adv_gen_success(final_result_traj, model, scene_graph, attack_agt):
    '''
    Computes whether scenario is successful in colliding w/ planner
    All inputs assumed NORMALIZED.
    :param final_result_traj: (NA, 1, FT, 4) the final scenario where the agent at idx=0 is the true planner
                                            reaction to the scenario (NOT the model's prediction of the planner)
    '''
    from losses.adv_gen_nusc import check_single_veh_coll

    # final result trajectories
    planner_gt_fut = model.get_normalizer().unnormalize(final_result_traj[0, 0])
    other_fut = model.get_normalizer().unnormalize(final_result_traj[1:, 0])
    # agent attribs
    planner_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[0])
    other_lw = model.get_att_normalizer().unnormalize(scene_graph.lw[1:])

    # DEBUG: print shapes and values
    print(f"DEBUG: final_result_traj shape: {final_result_traj.shape}")
    print(f"DEBUG: planner_gt_fut shape: {planner_gt_fut.shape}")
    print(f"DEBUG: other_fut shape: {other_fut.shape}")
    print(f"DEBUG: attack_agt: {attack_agt}")

    # Collision with planner target
    planner_coll_all, planner_coll_time = check_single_veh_coll(planner_gt_fut, planner_lw, other_fut, other_lw)
    print(f"DEBUG: planner_coll_all: {planner_coll_all}")
    print(f"DEBUG: planner_coll_time: {planner_coll_time}")
    print(f"DEBUG: attack_agt-1 index: {attack_agt-1}")
    attack_coll = planner_coll_all[attack_agt-1]
    adv_success = bool(attack_coll)
    print(f"DEBUG: adv_success: {adv_success}")

    return adv_success


############### HJ_EDITED: ego_pred_reactive planner - sliding window approach ###############
def run_adv_gen_optim_reactive(cur_z, lr, loss_weights, model, scene_graph, map_env, map_idx,
                               num_iters_per_round, embed_info, tgt_prior_distrib, other_prior_distrib,
                               feasibility_time, feasibility_infront_min,
                               attack_agt_idx=None,
                               future_len=None,
                               veh_coll_buffer=0.1):
    '''
    Sliding window approach for ego_pred_reactive planner.
    Ego predicts based on shifted past (including previous predictions).
    Sur optimizes entire t1-t12 at once, making Sur trajectory continuous.

    Each round:
    1. Ego predicts t1-t12 using current shifted past
    2. Sur optimizes t1-t12 against Ego's full prediction
    3. Confirm one timestep (round r confirms t_r)
    4. Shift past window: include newly confirmed timestep

    :param num_iters_per_round: number of optimization iterations per round (12 rounds total)
    '''
    B = map_idx.size(0)
    NA = cur_z.size(0)
    ego_inds = scene_graph.ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True

    if attack_agt_idx is not None:
        attack_agt_idx = torch.tensor(attack_agt_idx).to(scene_graph.ptr)
        attack_agt_idx = attack_agt_idx + scene_graph.ptr[:-1]

    if future_len is None:
        future_len = model.FT

    PT = model.PT  # past length (4)

    # Store original past for Sur optimization (always uses original)
    original_past = scene_graph.past.clone()
    original_past_vis = scene_graph.past_vis.clone()
    # HJ_EDITED: Use 'in' instead of hasattr for PyG Data objects
    original_past_gt = scene_graph.past_gt.clone() if 'past_gt' in scene_graph else None

    # Initialize: ego uses prior z (fixed)
    tgt_z = tgt_prior_distrib[0].clone().detach()
    tgt_z.requires_grad = False

    # Sur z will be optimized
    other_z_all = cur_z[~ego_mask].clone().detach()
    other_z_all.requires_grad = True

    # Track confirmed future steps
    confirmed_ego_future = []  # list of confirmed ego positions at each timestep
    # Sur's full t1-t12 trajectory from previous round (for Ego to see as Sur's past)
    prev_sur_full_traj = None  # (NA-B, 12, 4) - updated each round

    from losses.adv_gen_nusc import TgtMatchingLoss, AdvGenLoss

    Logger.log('NOTE: Using ego_pred_reactive planner (sliding window, model-based reactive)!')
    Logger.log(f'Running {future_len} rounds with {num_iters_per_round} iterations per round')

    # Run sliding window for each future timestep
    for round_idx in range(future_len):
        Logger.log(f'\n=== Round {round_idx + 1}/{future_len}: Confirming timestep t{round_idx + 1} ===')

        # Step 1: Update ego's past with shifted window
        # For round r, ego sees: original_past[-(PT-r):] + confirmed_predictions[:r]
        if round_idx > 0:
            # Build shifted past for ego
            # Take last (PT - round_idx) steps from original past, or less if round_idx >= PT
            steps_from_original = max(0, PT - round_idx)
            steps_from_confirmed = min(round_idx, PT)

            shifted_ego_past = []
            shifted_ego_past_vis = []

            if steps_from_original > 0:
                # Take last steps_from_original from original past
                shifted_ego_past.append(original_past[ego_mask][:, -steps_from_original:, :])
                shifted_ego_past_vis.append(original_past_vis[ego_mask][:, -steps_from_original:])

            if steps_from_confirmed > 0:
                # Take last steps_from_confirmed from confirmed predictions
                confirmed_start = max(0, round_idx - PT)
                confirmed_steps = torch.stack(confirmed_ego_future[confirmed_start:round_idx], dim=1)  # (B, steps, 4)
                # Pad to full state size (add speed and hdot as 0)
                confirmed_steps_full = torch.zeros(confirmed_steps.size(0), confirmed_steps.size(1), 6).to(confirmed_steps)
                confirmed_steps_full[:, :, :4] = confirmed_steps
                shifted_ego_past.append(confirmed_steps_full)

                confirmed_vis = torch.ones(confirmed_steps.size(0), confirmed_steps.size(1)).to(original_past_vis)
                shifted_ego_past_vis.append(confirmed_vis)

            # Concatenate
            shifted_ego_past = torch.cat(shifted_ego_past, dim=1)
            shifted_ego_past_vis = torch.cat(shifted_ego_past_vis, dim=1)

            # Similarly for Sur: use previous round's Sur optimization result (t1~t{round_idx})
            # This way Sur's past comes from a single continuous trajectory
            shifted_sur_past = []
            shifted_sur_past_vis = []

            if steps_from_original > 0:
                shifted_sur_past.append(original_past[~ego_mask][:, -steps_from_original:, :])
                shifted_sur_past_vis.append(original_past_vis[~ego_mask][:, -steps_from_original:])

            if steps_from_confirmed > 0 and prev_sur_full_traj is not None:
                # Get t_{confirmed_start+1} ~ t_{round_idx} from previous round's Sur trajectory
                # prev_sur_full_traj is (NA-B, 12, 4), indices 0-11 correspond to t1-t12
                sur_past_from_prev = prev_sur_full_traj[:, confirmed_start:round_idx, :]  # (NA-B, steps_from_confirmed, 4)
                sur_past_full = torch.zeros(sur_past_from_prev.size(0), sur_past_from_prev.size(1), 6).to(sur_past_from_prev)
                sur_past_full[:, :, :4] = sur_past_from_prev
                shifted_sur_past.append(sur_past_full)

                sur_past_vis = torch.ones(sur_past_from_prev.size(0), sur_past_from_prev.size(1)).to(original_past_vis)
                shifted_sur_past_vis.append(sur_past_vis)

            shifted_sur_past = torch.cat(shifted_sur_past, dim=1)
            shifted_sur_past_vis = torch.cat(shifted_sur_past_vis, dim=1)

            # Update scene_graph.past for ego prediction (shifted past)
            scene_graph.past[ego_mask] = shifted_ego_past
            scene_graph.past_vis[ego_mask] = shifted_ego_past_vis
            scene_graph.past[~ego_mask] = shifted_sur_past
            scene_graph.past_vis[~ego_mask] = shifted_sur_past_vis

            if original_past_gt is not None:
                scene_graph.past_gt[ego_mask] = shifted_ego_past
                scene_graph.past_gt[~ego_mask] = shifted_sur_past

        # Re-embed with updated past
        with torch.no_grad():
            embed_info_current_attached = model.embed(scene_graph, map_idx, map_env)
        # Detach embed_info to avoid graph issues during optimization
        # HJ_EDITED: Handle both 'past_feat' (original model) and 'past_seq_out' (transformer model)
        embed_info_current = {
            'prior_out': (embed_info_current_attached['prior_out'][0].clone().detach(),
                          embed_info_current_attached['prior_out'][1].clone().detach()),
            'map_feat': embed_info_current_attached['map_feat'].clone().detach(),
        }
        # Support both original model (past_feat) and transformer model (past_seq_out)
        if 'past_feat' in embed_info_current_attached:
            embed_info_current['past_feat'] = embed_info_current_attached['past_feat'].clone().detach()
        if 'past_seq_out' in embed_info_current_attached:
            embed_info_current['past_seq_out'] = embed_info_current_attached['past_seq_out'].clone().detach()
        if 'posterior_out' in embed_info_current_attached:
            embed_info_current['posterior_out'] = (
                embed_info_current_attached['posterior_out'][0].clone().detach(),
                embed_info_current_attached['posterior_out'][1].clone().detach()
            )

        # Get new prior distribution for Ego from shifted past
        tgt_prior_distrib_current = embed_info_current['prior_out']

        # Update tgt_z to new prior (ego follows road properly)
        tgt_z = tgt_prior_distrib_current[0][ego_mask].clone().detach()
        tgt_z.requires_grad = False

        # Remaining future steps in original timeline (t_{round_idx+1} ~ t_{future_len})
        remaining_steps = future_len - round_idx

        # Step 2: Ego predicts full 12 timesteps from shifted past perspective
        # Combine confirmed past predictions + new predictions to form full 12-step trajectory
        cur_z_for_pred = collate_tgt_other_z(scene_graph, tgt_z, other_z_all.clone().detach())
        with torch.no_grad():
            ego_pred_out = model.decode_embedding(cur_z_for_pred, embed_info_current, scene_graph,
                                                   map_idx, map_env, nfuture=future_len)  # always predict 12
            # New prediction covers t_{round_idx+1} ~ t_{future_len} (remaining_steps)
            new_ego_pred = ego_pred_out['future_pred'][ego_mask][:, :remaining_steps, :]  # (B, remaining_steps, 4)

            # Build full 12-step planner_fut: confirmed (t1~t_{round_idx}) + new prediction (t_{round_idx+1}~t12)
            if round_idx > 0:
                confirmed_ego_stack = torch.stack(confirmed_ego_future, dim=1)  # (B, round_idx, 4)
                planner_fut = torch.cat([confirmed_ego_stack, new_ego_pred], dim=1)  # (B, 12, 4)
            else:
                planner_fut = new_ego_pred  # (B, 12, 4) for round 0

        # Step 3: Sur optimization against ego's full prediction
        # Restore original past for Sur optimization (Sur uses t-3~t0, not shifted past)
        scene_graph.past = original_past.clone()
        scene_graph.past_vis = original_past_vis.clone()
        if original_past_gt is not None:
            scene_graph.past_gt = original_past_gt.clone()

        # Re-embed with original past for Sur optimization
        with torch.no_grad():
            embed_info_sur = model.embed(scene_graph, map_idx, map_env)
        embed_info_sur_detached = {
            'prior_out': (embed_info_sur['prior_out'][0].clone().detach(),
                          embed_info_sur['prior_out'][1].clone().detach()),
            'map_feat': embed_info_sur['map_feat'].clone().detach(),
        }
        if 'past_feat' in embed_info_sur:
            embed_info_sur_detached['past_feat'] = embed_info_sur['past_feat'].clone().detach()
        if 'past_seq_out' in embed_info_sur:
            embed_info_sur_detached['past_seq_out'] = embed_info_sur['past_seq_out'].clone().detach()
        if 'posterior_out' in embed_info_sur:
            embed_info_sur_detached['posterior_out'] = (
                embed_info_sur['posterior_out'][0].clone().detach(),
                embed_info_sur['posterior_out'][1].clone().detach()
            )

        # Get prior distribution from original past for Sur
        other_prior_distrib_sur = (
            embed_info_sur_detached['prior_out'][0][~ego_mask],
            embed_info_sur_detached['prior_out'][1][~ego_mask]
        )

        # HJ_EDITED: Create a new leaf tensor for optimization to avoid graph issues
        other_z_optim = other_z_all.clone().detach()
        other_z_optim.requires_grad = True
        optim_z = [other_z_optim]
        adv_optim = optim.Adam(optim_z, lr=lr)

        # Create loss functions
        tgt_loss = TgtMatchingLoss(loss_weights)
        adv_loss = AdvGenLoss(loss_weights,
                              model.get_att_normalizer().unnormalize(scene_graph.lw),
                              map_idx[scene_graph.batch],
                              map_env,
                              other_z_optim.clone().detach(),
                              scene_graph.ptr,
                              veh_coll_buffer=veh_coll_buffer,
                              crash_loss_min_time=feasibility_time,
                              crash_loss_min_infront=feasibility_infront_min)

        # Run optimization for this round
        pbar_optim = tqdm.tqdm(range(num_iters_per_round), desc=f'Round {round_idx+1}')
        for oidx in pbar_optim:
            adv_optim.zero_grad()

            tgt_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_optim.clone().detach())
            other_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z.clone().detach(), other_z_optim)

            # Forward pass with original past embedding for Sur optimization
            # Sur predicts t1-t12 from original past, optimized against Ego's full trajectory
            tgt_decoder_out = model.decode_embedding(tgt_loss_input_z, embed_info_sur_detached, scene_graph,
                                                     map_idx, map_env,
                                                     ext_future=planner_fut,
                                                     nfuture=future_len)
            other_decoder_out = model.decode_embedding(other_loss_input_z, embed_info_sur_detached, scene_graph,
                                                       map_idx, map_env,
                                                       ext_future=planner_fut,
                                                       nfuture=future_len)

            # Compute losses
            loss_dict = dict()
            tgt_match_loss_dict = tgt_loss(
                model.get_normalizer().unnormalize(tgt_decoder_out['future_pred'][ego_mask]),
                model.get_normalizer().unnormalize(planner_fut),
                tgt_z,
                (tgt_z, torch.ones_like(tgt_z))  # dummy prior for matching
            )
            loss_dict = {'tgt_match_' + k: v for k, v in tgt_match_loss_dict.items()}

            adv_loss_dict = adv_loss(
                model.get_normalizer().unnormalize(other_decoder_out['future_pred']),
                model.get_normalizer().unnormalize(planner_fut),
                other_z_optim,
                other_prior_distrib_sur,  # Use prior from original past for Sur
                attack_agt_idx=attack_agt_idx
            )
            adv_loss_dict = {'adv_' + k: v for k, v in adv_loss_dict.items()}
            loss_dict = {**loss_dict, **adv_loss_dict}

            loss = loss_dict['tgt_match_loss'] + loss_dict['adv_loss']

            progress_bar_metrics = {}
            for k, v in loss_dict.items():
                if v is not None:
                    progress_bar_metrics[k] = torch.mean(v).item()
            pbar_optim.set_postfix(progress_bar_metrics)

            loss.backward()
            adv_optim.step()

        # Step 4: Confirm this round's timestep (t_{round_idx+1})
        # Update other_z_all with optimized values for next round
        other_z_all = other_z_optim.clone().detach()

        # Get final predictions for this round (use original past embedding for Sur)
        cur_z_final = collate_tgt_other_z(scene_graph, tgt_z, other_z_optim.clone().detach())
        with torch.no_grad():
            final_out = model.decode_embedding(cur_z_final, embed_info_sur_detached, scene_graph,
                                                map_idx, map_env, nfuture=future_len,
                                                ext_future=planner_fut)

        # Confirm this round's timestep: t_{round_idx+1} in original timeline
        # planner_fut is full 12 steps: confirmed (t1~t_{round_idx}) + new pred (t_{round_idx+1}~t12)
        # So t_{round_idx+1} is at index round_idx (0-indexed)
        confirmed_ego_step = planner_fut[:, round_idx, :].clone().detach()  # (B, 4) - t_{round_idx+1}
        confirmed_ego_future.append(confirmed_ego_step)

        # Sur's full t1-t12 trajectory from this round's optimization
        # Store for: 1) next round's Ego to see as Sur's past, 2) final output if last round
        prev_sur_full_traj = final_out['future_pred'][~ego_mask].clone().detach()  # (NA-B, 12, 4)

        Logger.log(f'Confirmed t{round_idx + 1}')

    # Restore original past
    scene_graph.past = original_past
    scene_graph.past_vis = original_past_vis
    if original_past_gt is not None:
        scene_graph.past_gt = original_past_gt

    # Build final result trajectories
    # Ego: stack all confirmed steps
    final_ego_traj = torch.stack(confirmed_ego_future, dim=1)  # (B, future_len, 4)
    # Sur: use the full trajectory from the last round's optimization
    final_sur_traj = prev_sur_full_traj  # (NA-B, future_len, 4)

    # Construct final_result_traj in the expected format (NA, 1, FT, 4)
    final_result_traj = torch.zeros(NA, 1, future_len, 4).to(final_ego_traj)
    final_result_traj[ego_mask, 0] = final_ego_traj
    final_result_traj[~ego_mask, 0] = final_sur_traj

    # Get final z
    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all)

    # Re-embed with original past for final decoder output
    # HJ_EDITED: Wrap in no_grad to avoid graph issues
    with torch.no_grad():
        embed_info_final = model.embed(scene_graph, map_idx, map_env)
        final_decoder_out = model.decode_embedding(cur_z, embed_info_final, scene_graph, map_idx, map_env, nfuture=future_len)

    # Compute final min agt/t
    from losses.adv_gen_nusc import TgtMatchingLoss, AdvGenLoss
    adv_loss_final = AdvGenLoss(loss_weights,
                                model.get_att_normalizer().unnormalize(scene_graph.lw),
                                map_idx[scene_graph.batch],
                                map_env,
                                cur_z[~ego_mask].clone().detach(),
                                scene_graph.ptr,
                                veh_coll_buffer=veh_coll_buffer,
                                crash_loss_min_time=feasibility_time,
                                crash_loss_min_infront=feasibility_infront_min)

    tgt_traj = final_result_traj[ego_inds, torch.zeros_like(ego_inds)]
    adv_loss_dict = adv_loss_final(
        model.get_normalizer().unnormalize(final_result_traj[:, 0]),
        model.get_normalizer().unnormalize(tgt_traj),
        cur_z[~ego_mask].clone().detach(),
        other_prior_distrib,
        return_mins=True
    )

    cur_min_agt = cur_min_t = None
    if 'min_agt' in adv_loss_dict:
        cur_min_agt = adv_loss_dict['min_agt'] + scene_graph.ptr[:-1].cpu().numpy()
        print('Final min agt: ' + str(cur_min_agt))
    if 'min_t' in adv_loss_dict:
        cur_min_t = adv_loss_dict['min_t']
        print('Final min t: ' + str(cur_min_t))

    return cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t
##########################################################################################


############### HJ_ADDED: run_adv_gen_optim_agent_attention ###############
# Agent Cross-Attention based Hardcode-style Optimization
#
# Algorithm (per iteration):
#   1. Decode Sur trajectory (12 steps) with current Sur z
#   2. Inject Sur traj via ext_future -> Ego reacts via Agent Cross-Attention
#   3. Optimize Sur z to collide with reactive Ego trajectory
#   4. Repeat
#
# Key: Uses model's internal Agent Cross-Attention instead of external planner
##########################################################################
def run_adv_gen_optim_agent_attention(cur_z, lr, loss_weights, model, scene_graph, map_env, map_idx,
                                       num_iters, embed_info, tgt_prior_distrib, other_prior_distrib,
                                       feasibility_time, feasibility_infront_min,
                                       attack_agt_idx=None,
                                       future_len=None,
                                       veh_coll_buffer=0.1):
    B = map_idx.size(0)
    NA = cur_z.size(0)
    ego_inds = scene_graph.ptr[:-1]

    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True
    sur_mask = ~ego_mask

    if attack_agt_idx is not None:
        attack_agt_idx = torch.tensor(attack_agt_idx).to(scene_graph.ptr)
        attack_agt_idx = attack_agt_idx + scene_graph.ptr[:-1]

    if future_len is None:
        future_len = model.FT

    # Ego: fixed prior z (reaction via Agent Attention, not z optimization)
    tgt_z = tgt_prior_distrib[0].clone().detach()
    tgt_z.requires_grad = False

    # Sur: z will be optimized
    other_z_all = cur_z[sur_mask].clone().detach()
    other_z_all.requires_grad = True

    from losses.adv_gen_nusc import TgtMatchingLoss, AdvGenLoss

    Logger.log('NOTE: Using Agent Cross-Attention based optimization (hardcode-style)!')
    Logger.log(f'Running {num_iters} iterations, Ego reacts via Agent Attention')

    # ext_future_mask: inject Sur trajectory when decoding Ego
    sur_mask_for_injection = sur_mask.clone().to(cur_z.device)

    # Optimizer & Loss
    other_z_optim = other_z_all.clone().detach()
    other_z_optim.requires_grad = True
    adv_optim = optim.Adam([other_z_optim], lr=lr)

    tgt_loss = TgtMatchingLoss(loss_weights)
    adv_loss = AdvGenLoss(loss_weights,
                          model.get_att_normalizer().unnormalize(scene_graph.lw),
                          map_idx[scene_graph.batch],
                          map_env,
                          other_z_optim.clone().detach(),
                          scene_graph.ptr,
                          veh_coll_buffer=veh_coll_buffer,
                          crash_loss_min_time=feasibility_time,
                          crash_loss_min_infront=feasibility_infront_min)

    pbar_optim = tqdm.tqdm(range(num_iters), desc='AgentAttn Optim')
    ego_traj = None
    sur_traj = None

    for oidx in pbar_optim:
        # Enable attention debug on first iteration only
        model._debug_attention = (oidx == 0)

        ### STEP 1: Decode Sur trajectory (12 steps) ###
        with torch.no_grad():
            cur_z_sur_only = collate_tgt_other_z(scene_graph, tgt_z, other_z_optim.clone().detach())
            sur_decode_out = model.decode_embedding(
                cur_z_sur_only, embed_info, scene_graph, map_idx, map_env,
                nfuture=future_len
            )
            sur_traj = sur_decode_out['future_pred'][sur_mask].clone().detach()

        ### STEP 2: Ego reacts to Sur via Agent Cross-Attention ###
        # Inject Sur trajectory -> prev_state updated -> Agent Attention sees Sur
        full_ext_future = torch.zeros(NA, future_len, 4).to(sur_traj)
        full_ext_future[sur_mask] = sur_traj

        with torch.no_grad():
            cur_z_ego_react = collate_tgt_other_z(scene_graph, tgt_z, other_z_optim.clone().detach())

            # DEBUG: Ego without Sur injection
            ego_no_inject_out = model.decode_embedding(
                cur_z_ego_react, embed_info, scene_graph, map_idx, map_env,
                nfuture=future_len
            )
            ego_traj_no_inject = ego_no_inject_out['future_pred'][ego_mask].clone().detach()

            # DEBUG TEST: Create FAKE Sur trajectory right in front of Ego
            # Get Ego's past position and put Sur directly ahead
            ego_past_pos = scene_graph.past[ego_mask][:, -1, :4]  # (B, 4) - Ego's last past position
            fake_sur_traj = torch.zeros_like(full_ext_future)
            for t_idx in range(future_len):
                # Put Sur 3 meters ahead of Ego (in Ego's heading direction)
                offset = 3.0 + t_idx * 0.5  # Getting closer over time
                fake_sur_traj[sur_mask, t_idx, 0] = ego_past_pos[:, 0] + offset * ego_past_pos[:, 2]  # x
                fake_sur_traj[sur_mask, t_idx, 1] = ego_past_pos[:, 1] + offset * ego_past_pos[:, 3]  # y
                fake_sur_traj[sur_mask, t_idx, 2] = ego_past_pos[:, 2]  # hx
                fake_sur_traj[sur_mask, t_idx, 3] = ego_past_pos[:, 3]  # hy

            # Ego with FAKE Sur injection (Sur teleported in front of Ego)
            ego_react_out = model.decode_embedding(
                cur_z_ego_react, embed_info, scene_graph, map_idx, map_env,
                ext_future=fake_sur_traj,  # Use FAKE trajectory
                ext_future_mask=sur_mask_for_injection,
                nfuture=future_len
            )
            ego_traj = ego_react_out['future_pred'][ego_mask].clone().detach()

            # DEBUG: Print difference every 10 iterations
            if oidx % 10 == 0:
                diff = (ego_traj_no_inject - ego_traj).abs().mean()
                print(f"DEBUG [iter {oidx}]: Ego traj diff (no_inject vs FAKE inject): {diff.item():.6f}")
                print(f"DEBUG [iter {oidx}]: Ego past pos: {ego_past_pos[0].tolist()}")
                print(f"DEBUG [iter {oidx}]: Fake Sur pos (t=0): {fake_sur_traj[sur_mask][0, 0, :].tolist()}")

        ### STEP 3: Optimize Sur z against reactive Ego ###
        adv_optim.zero_grad()

        # Inject Ego trajectory so Sur optimizes against fixed reactive Ego
        full_ext_future_ego = torch.zeros(NA, future_len, 4).to(ego_traj)
        full_ext_future_ego[ego_mask] = ego_traj
        ego_mask_for_injection = ego_mask.clone().to(cur_z.device)

        tgt_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_optim.clone().detach())
        other_loss_input_z = collate_tgt_other_z(scene_graph, tgt_z.clone().detach(), other_z_optim)

        # Forward pass for tgt loss (with Ego trajectory injected)
        tgt_decoder_out = model.decode_embedding(
            tgt_loss_input_z, embed_info, scene_graph, map_idx, map_env,
            ext_future=full_ext_future_ego,
            ext_future_mask=ego_mask_for_injection,
            nfuture=future_len
        )
        # Forward pass for adv loss (with Ego trajectory injected)
        other_decoder_out = model.decode_embedding(
            other_loss_input_z, embed_info, scene_graph, map_idx, map_env,
            ext_future=full_ext_future_ego,
            ext_future_mask=ego_mask_for_injection,
            nfuture=future_len
        )

        # Compute losses
        loss_dict = dict()
        tgt_match_loss_dict = tgt_loss(
            model.get_normalizer().unnormalize(tgt_decoder_out['future_pred'][ego_mask]),
            model.get_normalizer().unnormalize(ego_traj),
            tgt_z,
            tgt_prior_distrib
        )
        loss_dict = {'tgt_match_' + k: v for k, v in tgt_match_loss_dict.items()}

        adv_loss_dict = adv_loss(
            model.get_normalizer().unnormalize(other_decoder_out['future_pred']),
            model.get_normalizer().unnormalize(ego_traj),
            other_z_optim,
            other_prior_distrib,
            attack_agt_idx=attack_agt_idx
        )
        adv_loss_dict = {'adv_' + k: v for k, v in adv_loss_dict.items()}
        loss_dict = {**loss_dict, **adv_loss_dict}

        loss = loss_dict['tgt_match_loss'] + loss_dict['adv_loss']

        progress_bar_metrics = {}
        for k, v in loss_dict.items():
            if v is not None:
                progress_bar_metrics[k] = torch.mean(v).item()
        pbar_optim.set_postfix(progress_bar_metrics)

        loss.backward()
        adv_optim.step()

    # Final result generation
    other_z_all = other_z_optim.clone().detach()
    cur_z = collate_tgt_other_z(scene_graph, tgt_z, other_z_all)

    with torch.no_grad():
        # Final Sur trajectory
        final_sur_out = model.decode_embedding(cur_z, embed_info, scene_graph, map_idx, map_env, nfuture=future_len)
        sur_traj = final_sur_out['future_pred'][sur_mask].clone().detach()

        # Final Ego trajectory (reacting to final Sur)
        full_ext_future = torch.zeros(NA, future_len, 4).to(sur_traj)
        full_ext_future[sur_mask] = sur_traj

        final_ego_out = model.decode_embedding(
            cur_z, embed_info, scene_graph, map_idx, map_env,
            ext_future=full_ext_future,
            ext_future_mask=sur_mask_for_injection,
            nfuture=future_len
        )
        ego_traj = final_ego_out['future_pred'][ego_mask].clone().detach()

    final_result_traj = torch.zeros(NA, 1, future_len, 4).to(ego_traj)
    final_result_traj[ego_mask, 0] = ego_traj
    final_result_traj[sur_mask, 0] = sur_traj

    final_decoder_out = final_ego_out

    # Compute final metrics
    adv_loss_final = AdvGenLoss(loss_weights,
                                model.get_att_normalizer().unnormalize(scene_graph.lw),
                                map_idx[scene_graph.batch],
                                map_env,
                                cur_z[sur_mask].clone().detach(),
                                scene_graph.ptr,
                                veh_coll_buffer=veh_coll_buffer,
                                crash_loss_min_time=feasibility_time,
                                crash_loss_min_infront=feasibility_infront_min)

    tgt_traj = final_result_traj[ego_inds, torch.zeros_like(ego_inds)]
    adv_loss_dict = adv_loss_final(
        model.get_normalizer().unnormalize(final_result_traj[:, 0]),
        model.get_normalizer().unnormalize(tgt_traj),
        cur_z[sur_mask].clone().detach(),
        other_prior_distrib,
        return_mins=True
    )

    cur_min_agt = cur_min_t = None
    if 'min_agt' in adv_loss_dict:
        cur_min_agt = adv_loss_dict['min_agt'] + scene_graph.ptr[:-1].cpu().numpy()
        print('Final min agt: ' + str(cur_min_agt))
    if 'min_t' in adv_loss_dict:
        cur_min_t = adv_loss_dict['min_t']
        print('Final min t: ' + str(cur_min_t))

    return cur_z, final_result_traj, final_decoder_out, cur_min_agt, cur_min_t
############### END: run_adv_gen_optim_agent_attention ###############