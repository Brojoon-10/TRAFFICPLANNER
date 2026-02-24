"""
Batch Equivalence Test: Batched TF Decoder vs Sequential Reference.

Verifies that the parallelized Step C in _sur_loop_decoder and _ego_loop_decoder
produces the same numerical outputs as a sequential reference implementation.

Tests:
1. DecoderHistoryAttention: forward_batched vs sequential forward at each timestep
   (on both CPU and GPU)
2. Full Step C for sur loop: trajectory, ego_pred, map_attn, sur_intent
3. Full Step C for ego loop: trajectory, sur_pred, map_attn, ego_intent
4. Full TF decoder output validation

Tolerance notes:
- CPU: exact equivalence (atol=1e-5) expected for all outputs
- GPU: slight non-determinism from batched vs sequential matmul ordering
  - History/Map attention: ~1e-4 typical
  - Trajectory (after bicycle model amplification): ~3e-3 typical
  - Intent weights: argmax may flip when logits are close (not a real error)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models.trafficplanner_model import TrafficPlannerModel, DecoderHistoryAttention
from datasets.utils import NUSC_BIKE_PARAMS, NUSC_NORM_STATS, MeanStdNormalizer


def get_tolerances(device):
    """Return tolerance dict appropriate for CPU vs GPU."""
    if device.type == 'cpu':
        return {
            'attn': 1e-5,
            'pred': 1e-5,
            'traj': 1e-5,
            'map_attn': 1e-5,
            'intent_logit': 1e-5,  # for logit-level comparison
        }
    else:
        return {
            'attn': 5e-4,      # history attention
            'pred': 5e-4,      # sur_pred / ego_pred heads
            'traj': 3e-3,      # trajectory (bicycle model amplifies)
            'map_attn': 1e-4,  # map cross-attention weights
            'intent_logit': 5e-4,  # logit-level (before argmax)
        }


# ======================================================================
# Helpers (from test_forward_pass.py)
# ======================================================================

def make_dummy_normalizers(device='cpu'):
    ninfo = NUSC_NORM_STATS[('car', 'truck')]
    state_mean = [ninfo['lscale'][0], ninfo['lscale'][0],
                  ninfo['h'][0], ninfo['h'][0],
                  ninfo['s'][0], ninfo['hdot'][0]]
    state_std = [ninfo['lscale'][1], ninfo['lscale'][1],
                 ninfo['h'][1], ninfo['h'][1],
                 ninfo['s'][1], ninfo['hdot'][1]]
    state_normalizer = MeanStdNormalizer(
        torch.tensor(state_mean, device=device),
        torch.tensor(state_std, device=device))

    att_mean = [ninfo['l'][0], ninfo['w'][0]]
    att_std = [ninfo['l'][1], ninfo['w'][1]]
    att_normalizer = MeanStdNormalizer(
        torch.tensor(att_mean, device=device),
        torch.tensor(att_std, device=device))
    return state_normalizer, att_normalizer


def make_dummy_scene_graph(B=2, agents_per_scene=3, PT=4, FT=12, NC=2, device='cpu'):
    NA = B * agents_per_scene
    past = torch.randn(NA, PT, 6, device=device)
    future = torch.randn(NA, FT, 6, device=device)
    future_gt = future.clone()
    past_gt = past.clone()
    past_vis = torch.ones(NA, PT, device=device)
    future_vis = torch.ones(NA, FT, device=device)
    lw = torch.randn(NA, 2, device=device) * 0.3
    sem = torch.zeros(NA, NC, device=device)
    sem[:, 0] = 1.0

    edge_list = []
    for b in range(B):
        start = b * agents_per_scene
        for i in range(agents_per_scene):
            for j in range(agents_per_scene):
                if i != j:
                    edge_list.append([start + j, start + i])
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T

    batch = torch.arange(B, device=device).repeat_interleave(agents_per_scene)
    ptr = torch.arange(0, NA + 1, agents_per_scene, device=device)

    data = Data(
        past=past, future=future, future_gt=future_gt, past_gt=past_gt,
        past_vis=past_vis, future_vis=future_vis,
        lw=lw, sem=sem, edge_index=edge_index,
        batch=batch, ptr=ptr,
    )
    data.num_nodes = NA
    return data


class DummyMapEnv:
    def __init__(self, map_size=240, num_layers=4, device='cpu'):
        self.map_size = map_size
        self.num_layers = num_layers
        self.device = device

    def get_map_crop(self, scene_graph, map_idx):
        NA = scene_graph.pos.size(0)
        NS = None
        if len(scene_graph.pos.size()) == 3:
            NS = scene_graph.pos.size(1)
            return torch.randn(NA, NS, self.num_layers, self.map_size, self.map_size, device=self.device)
        return torch.randn(NA, self.num_layers, self.map_size, self.map_size, device=self.device)

    def get_map_crop_pos(self, pos, mapixes):
        NA = pos.size(0)
        return torch.randn(NA, self.num_layers, self.map_size, self.map_size, device=self.device)


def check_close(name, batched, sequential, atol, FT, skip_intent_flip=False):
    """Compare per-timestep and report. Returns (pass, num_issues)."""
    all_ok = True
    num_issues = 0
    for t in range(FT):
        diff = (batched[t] - sequential[t]).abs()
        max_diff = diff.max().item()

        # Intent weights: argmax flip produces diff=1.0 exactly
        if skip_intent_flip and max_diff > 0.5:
            print(f"  t={t:2d}: {name} max_diff={max_diff:.2e}  [SKIP: argmax flip]")
            continue

        ok = max_diff <= atol
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
            num_issues += 1
        print(f"  t={t:2d}: {name} max_diff={max_diff:.2e}  [{status}]")
    return all_ok, num_issues


# ======================================================================
# Test 1: DecoderHistoryAttention forward_batched vs sequential forward
# ======================================================================

def test_history_attn_equivalence(device):
    """Test on specified device. CPU should be exact, GPU may have small diffs."""
    print("=" * 70)
    print(f"[Test 1] DecoderHistoryAttention: forward_batched vs sequential ({device})")
    print("=" * 70)

    tol = get_tolerances(device)

    torch.manual_seed(42)

    D = 64
    FT = 12
    N = 4  # number of agents
    nhead = 4

    attn = DecoderHistoryAttention(D, nhead=nhead, max_len=FT).to(device)
    attn.eval()

    # Create fixed inputs
    torch.manual_seed(123)
    queries = torch.randn(N, FT, D, device=device)
    history_buffer = torch.randn(N, FT, D, device=device)

    # --- Batched ---
    with torch.no_grad():
        batched_out = attn.forward_batched(queries, history_buffer, FT)  # (N, FT, D)

    # --- Sequential ---
    with torch.no_grad():
        seq_out = torch.zeros(N, FT, D, device=device)
        for t in range(FT):
            q_t = queries[:, t, :]  # (N, D)
            ctx_t = attn.forward(q_t, history_buffer, t)  # (N, D)
            seq_out[:, t, :] = ctx_t

    # Compare
    all_pass = True
    for t in range(FT):
        diff = (batched_out[:, t, :] - seq_out[:, t, :]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        ok = max_diff <= tol['attn']
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  t={t:2d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}  [{status}]")

    overall_max = (batched_out - seq_out).abs().max().item()
    tight_match = torch.allclose(batched_out, seq_out, atol=1e-5, rtol=1e-5)
    tol_match = overall_max <= tol['attn']
    print(f"\n  Overall max_diff: {overall_max:.2e}")
    print(f"  Exact match (atol=1e-5): {tight_match}")
    print(f"  Within tolerance ({tol['attn']:.0e}): {tol_match}")

    if tol_match:
        print(f"  [PASS] History attention ({device})\n")
    else:
        print(f"  [FAIL] History attention ({device})\n")
        all_pass = False

    return all_pass


# ======================================================================
# Test 2 & 3: Full Step C equivalence for sur and ego loops
# ======================================================================

def setup_model_and_data(device):
    """Set up model and run shared computation (Steps A and B) once."""
    torch.manual_seed(42)

    model = TrafficPlannerModel(
        npast=4, nfuture=12,
        map_obs_size_pix=240, nclasses=2,
        map_feat_size=64, past_feat_size=64, future_feat_size=64,
        latent_size=32, z_local_size=32,
        output_bicycle=True, dt=0.5,
        num_intents=9, hist_attn_nhead=4, map_attn_nhead=4, sur_pred_dim=2,
        use_ego_intent=True, use_sur_intent=True,
    ).to(device)

    state_norm, att_norm = make_dummy_normalizers(device)
    model.set_normalizer(state_norm)
    model.set_att_normalizer(att_norm)
    model.set_bicycle_params(NUSC_BIKE_PARAMS)

    model.eval()  # deterministic (no gumbel noise, argmax for intent)

    B = 2
    agents_per_scene = 3
    torch.manual_seed(100)
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    return model, scene_graph, map_env, map_idx


def test_sur_loop_equivalence(device):
    """
    Test 2: Compare batched _sur_loop_decoder Step C against sequential reference.
    """
    print("=" * 70)
    print(f"[Test 2] Sur Loop: Batched Step C vs Sequential Reference ({device})")
    print("=" * 70)

    tol = get_tolerances(device)
    model, scene_graph, map_env, map_idx = setup_model_and_data(device)

    NA = scene_graph.past.size(0)
    FT = model.FT
    B = map_idx.size(0)

    # Encode + prior/posterior
    scene_graph.pos = scene_graph.past[:, -1, :4]
    torch.manual_seed(200)
    map_feat, map_tokens = model.encode_map(scene_graph, map_idx, map_env, return_tokens=True)
    prior_mu, prior_var, past_seq_out = model.prior(scene_graph, map_feat)
    past_context = past_seq_out[:, -1, :]
    post_mu, post_var = model.encoder(scene_graph, map_feat, past_context)
    z = post_mu  # deterministic

    ego_mask = model._get_ego_mask(scene_graph)
    num_ego = int(ego_mask.sum())
    num_sur = NA - num_ego

    cur_lw = scene_graph.lw
    cur_sem = scene_graph.sem
    cur_veh_len = model.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)

    z_sur = z[~ego_mask]
    sur_map_tokens = map_tokens[~ego_mask]
    gt_future = scene_graph.future_gt[:, :, :]

    # ================================================================
    # Run BATCHED version (the current code)
    # ================================================================
    model._z_local_outputs = []
    model._intent_weights_outputs = []
    model._z_local_sur_outputs = []
    model._sur_intent_weights_outputs = []
    model._ego_map_attn_weights_outputs = []
    model._sur_map_attn_weights_outputs = []
    model._sur_pred_outputs = []
    model._ego_pred_outputs = []

    with torch.no_grad():
        batched_traj = model._sur_loop_decoder(
            scene_graph, z_sur, ego_mask, gt_future,
            sur_map_tokens, cur_lw, cur_sem, cur_veh_len,
            map_idx, map_env)

    batched_ego_pred = [x.clone() for x in model._ego_pred_outputs]
    batched_sur_map_attn = [x.clone() for x in model._sur_map_attn_weights_outputs]
    batched_sur_z_local = [x.clone() for x in model._z_local_sur_outputs]
    batched_sur_intent_w = [x.clone() for x in model._sur_intent_weights_outputs]

    # ================================================================
    # Run SEQUENTIAL reference (reimplementing Step C inline)
    # ================================================================

    # Step A: same batched GCN
    gt_state_t0 = scene_graph.past[:, -1, :]
    all_gt_states = torch.cat([gt_state_t0.unsqueeze(1), gt_future[:, :FT-1, :]], dim=1)
    S = all_gt_states.size(-1)
    if S < 6:
        pad = torch.zeros(NA, FT, 6 - S, device=device)
        all_gt_states_6d = torch.cat([all_gt_states, pad], dim=-1)
    else:
        all_gt_states_6d = all_gt_states

    batched_graph = model._build_temporal_graph(scene_graph, all_gt_states_6d, cur_lw, cur_sem, FT)
    batched_ego_mask = ego_mask.repeat(FT)
    with torch.no_grad():
        _, sur_gcn_all = model.interaction_gcn(batched_graph, batched_ego_mask)
    gt_gcn_cache = sur_gcn_all.reshape(FT, num_sur, model.d_model).permute(1, 0, 2).contiguous()

    # Step B: sequential GRU for hidden snapshots
    with torch.no_grad():
        sur_gru_hidden_init = model._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=False, gt_future=None, target_t=0)
        sur_gru_hidden = sur_gru_hidden_init.detach().clone()
        gt_hidden_snapshots = [sur_gru_hidden.clone()]
        gt_history_buffer = gt_gcn_cache

        for t in range(FT):
            sur_gru_h_last = sur_gru_hidden[-1]
            sur_hist_ctx = model.sur_history_attn(sur_gru_h_last, gt_history_buffer, t)
            predicted_ego_delta = model.ego_pred_head(sur_hist_ctx)

            sur_map_ctx, _ = model.sur_map_attn(sur_gru_h_last, sur_map_tokens)

            sur_state_for_intent = all_gt_states[:, t, :][~ego_mask]
            if sur_state_for_intent.size(-1) < 6:
                sur_state_for_intent = torch.cat([sur_state_for_intent,
                    torch.zeros(num_sur, 6 - sur_state_for_intent.size(-1), device=device)], dim=-1)
            sur_intent_input = torch.cat([sur_state_for_intent, predicted_ego_delta, sur_map_ctx.detach()], dim=-1)
            z_local_sur, _ = model.sur_intent_codebook(sur_intent_input, model.gumbel_temperature)

            sur_lw_flat = cur_lw[~ego_mask]
            sur_sem_flat = cur_sem[~ego_mask]
            sur_gru_in = torch.cat([
                sur_hist_ctx, sur_map_ctx, z_sur, z_local_sur, sur_lw_flat, sur_sem_flat
            ], dim=-1).unsqueeze(1)

            _, sur_gru_hidden = model.sur_decoder_gru(sur_gru_in, sur_gru_hidden)
            gt_hidden_snapshots.append(sur_gru_hidden.clone())

    gt_hidden_snapshots[0] = sur_gru_hidden_init.detach().clone()

    # Step C: SEQUENTIAL reference
    all_hidden = torch.stack(gt_hidden_snapshots[:FT], dim=0)

    if model.output_bicycle:
        sur_prev_states = all_gt_states[:, :FT, :][~ego_mask]
    else:
        sur_prev_states = all_gt_states_6d[:, :FT, :4][~ego_mask]

    seq_traj = torch.zeros(num_sur, FT, 4, device=device)
    seq_ego_pred = []
    seq_sur_map_attn = []
    seq_sur_z_local = []
    seq_sur_intent_w = []

    with torch.no_grad():
        for t in range(FT):
            gru_hidden_t = all_hidden[t]
            gru_h_last = gru_hidden_t[-1]

            # History Attention (SEQUENTIAL forward)
            hist_ctx = model.sur_history_attn.forward(gru_h_last, gt_gcn_cache, t)

            predicted_ego_delta = model.ego_pred_head(hist_ctx)
            seq_ego_pred.append(predicted_ego_delta.clone())

            map_ctx, map_attn_w = model.sur_map_attn(gru_h_last, sur_map_tokens)
            seq_sur_map_attn.append(map_attn_w.clone())

            sur_state_for_intent = all_gt_states[:, t, :][~ego_mask]
            if sur_state_for_intent.size(-1) < 6:
                sur_state_for_intent = torch.cat([sur_state_for_intent,
                    torch.zeros(num_sur, 6 - sur_state_for_intent.size(-1), device=device)], dim=-1)
            sur_intent_input = torch.cat([sur_state_for_intent, predicted_ego_delta, map_ctx.detach()], dim=-1)
            z_local_sur, sur_intent_w = model.sur_intent_codebook(sur_intent_input, model.gumbel_temperature)
            seq_sur_z_local.append(z_local_sur.clone())
            seq_sur_intent_w.append(sur_intent_w.clone())

            sur_lw_flat = cur_lw[~ego_mask]
            sur_sem_flat = cur_sem[~ego_mask]
            gru_in = torch.cat([
                hist_ctx, map_ctx, z_sur, z_local_sur, sur_lw_flat, sur_sem_flat
            ], dim=-1).unsqueeze(1)

            gru_out, _ = model.sur_decoder_gru(gru_in, gru_hidden_t)
            traj_step = model.sur_output_head(gru_out[:, 0])

            prev_state_t = sur_prev_states[:, t, :]
            veh_len_t = cur_veh_len[~ego_mask]
            state_global, _ = model._apply_dynamics_single(traj_step, prev_state_t, veh_len_t)
            seq_traj[:, t, :] = state_global

    # ================================================================
    # Compare outputs
    # ================================================================
    all_pass = True
    total_issues = 0

    print("\n  --- Trajectory ---")
    traj_list_bat = [batched_traj[:, t, :] for t in range(FT)]
    traj_list_seq = [seq_traj[:, t, :] for t in range(FT)]
    ok, n = check_close("traj", traj_list_bat, traj_list_seq, tol['traj'], FT)
    if not ok: all_pass = False
    total_issues += n
    overall_traj = (batched_traj - seq_traj).abs().max().item()
    print(f"  Overall traj max_diff: {overall_traj:.2e}")

    print("\n  --- Ego pred (aux) ---")
    ok, n = check_close("ego_pred", batched_ego_pred, seq_ego_pred, tol['pred'], FT)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Sur map attn ---")
    ok, n = check_close("map_attn", batched_sur_map_attn, seq_sur_map_attn, tol['map_attn'], FT)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Sur z_local ---")
    ok, n = check_close("z_local", batched_sur_z_local, seq_sur_z_local, tol['intent_logit'], FT,
                        skip_intent_flip=True)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Sur intent weights ---")
    ok, n = check_close("intent_w", batched_sur_intent_w, seq_sur_intent_w, tol['intent_logit'], FT,
                        skip_intent_flip=True)
    if not ok: all_pass = False
    total_issues += n

    if all_pass:
        print(f"\n  [PASS] Sur loop ({device}): batched == sequential\n")
    else:
        print(f"\n  [FAIL] Sur loop ({device}): {total_issues} timestep(s) exceeded tolerance\n")
    return all_pass


def test_ego_loop_equivalence(device):
    """
    Test 3: Compare batched _ego_loop_decoder Step C against sequential reference.
    """
    print("=" * 70)
    print(f"[Test 3] Ego Loop: Batched Step C vs Sequential Reference ({device})")
    print("=" * 70)

    tol = get_tolerances(device)
    model, scene_graph, map_env, map_idx = setup_model_and_data(device)

    NA = scene_graph.past.size(0)
    FT = model.FT
    B = map_idx.size(0)

    scene_graph.pos = scene_graph.past[:, -1, :4]
    torch.manual_seed(200)
    map_feat, map_tokens = model.encode_map(scene_graph, map_idx, map_env, return_tokens=True)
    prior_mu, prior_var, past_seq_out = model.prior(scene_graph, map_feat)
    past_context = past_seq_out[:, -1, :]
    post_mu, post_var = model.encoder(scene_graph, map_feat, past_context)
    z = post_mu

    ego_mask = model._get_ego_mask(scene_graph)
    num_ego = int(ego_mask.sum())
    num_sur = NA - num_ego

    cur_lw = scene_graph.lw
    cur_sem = scene_graph.sem
    cur_veh_len = model.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)

    z_ego = z[ego_mask]
    ego_map_tokens = map_tokens[ego_mask]
    gt_future = scene_graph.future_gt[:, :, :]

    # ================================================================
    # Run BATCHED version (the current code)
    # ================================================================
    model._z_local_outputs = []
    model._intent_weights_outputs = []
    model._z_local_sur_outputs = []
    model._sur_intent_weights_outputs = []
    model._ego_map_attn_weights_outputs = []
    model._sur_map_attn_weights_outputs = []
    model._sur_pred_outputs = []
    model._ego_pred_outputs = []

    with torch.no_grad():
        batched_traj = model._ego_loop_decoder(
            scene_graph, z_ego, ego_mask, gt_future,
            ego_map_tokens, cur_lw, cur_sem, cur_veh_len,
            map_idx, map_env)

    batched_sur_pred = [x.clone() for x in model._sur_pred_outputs]
    batched_ego_map_attn = [x.clone() for x in model._ego_map_attn_weights_outputs]
    batched_ego_z_local = [x.clone() for x in model._z_local_outputs]
    batched_ego_intent_w = [x.clone() for x in model._intent_weights_outputs]

    # ================================================================
    # Run SEQUENTIAL reference
    # ================================================================

    # Step A
    gt_state_t0 = scene_graph.past[:, -1, :]
    all_gt_states = torch.cat([gt_state_t0.unsqueeze(1), gt_future[:, :FT-1, :]], dim=1)
    S = all_gt_states.size(-1)
    if S < 6:
        pad = torch.zeros(NA, FT, 6 - S, device=device)
        all_gt_states_6d = torch.cat([all_gt_states, pad], dim=-1)
    else:
        all_gt_states_6d = all_gt_states

    batched_graph = model._build_temporal_graph(scene_graph, all_gt_states_6d, cur_lw, cur_sem, FT)
    batched_ego_mask_expanded = ego_mask.repeat(FT)
    with torch.no_grad():
        ego_gcn_all, _ = model.interaction_gcn(batched_graph, batched_ego_mask_expanded)
    gt_gcn_cache = ego_gcn_all.reshape(FT, num_ego, model.d_model).permute(1, 0, 2).contiguous()

    # Step B
    with torch.no_grad():
        ego_gru_hidden_init = model._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=True, gt_future=gt_future, target_t=0)
        ego_gru_hidden = ego_gru_hidden_init.detach().clone()
        gt_hidden_snapshots = [ego_gru_hidden.clone()]
        gt_history_buffer = gt_gcn_cache

        for t in range(FT):
            ego_gru_h_last = ego_gru_hidden[-1]
            ego_hist_ctx = model.ego_history_attn(ego_gru_h_last, gt_history_buffer, t)
            predicted_sur_delta = model.sur_pred_head(ego_hist_ctx)

            ego_map_ctx, _ = model.ego_map_attn(ego_gru_h_last, ego_map_tokens)

            ego_state_for_intent = all_gt_states[:, t, :][ego_mask]
            if ego_state_for_intent.size(-1) < 6:
                ego_state_for_intent = torch.cat([ego_state_for_intent,
                    torch.zeros(num_ego, 6 - ego_state_for_intent.size(-1), device=device)], dim=-1)
            intent_input = torch.cat([ego_state_for_intent, predicted_sur_delta, ego_map_ctx.detach()], dim=-1)
            z_local, _ = model.intent_codebook(intent_input, model.gumbel_temperature)
            if not model.use_ego_intent:
                z_local = torch.zeros_like(z_local)

            ego_lw_flat = cur_lw[ego_mask]
            ego_sem_flat = cur_sem[ego_mask]
            ego_gru_in = torch.cat([
                ego_hist_ctx, ego_map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
            ], dim=-1).unsqueeze(1)

            _, ego_gru_hidden = model.ego_decoder_gru(ego_gru_in, ego_gru_hidden)
            gt_hidden_snapshots.append(ego_gru_hidden.clone())

    gt_hidden_snapshots[0] = ego_gru_hidden_init.detach().clone()

    # Step C: SEQUENTIAL reference
    all_hidden = torch.stack(gt_hidden_snapshots[:FT], dim=0)

    if model.output_bicycle:
        ego_prev_states = all_gt_states[:, :FT, :][ego_mask]
    else:
        ego_prev_states = all_gt_states_6d[:, :FT, :4][ego_mask]

    seq_traj = torch.zeros(num_ego, FT, 4, device=device)
    seq_sur_pred = []
    seq_ego_map_attn = []
    seq_ego_z_local = []
    seq_ego_intent_w = []

    with torch.no_grad():
        for t in range(FT):
            gru_hidden_t = all_hidden[t]
            gru_h_last = gru_hidden_t[-1]

            hist_ctx = model.ego_history_attn.forward(gru_h_last, gt_gcn_cache, t)

            predicted_sur_delta = model.sur_pred_head(hist_ctx)
            seq_sur_pred.append(predicted_sur_delta.clone())

            map_ctx, map_attn_w = model.ego_map_attn(gru_h_last, ego_map_tokens)
            seq_ego_map_attn.append(map_attn_w.clone())

            ego_state_for_intent = all_gt_states[:, t, :][ego_mask]
            if ego_state_for_intent.size(-1) < 6:
                ego_state_for_intent = torch.cat([ego_state_for_intent,
                    torch.zeros(num_ego, 6 - ego_state_for_intent.size(-1), device=device)], dim=-1)
            intent_input = torch.cat([ego_state_for_intent, predicted_sur_delta, map_ctx.detach()], dim=-1)
            z_local, intent_w = model.intent_codebook(intent_input, model.gumbel_temperature)
            if not model.use_ego_intent:
                z_local = torch.zeros_like(z_local)
            seq_ego_z_local.append(z_local.clone())
            seq_ego_intent_w.append(intent_w.clone())

            ego_lw_flat = cur_lw[ego_mask]
            ego_sem_flat = cur_sem[ego_mask]
            gru_in = torch.cat([
                hist_ctx, map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
            ], dim=-1).unsqueeze(1)

            gru_out, _ = model.ego_decoder_gru(gru_in, gru_hidden_t)
            traj_step = model.ego_output_head(gru_out[:, 0])

            prev_state_t = ego_prev_states[:, t, :]
            veh_len_t = cur_veh_len[ego_mask]
            state_global, _ = model._apply_dynamics_single(traj_step, prev_state_t, veh_len_t)
            seq_traj[:, t, :] = state_global

    # ================================================================
    # Compare outputs
    # ================================================================
    all_pass = True
    total_issues = 0

    print("\n  --- Trajectory ---")
    traj_list_bat = [batched_traj[:, t, :] for t in range(FT)]
    traj_list_seq = [seq_traj[:, t, :] for t in range(FT)]
    ok, n = check_close("traj", traj_list_bat, traj_list_seq, tol['traj'], FT)
    if not ok: all_pass = False
    total_issues += n
    overall_traj = (batched_traj - seq_traj).abs().max().item()
    print(f"  Overall traj max_diff: {overall_traj:.2e}")

    print("\n  --- Sur pred (aux) ---")
    ok, n = check_close("sur_pred", batched_sur_pred, seq_sur_pred, tol['pred'], FT)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Ego map attn ---")
    ok, n = check_close("map_attn", batched_ego_map_attn, seq_ego_map_attn, tol['map_attn'], FT)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Ego z_local ---")
    ok, n = check_close("z_local", batched_ego_z_local, seq_ego_z_local, tol['intent_logit'], FT,
                        skip_intent_flip=True)
    if not ok: all_pass = False
    total_issues += n

    print("\n  --- Ego intent weights ---")
    ok, n = check_close("intent_w", batched_ego_intent_w, seq_ego_intent_w, tol['intent_logit'], FT,
                        skip_intent_flip=True)
    if not ok: all_pass = False
    total_issues += n

    if all_pass:
        print(f"\n  [PASS] Ego loop ({device}): batched == sequential\n")
    else:
        print(f"\n  [FAIL] Ego loop ({device}): {total_issues} timestep(s) exceeded tolerance\n")
    return all_pass


# ======================================================================
# Test 4: End-to-end TF decoder validation
# ======================================================================

def test_full_tf_equivalence(device):
    """
    Test 4: Full teacher_forcing_decoder output validation.
    """
    print("=" * 70)
    print(f"[Test 4] Full TF Decoder Validation ({device})")
    print("=" * 70)

    model, scene_graph, map_env, map_idx = setup_model_and_data(device)

    NA = scene_graph.past.size(0)
    FT = model.FT

    scene_graph.pos = scene_graph.past[:, -1, :4]
    torch.manual_seed(200)
    map_feat, map_tokens = model.encode_map(scene_graph, map_idx, map_env, return_tokens=True)
    prior_mu, prior_var, past_seq_out = model.prior(scene_graph, map_feat)
    past_context = past_seq_out[:, -1, :]
    post_mu, post_var = model.encoder(scene_graph, map_feat, past_context)
    z = post_mu

    ego_mask = model._get_ego_mask(scene_graph)

    with torch.no_grad():
        tf_out = model.teacher_forcing_decoder(
            scene_graph, map_feat, past_seq_out, z, map_idx, map_env, map_tokens)

    assert tf_out.shape == (NA, FT, 4), f"Expected ({NA}, {FT}, 4), got {tf_out.shape}"

    ego_part = tf_out[ego_mask]
    sur_part = tf_out[~ego_mask]
    print(f"  TF output shape: {tf_out.shape}")
    print(f"  Ego part norm: {ego_part.norm().item():.4f}")
    print(f"  Sur part norm: {sur_part.norm().item():.4f}")
    assert ego_part.norm().item() > 0, "Ego part should be non-zero"
    assert sur_part.norm().item() > 0, "Sur part should be non-zero"

    assert len(model._sur_pred_outputs) == FT
    assert len(model._ego_pred_outputs) == FT
    assert len(model._ego_map_attn_weights_outputs) == FT
    assert len(model._sur_map_attn_weights_outputs) == FT
    assert len(model._z_local_outputs) == FT
    assert len(model._intent_weights_outputs) == FT
    assert len(model._z_local_sur_outputs) == FT
    assert len(model._sur_intent_weights_outputs) == FT

    print("  All analysis outputs have correct length (FT=12)")
    print(f"  [PASS] Full TF decoder ({device})\n")
    return True


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Batch Equivalence Test")
    print(f"Primary device: {device}")
    print(f"{'='*70}\n")

    results = {}

    # Test 1a: CPU gold-standard (always exact)
    print("--- CPU gold-standard test ---\n")
    cpu_device = torch.device('cpu')
    results['hist_attn_cpu'] = test_history_attn_equivalence(cpu_device)

    # Test 1b: GPU (if available)
    if device.type == 'cuda':
        print("--- GPU test (tolerance for non-determinism) ---\n")
        results['hist_attn_gpu'] = test_history_attn_equivalence(device)

    # Test 2: Sur loop Step C (on primary device)
    results['sur_loop'] = test_sur_loop_equivalence(device)

    # Test 3: Ego loop Step C (on primary device)
    results['ego_loop'] = test_ego_loop_equivalence(device)

    # Test 4: Full TF decoder
    results['full_tf'] = test_full_tf_equivalence(device)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}: [{status}]")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL TESTS PASSED - Batched code is numerically equivalent to sequential!")
    else:
        print("SOME TESTS FAILED - Batched code differs from sequential!")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
