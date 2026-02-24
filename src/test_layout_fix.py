"""
Numerical equivalence test: sequential Step C vs batched Step C.
Both use the SAME gt_gcn_cache and gt_hidden_snapshots (Step A, B shared).
Verifies that the time-major layout fix produces identical results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')  # CPU for deterministic comparison

# ============================================================
# 1. Build model (same as test_forward_pass.py)
# ============================================================
from models.trafficplanner_model import TrafficPlannerModel
from datasets.utils import MeanStdNormalizer, NUSC_BIKE_PARAMS, NUSC_NORM_STATS

model = TrafficPlannerModel(
    npast=4, nfuture=12,
    map_obs_size_pix=240, nclasses=2,
    map_feat_size=64, past_feat_size=64, future_feat_size=64,
    latent_size=32, z_local_size=32,
    output_bicycle=True, dt=0.5,
    num_intents=9, hist_attn_nhead=4, map_attn_nhead=4, sur_pred_dim=2,
    use_ego_intent=True, use_sur_intent=True,
).to(device)
# Set normalizers and bicycle params
ninfo = NUSC_NORM_STATS[('car', 'truck')]
state_mean = [ninfo['lscale'][0], ninfo['lscale'][0],
              ninfo['h'][0], ninfo['h'][0],
              ninfo['s'][0], ninfo['hdot'][0]]
state_std = [ninfo['lscale'][1], ninfo['lscale'][1],
             ninfo['h'][1], ninfo['h'][1],
             ninfo['s'][1], ninfo['hdot'][1]]
state_norm = MeanStdNormalizer(torch.tensor(state_mean), torch.tensor(state_std))
att_mean = [ninfo['l'][0], ninfo['w'][0]]
att_std = [ninfo['l'][1], ninfo['w'][1]]
att_norm = MeanStdNormalizer(torch.tensor(att_mean), torch.tensor(att_std))
model.set_normalizer(state_norm)
model.set_att_normalizer(att_norm)
model.set_bicycle_params({k: (v[0], v[1]) if isinstance(v, tuple) else v for k, v in NUSC_BIKE_PARAMS.items()})

model.eval()

FT = model.FT
d_model = model.d_model
print(f"FT={FT}, d_model={d_model}")

# ============================================================
# 2. Create fake data for sur loop
# ============================================================
NA = 4  # 4 agents
num_ego = 1
num_sur = NA - num_ego

# Fake gt_gcn_cache and gt_hidden_snapshots (shared between sequential and batched)
gt_gcn_cache = torch.randn(num_sur, FT, d_model, device=device)
gt_hidden_snapshots_raw = []
for t in range(FT + 1):
    h = torch.randn(3, num_sur, d_model, device=device)
    gt_hidden_snapshots_raw.append(h)

# Fake prev_states, map_tokens, lw, sem, veh_len, z
all_gt_states_sur = torch.randn(num_sur, FT, 6, device=device)
sur_map_tokens = torch.randn(num_sur, model.map_num_tokens, model.map_token_ch, device=device)
cur_lw_sur = torch.randn(num_sur, 2, device=device)
cur_sem_sur = torch.randn(num_sur, model.NC, device=device)
cur_veh_len_sur = torch.ones(num_sur, 1, device=device) * 4.5
z_sur = torch.randn(num_sur, model.z_size, device=device)

# ============================================================
# 3. Sequential Step C (reference)
# ============================================================
print("\n=== Sequential Step C (reference) ===")

seq_traj_list = []
seq_hist_list = []
seq_h_last_list = []
seq_map_ctx_list = []
seq_gru_out_list = []
seq_traj_step_list = []

with torch.no_grad():
    for t in range(FT):
        h_init = gt_hidden_snapshots_raw[t]  # (3, num_sur, D)
        gru_h_last = h_init[-1]  # (num_sur, D)
        seq_h_last_list.append(gru_h_last.clone())

        # History Attention
        hist_ctx = model.sur_history_attn(gru_h_last, gt_gcn_cache, t)  # (num_sur, D)
        seq_hist_list.append(hist_ctx.clone())

        # Aux pred
        ego_delta = model.ego_pred_head(hist_ctx)

        # Map Attention
        map_ctx, _ = model.sur_map_attn(gru_h_last, sur_map_tokens)
        seq_map_ctx_list.append(map_ctx.clone())

        # Intent
        state_for_intent = all_gt_states_sur[:, t, :]
        intent_in = torch.cat([state_for_intent, ego_delta, map_ctx.detach()], dim=-1)
        if model.use_sur_intent:
            z_local, _ = model.sur_intent_codebook(intent_in, model.gumbel_temperature)
        else:
            z_local = torch.zeros(num_sur, model.intent_dim, device=device)

        # GRU 1-step
        gru_in = torch.cat([hist_ctx, map_ctx, z_sur, z_local, cur_lw_sur, cur_sem_sur], dim=-1).unsqueeze(1)
        gru_out, _ = model.sur_decoder_gru(gru_in, h_init)
        seq_gru_out_list.append(gru_out[:, 0].clone())

        traj_step = model.sur_output_head(gru_out[:, 0])
        seq_traj_step_list.append(traj_step.clone())

        # Bicycle
        prev_state = all_gt_states_sur[:, t, :]
        state_global, _ = model._apply_dynamics_single(traj_step, prev_state, cur_veh_len_sur)
        seq_traj_list.append(state_global)

seq_traj = torch.stack(seq_traj_list, dim=1)  # (num_sur, FT, 4)
print(f"  seq_traj shape: {seq_traj.shape}")

# ============================================================
# 4. Batched Step C (with time-major layout fix)
# ============================================================
print("\n=== Batched Step C (time-major) ===")

with torch.no_grad():
    all_hidden = torch.stack(gt_hidden_snapshots_raw[:FT], dim=0)  # (FT, 3, num_sur, D)

    # prev_state — time-major
    sur_prev_flat = all_gt_states_sur.permute(1, 0, 2).reshape(num_sur * FT, -1)  # time-major

    # History Attention (batched)
    queries = all_hidden[:, -1, :, :]  # (FT, num_sur, D)
    queries = queries.permute(1, 0, 2)  # (num_sur, FT, D)
    hist_ctx_all = model.sur_history_attn.forward_batched(queries, gt_gcn_cache, FT)  # (num_sur, FT, D)
    hist_ctx_flat = hist_ctx_all.permute(1, 0, 2).reshape(num_sur * FT, d_model)  # time-major

    # Aux pred
    ego_delta_flat = model.ego_pred_head(hist_ctx_flat)  # time-major

    # Map Attention — time-major
    map_tokens_flat = sur_map_tokens.unsqueeze(0).expand(FT, -1, -1, -1).reshape(
        num_sur * FT, -1, model.map_token_ch)
    gru_h_last_flat = all_hidden[:, -1, :, :].reshape(num_sur * FT, d_model)  # time-major (no permute)
    map_ctx_flat, _ = model.sur_map_attn(gru_h_last_flat, map_tokens_flat)

    # Intent — time-major
    states_intent_flat = all_gt_states_sur.permute(1, 0, 2).reshape(num_sur * FT, -1)  # time-major
    intent_in_flat = torch.cat([states_intent_flat, ego_delta_flat, map_ctx_flat.detach()], dim=-1)
    if model.use_sur_intent:
        z_local_flat, _ = model.sur_intent_codebook(intent_in_flat, model.gumbel_temperature)
    else:
        z_local_flat = torch.zeros(num_sur * FT, model.intent_dim, device=device)

    # GRU 1-step — all time-major
    lw_flat = cur_lw_sur.unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)
    sem_flat = cur_sem_sur.unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)
    z_flat = z_sur.unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)

    gru_in_flat = torch.cat([
        hist_ctx_flat, map_ctx_flat, z_flat, z_local_flat, lw_flat, sem_flat
    ], dim=-1).unsqueeze(1)

    all_hidden_gru = all_hidden.permute(1, 0, 2, 3).contiguous().reshape(3, num_sur * FT, d_model).contiguous()

    gru_out_flat, _ = model.sur_decoder_gru(gru_in_flat, all_hidden_gru)
    traj_step_flat = model.sur_output_head(gru_out_flat[:, 0])

    # Bicycle — time-major
    veh_len_flat = cur_veh_len_sur.unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)
    state_global_flat, _ = model._apply_dynamics_single(traj_step_flat, sur_prev_flat, veh_len_flat)

    # Reshape: time-major → (FT, num_sur, 4) → (num_sur, FT, 4)
    batch_traj = state_global_flat.reshape(FT, num_sur, 4).permute(1, 0, 2).contiguous()
    print(f"  batch_traj shape: {batch_traj.shape}")

# ============================================================
# 5. Compare intermediate tensors (time-major indexing: element [t*N+a])
# ============================================================
print("\n=== Intermediate tensor comparison (time-major) ===")

with torch.no_grad():
    # Compare gru_h_last for each timestep
    print("  [gru_h_last]")
    for t in range(min(3, FT)):
        seq_val = seq_h_last_list[t]  # (num_sur, D)
        batch_val = gru_h_last_flat[t * num_sur : (t+1) * num_sur]
        diff = (seq_val - batch_val).abs().max().item()
        print(f"    t={t}: max_diff={diff:.2e}")

    # Compare hist_ctx for each timestep
    print("  [hist_ctx]")
    for t in range(min(3, FT)):
        seq_val = seq_hist_list[t]
        batch_val = hist_ctx_flat[t * num_sur : (t+1) * num_sur]
        diff = (seq_val - batch_val).abs().max().item()
        print(f"    t={t}: max_diff={diff:.2e}")

    # Compare map_ctx for each timestep
    print("  [map_ctx]")
    for t in range(min(3, FT)):
        seq_val = seq_map_ctx_list[t]
        batch_val = map_ctx_flat[t * num_sur : (t+1) * num_sur]
        diff = (seq_val - batch_val).abs().max().item()
        print(f"    t={t}: max_diff={diff:.2e}")

    # Compare GRU output for each timestep
    print("  [gru_out]")
    for t in range(min(3, FT)):
        seq_val = seq_gru_out_list[t]
        batch_val = gru_out_flat[t * num_sur : (t+1) * num_sur, 0]
        diff = (seq_val - batch_val).abs().max().item()
        print(f"    t={t}: max_diff={diff:.2e}")

    # Compare GRU hidden for each timestep
    print("  [hidden state]")
    for t in range(min(3, FT)):
        h_init = gt_hidden_snapshots_raw[t]  # (3, num_sur, D)
        batch_h_t = all_hidden_gru[:, t * num_sur : (t+1) * num_sur, :]
        diff = (h_init - batch_h_t).abs().max().item()
        print(f"    t={t}: max_diff={diff:.2e}")

# ============================================================
# 6. Compare final trajectory per-timestep
# ============================================================
print("\n=== Per-timestep trajectory comparison ===")
all_close = True
for t in range(FT):
    diff = (seq_traj[:, t, :] - batch_traj[:, t, :]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "OK" if max_diff < 1e-5 else "MISMATCH"
    if max_diff >= 1e-5:
        all_close = False
    print(f"  t={t:2d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}  [{status}]")

overall_max = (seq_traj - batch_traj).abs().max().item()
overall_mean = (seq_traj - batch_traj).abs().mean().item()
print(f"\n  Overall: max_diff={overall_max:.2e}, mean_diff={overall_mean:.2e}")

if all_close:
    print("\n  [PASS] Layout fix verified — sequential and batched Step C are numerically identical on CPU!")
else:
    print("\n  [FAIL] Layout mismatch still exists!")
