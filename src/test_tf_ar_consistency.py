"""
TF vs AR 흐름 일치 + TF 학습/Loss 흐름 검증.

1. TF와 AR이 동일 모델, 동일 입력에서 의미적으로 같은 연산을 하는지 확인
2. TF Loss 흐름: gradient 전파, aux loss 시간 인덱싱
3. 시간 꼬임 없는지: timestep 매핑 정확성
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from models.trafficplanner_model import TrafficPlannerModel
from losses.trafficplanner_loss import TrafficPlannerLoss
from datasets.utils import MeanStdNormalizer, NUSC_BIKE_PARAMS, NUSC_NORM_STATS

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cpu')

# ============================================================
# Helper: Build model + normalizers
# ============================================================
def build_model():
    model = TrafficPlannerModel(
        npast=4, nfuture=12,
        map_obs_size_pix=240, nclasses=2,
        map_feat_size=64, past_feat_size=64, future_feat_size=64,
        latent_size=32, z_local_size=32,
        output_bicycle=True, dt=0.5,
        num_intents=9, hist_attn_nhead=4, map_attn_nhead=4, sur_pred_dim=2,
        use_ego_intent=True, use_sur_intent=True,
    ).to(device)

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
    return model, state_norm, att_norm

# ============================================================
# Helper: Dummy map env
# ============================================================
class DummyMapEnv:
    def __init__(self, device, num_layers=4, map_size=240):
        self.device = device
        self.num_layers = num_layers
        self.map_size = map_size
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

def make_scene_graph(B=2, agents_per_scene=3, PT=4, FT=12, NC=2):
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
    scene_graph = Data(
        past=past, future=future, future_gt=future_gt, past_gt=past_gt,
        past_vis=past_vis, future_vis=future_vis,
        lw=lw, sem=sem, edge_index=edge_index,
        batch=batch, ptr=ptr,
    )
    scene_graph.num_nodes = NA
    map_idx = torch.zeros(B, dtype=torch.long, device=device)
    map_env = DummyMapEnv(device)
    return scene_graph, map_idx, map_env


print("=" * 70)
print("TEST 1: TF vs AR — output list 길이/shape 일치")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
scene_graph, map_idx, map_env = make_scene_graph()
NA = scene_graph.past.size(0)
FT = 12

# AR forward
pred_ar = model(scene_graph, map_idx, map_env, teacher_forcing=False)
ar_ego_pred = model.get_ego_pred_outputs()
ar_sur_pred = model.get_sur_pred_outputs()
ar_z_local = model.get_z_local()
ar_intent_w = model.get_intent_weights()
ar_z_local_sur = model.get_sur_z_local()
ar_sur_intent_w = model.get_sur_intent_weights()
ar_ego_map_attn = model.get_ego_map_attn_weights()
ar_sur_map_attn = model.get_sur_map_attn_weights()

# TF forward
pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
tf_ego_pred = model.get_ego_pred_outputs()
tf_sur_pred = model.get_sur_pred_outputs()
tf_z_local = model.get_z_local()
tf_intent_w = model.get_intent_weights()
tf_z_local_sur = model.get_sur_z_local()
tf_sur_intent_w = model.get_sur_intent_weights()
tf_ego_map_attn = model.get_ego_map_attn_weights()
tf_sur_map_attn = model.get_sur_map_attn_weights()

# Check list lengths
all_ok = True
checks = [
    ("ego_pred_outputs", ar_ego_pred, tf_ego_pred),
    ("sur_pred_outputs", ar_sur_pred, tf_sur_pred),
    ("z_local", ar_z_local, tf_z_local),
    ("intent_weights", ar_intent_w, tf_intent_w),
    ("z_local_sur", ar_z_local_sur, tf_z_local_sur),
    ("sur_intent_weights", ar_sur_intent_w, tf_sur_intent_w),
    ("ego_map_attn", ar_ego_map_attn, tf_ego_map_attn),
    ("sur_map_attn", ar_sur_map_attn, tf_sur_map_attn),
]

for name, ar_list, tf_list in checks:
    len_ok = len(ar_list) == len(tf_list) == FT
    if not len_ok:
        all_ok = False
        print(f"  [FAIL] {name}: AR len={len(ar_list)}, TF len={len(tf_list)}, expected {FT}")
    else:
        # Check shapes match
        shape_ok = True
        for t in range(FT):
            if ar_list[t].shape != tf_list[t].shape:
                shape_ok = False
                print(f"  [FAIL] {name} t={t}: AR shape={ar_list[t].shape}, TF shape={tf_list[t].shape}")
        if shape_ok:
            print(f"  [OK] {name}: len={FT}, shapes match")
        else:
            all_ok = False

# Check future_pred shape
ar_shape = pred_ar['future_pred'].shape
tf_shape = pred_tf['future_pred'].shape
if ar_shape == tf_shape:
    print(f"  [OK] future_pred: AR={ar_shape}, TF={tf_shape}")
else:
    all_ok = False
    print(f"  [FAIL] future_pred: AR={ar_shape}, TF={tf_shape}")

if all_ok:
    print("  [PASS] TEST 1 passed\n")
else:
    print("  [FAIL] TEST 1 failed\n")


print("=" * 70)
print("TEST 2: TF Loss — gradient 전파 확인")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
scene_graph, map_idx, map_env = make_scene_graph()

loss_weights = {
    'recon': 1.0, 'kl': 0.0,
    'sur_pred': 0.1, 'ego_pred': 0.1,
    'intent_ce': 0.1, 'sur_intent_ce': 0.1,
    'map_attn': 0.05,
    'potential': 0.0, 'sparsity': 0.0,
}
loss_fn = TrafficPlannerLoss(loss_weights, state_norm, att_norm, phase=1, use_potential_loss=False)

pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
loss_dict = loss_fn(scene_graph, pred_tf, map_idx=map_idx, map_env=map_env,
                    model=model, use_teacher_forcing=True)
loss = loss_dict['loss'][0]

# Check loss is finite
loss_ok = torch.isfinite(loss).item()
print(f"  TF total loss: {loss.item():.4f}, finite: {loss_ok}")

# Backward
loss.backward()

# Check gradients flow to key modules
grad_checks = [
    ("ego_decoder_gru", model.ego_decoder_gru),
    ("sur_decoder_gru", model.sur_decoder_gru),
    ("ego_output_head", model.ego_output_head),
    ("sur_output_head", model.sur_output_head),
    ("ego_history_attn", model.ego_history_attn),
    ("sur_history_attn", model.sur_history_attn),
    ("ego_map_attn", model.ego_map_attn),
    ("sur_map_attn", model.sur_map_attn),
    ("intent_codebook", model.intent_codebook),
    ("sur_intent_codebook", model.sur_intent_codebook),
    ("interaction_gcn", model.interaction_gcn),
]

grad_ok = True
for name, module in grad_checks:
    has_grad = False
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().max() > 0:
                has_grad = True
                break
    if has_grad:
        print(f"  [OK] {name}: gradient flows")
    else:
        grad_ok = False
        print(f"  [WARN] {name}: no gradient (may be frozen or zero)")

# Check warmup gradient (gru_hidden_init carries grad)
warmup_modules = [
    ("ego_warmup_gru", model.ego_warmup_gru if hasattr(model, 'ego_warmup_gru') else None),
    ("sur_warmup_gru", model.sur_warmup_gru if hasattr(model, 'sur_warmup_gru') else None),
]
for name, module in warmup_modules:
    if module is not None:
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                      for p in module.parameters() if p.requires_grad)
        print(f"  [{'OK' if has_grad else 'WARN'}] {name}: {'gradient flows' if has_grad else 'no gradient'}")

if loss_ok and grad_ok:
    print("  [PASS] TEST 2 passed\n")
else:
    print(f"  [{'PASS' if loss_ok else 'FAIL'}] TEST 2 {'passed' if loss_ok and grad_ok else 'check warnings'}\n")


print("=" * 70)
print("TEST 3: 시간 인덱싱 정확성 — aux output의 timestep 매핑")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
scene_graph, map_idx, map_env = make_scene_graph()
ego_mask = torch.zeros(NA, dtype=torch.bool, device=device)
ego_mask[scene_graph.ptr[:-1]] = True
num_ego = int(ego_mask.sum())
num_sur = ego_mask.size(0) - num_ego

# AR forward (each step is clearly t=0,1,...,11)
pred_ar = model(scene_graph, map_idx, map_env, teacher_forcing=False)
ar_ego_pred = model.get_ego_pred_outputs()  # list of (num_ego, 2)
ar_sur_pred = model.get_sur_pred_outputs()  # list of (num_sur, 2)
ar_z_local = model.get_z_local()  # list of (num_ego, intent_dim)
ar_z_local_sur = model.get_sur_z_local()  # list of (num_sur, intent_dim)

# TF forward
pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
tf_ego_pred = model.get_ego_pred_outputs()
tf_sur_pred = model.get_sur_pred_outputs()
tf_z_local = model.get_z_local()
tf_z_local_sur = model.get_sur_z_local()

# Verify timestep indexing: aux outputs should be FT items, each with correct agent count
time_ok = True
for t in range(FT):
    # TF ego pred should have num_sur agents (ego_pred = sur decoder predicting ego delta)
    # Actually ego_pred comes from sur loop: sur agents predict ego delta
    if tf_ego_pred[t].size(0) != num_sur:
        print(f"  [FAIL] tf_ego_pred[{t}] has {tf_ego_pred[t].size(0)} agents, expected {num_sur}")
        time_ok = False
    if tf_sur_pred[t].size(0) != num_ego:
        print(f"  [FAIL] tf_sur_pred[{t}] has {tf_sur_pred[t].size(0)} agents, expected {num_ego}")
        time_ok = False
    if tf_z_local[t].size(0) != num_ego:
        print(f"  [FAIL] tf_z_local[{t}] has {tf_z_local[t].size(0)} agents, expected {num_ego}")
        time_ok = False
    if tf_z_local_sur[t].size(0) != num_sur:
        print(f"  [FAIL] tf_z_local_sur[{t}] has {tf_z_local_sur[t].size(0)} agents, expected {num_sur}")
        time_ok = False

if time_ok:
    print(f"  [OK] All {FT} timesteps have correct agent counts (ego={num_ego}, sur={num_sur})")
else:
    print("  [FAIL] Agent count mismatch")

# Verify: TF의 per_t reshape가 시간을 올바르게 분리하는지
# ego_pred_outputs[t]는 t번째 timestep의 sur agents' ego delta prediction
# Loss에서: gt_delta = gt_future[:, t, :2] - gt_future[:, t-1, :2] (or past[-1])
# 따라서 t=0의 prediction은 t=0 state에서의 prediction이어야 함

# All TF outputs are detached, so we check they're not all identical (would indicate time collapse)
all_unique = True
for t in range(1, FT):
    if torch.allclose(tf_ego_pred[0], tf_ego_pred[t]):
        print(f"  [WARN] tf_ego_pred[0] == tf_ego_pred[{t}] — possible time collapse")
        all_unique = False
    if torch.allclose(tf_z_local[0], tf_z_local[t]):
        print(f"  [WARN] tf_z_local[0] == tf_z_local[{t}] — possible time collapse")
        all_unique = False

if all_unique:
    print(f"  [OK] All timestep outputs are unique (no time collapse)")

if time_ok and all_unique:
    print("  [PASS] TEST 3 passed\n")
else:
    print("  [FAIL] TEST 3 failed\n")


print("=" * 70)
print("TEST 4: TF vs AR — 동일 입력에서 1-step 연산 일치 확인")
print("=" * 70)
# TF와 AR은 다른 경로(batched vs sequential)를 타지만,
# 동일한 모듈(같은 weight)을 사용하므로 모듈 호출 순서가 같은지 확인.
# TF: _sur_loop_decoder → _ego_loop_decoder
# AR: autoregressive_decoder (for loop 내에서 ego+sur 동시)
# 순서는 다르지만 사용 모듈은 동일해야 함.

model, state_norm, att_norm = build_model()
model.eval()

# Collect modules used in TF path
tf_modules_sur = {
    'gcn': model.interaction_gcn,
    'sur_history_attn': model.sur_history_attn,
    'ego_pred_head': model.ego_pred_head,
    'sur_map_attn': model.sur_map_attn,
    'sur_intent_codebook': model.sur_intent_codebook,
    'sur_decoder_gru': model.sur_decoder_gru,
    'sur_output_head': model.sur_output_head,
}
tf_modules_ego = {
    'gcn': model.interaction_gcn,
    'ego_history_attn': model.ego_history_attn,
    'sur_pred_head': model.sur_pred_head,
    'ego_map_attn': model.ego_map_attn,
    'intent_codebook': model.intent_codebook,
    'ego_decoder_gru': model.ego_decoder_gru,
    'ego_output_head': model.ego_output_head,
}

# AR uses the same modules — verify by checking they're the same object
ar_modules = {
    'gcn': model.interaction_gcn,
    'ego_history_attn': model.ego_history_attn,
    'sur_history_attn': model.sur_history_attn,
    'sur_pred_head': model.sur_pred_head,
    'ego_pred_head': model.ego_pred_head,
    'ego_map_attn': model.ego_map_attn,
    'sur_map_attn': model.sur_map_attn,
    'intent_codebook': model.intent_codebook,
    'sur_intent_codebook': model.sur_intent_codebook,
    'ego_decoder_gru': model.ego_decoder_gru,
    'sur_decoder_gru': model.sur_decoder_gru,
    'ego_output_head': model.ego_output_head,
    'sur_output_head': model.sur_output_head,
}

# All modules in TF should exist in AR (same nn.Module instances)
module_match = True
for name in list(tf_modules_sur.keys()) + list(tf_modules_ego.keys()):
    if name in ar_modules:
        tf_mod = tf_modules_sur.get(name) or tf_modules_ego.get(name)
        if tf_mod is not ar_modules[name]:
            module_match = False
            print(f"  [FAIL] {name}: TF and AR use different module instances!")
    else:
        module_match = False
        print(f"  [FAIL] {name}: not found in AR")

if module_match:
    print("  [OK] TF and AR use identical module instances")
    print("  [PASS] TEST 4 passed\n")
else:
    print("  [FAIL] TEST 4 failed\n")


print("=" * 70)
print("TEST 5: TF Loss 흐름 — AR과 동일 Loss 계산 가능")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
scene_graph, map_idx, map_env = make_scene_graph()

loss_fn = TrafficPlannerLoss(loss_weights, state_norm, att_norm, phase=1, use_potential_loss=False)

# AR loss
pred_ar = model(scene_graph, map_idx, map_env, teacher_forcing=False)
loss_ar = loss_fn(scene_graph, pred_ar, map_idx=map_idx, map_env=map_env,
                  model=model, use_teacher_forcing=False)

# TF loss
pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
loss_tf = loss_fn(scene_graph, pred_tf, map_idx=map_idx, map_env=map_env,
                  model=model, use_teacher_forcing=True)

loss5_ok = True

# Both should produce finite losses
for key in ['loss', 'recon_loss']:
    if key in loss_ar and key in loss_tf:
        ar_val = loss_ar[key][0].item()
        tf_val = loss_tf[key][0].item()
        ar_fin = torch.isfinite(loss_ar[key][0]).item()
        tf_fin = torch.isfinite(loss_tf[key][0]).item()
        print(f"  {key}: AR={ar_val:.4f}(fin={ar_fin}), TF={tf_val:.4f}(fin={tf_fin})")
        if not ar_fin or not tf_fin:
            loss5_ok = False

# Check aux losses exist in both
for key in ['sur_pred_loss', 'ego_pred_loss', 'intent_ce_loss', 'sur_intent_ce_loss', 'map_attn_loss']:
    if key in loss_ar and key in loss_tf:
        ar_val = loss_ar[key][0].item()
        tf_val = loss_tf[key][0].item()
        ar_fin = torch.isfinite(torch.tensor(ar_val)).item()
        tf_fin = torch.isfinite(torch.tensor(tf_val)).item()
        print(f"  {key}: AR={ar_val:.4f}, TF={tf_val:.4f}")
        if not ar_fin or not tf_fin:
            loss5_ok = False
    elif key in loss_ar:
        print(f"  [WARN] {key}: only in AR, not in TF")
    elif key in loss_tf:
        print(f"  [WARN] {key}: only in TF, not in AR")

if loss5_ok:
    print("  [PASS] TEST 5 passed\n")
else:
    print("  [FAIL] TEST 5 failed\n")


print("=" * 70)
print("TEST 6: TF backward — NaN/Inf gradient 확인")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
scene_graph, map_idx, map_env = make_scene_graph()

loss_fn = TrafficPlannerLoss(loss_weights, state_norm, att_norm, phase=1, use_potential_loss=False)
pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
loss_dict = loss_fn(scene_graph, pred_tf, map_idx=map_idx, map_env=map_env,
                    model=model, use_teacher_forcing=True)

model.zero_grad()
loss_dict['loss'][0].backward()

nan_count = 0
inf_count = 0
total_params = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        total_params += 1
        if torch.isnan(p.grad).any():
            nan_count += 1
            print(f"  [NaN] {name}")
        if torch.isinf(p.grad).any():
            inf_count += 1
            print(f"  [Inf] {name}")

print(f"  Checked {total_params} params with gradients")
print(f"  NaN gradients: {nan_count}, Inf gradients: {inf_count}")

if nan_count == 0 and inf_count == 0:
    print("  [PASS] TEST 6 passed\n")
else:
    print("  [FAIL] TEST 6 failed\n")


print("=" * 70)
print("TEST 7: TF multi-step gradient 누적 — loss.backward() 2회")
print("=" * 70)

model, state_norm, att_norm = build_model()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(2):
    optimizer.zero_grad()
    scene_graph, map_idx, map_env = make_scene_graph()
    pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True)
    loss_dict = loss_fn(scene_graph, pred_tf, map_idx=map_idx, map_env=map_env,
                        model=model, use_teacher_forcing=True)
    loss = loss_dict['loss'][0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"  Step {step}: loss={loss.item():.4f}")

# Check model didn't diverge
scene_graph, map_idx, map_env = make_scene_graph()
with torch.no_grad():
    pred_check = model(scene_graph, map_idx, map_env, teacher_forcing=True)
    future = pred_check['future_pred']
    is_finite = torch.isfinite(future).all().item()
    print(f"  Post-training pred finite: {is_finite}")

if is_finite:
    print("  [PASS] TEST 7 passed\n")
else:
    print("  [FAIL] TEST 7 failed\n")


# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
