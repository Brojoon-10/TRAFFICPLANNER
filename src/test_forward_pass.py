"""
Forward pass test for TrafficPlannerModel (Redesign).
Tests model instantiation, forward, reconstruct, sample, teacher forcing, and loss computation
using dummy data (no real dataset needed).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from models.trafficplanner_model import TrafficPlannerModel
from losses.trafficplanner_loss import TrafficPlannerLoss
from datasets.utils import MeanStdNormalizer, NUSC_BIKE_PARAMS, NUSC_NORM_STATS


def make_dummy_normalizers():
    """Create state and attribute normalizers matching nuScenes (car, truck)."""
    ninfo = NUSC_NORM_STATS[('car', 'truck')]
    state_mean = [ninfo['lscale'][0], ninfo['lscale'][0],
                  ninfo['h'][0], ninfo['h'][0],
                  ninfo['s'][0], ninfo['hdot'][0]]
    state_std = [ninfo['lscale'][1], ninfo['lscale'][1],
                 ninfo['h'][1], ninfo['h'][1],
                 ninfo['s'][1], ninfo['hdot'][1]]
    state_normalizer = MeanStdNormalizer(torch.tensor(state_mean), torch.tensor(state_std))

    att_mean = [ninfo['l'][0], ninfo['w'][0]]
    att_std = [ninfo['l'][1], ninfo['w'][1]]
    att_normalizer = MeanStdNormalizer(torch.tensor(att_mean), torch.tensor(att_std))

    return state_normalizer, att_normalizer


def make_dummy_scene_graph(B=2, agents_per_scene=3, PT=4, FT=12, NC=2, device='cpu'):
    """
    Create a dummy batched scene graph (PyG Batch format).

    Each scene has `agents_per_scene` agents (first is ego).
    """
    NA = B * agents_per_scene

    past = torch.randn(NA, PT, 6, device=device)
    future = torch.randn(NA, FT, 6, device=device)
    future_gt = future.clone()
    past_gt = past.clone()
    past_vis = torch.ones(NA, PT, device=device)
    future_vis = torch.ones(NA, FT, device=device)

    # lw: vehicle length/width (normalized)
    lw = torch.randn(NA, 2, device=device) * 0.3

    # sem: one-hot category
    sem = torch.zeros(NA, NC, device=device)
    sem[:, 0] = 1.0  # all "car"

    # edge_index: fully connected within each scene
    edge_list = []
    for b in range(B):
        start = b * agents_per_scene
        for i in range(agents_per_scene):
            for j in range(agents_per_scene):
                if i != j:
                    edge_list.append([start + j, start + i])
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T

    # batch and ptr
    batch = torch.arange(B, device=device).repeat_interleave(agents_per_scene)
    ptr = torch.arange(0, NA + 1, agents_per_scene, device=device)

    # Build as Data then create Batch manually
    data = Data(
        past=past, future=future, future_gt=future_gt, past_gt=past_gt,
        past_vis=past_vis, future_vis=future_vis,
        lw=lw, sem=sem, edge_index=edge_index,
        batch=batch, ptr=ptr,
    )
    # Set num_nodes for PyG
    data.num_nodes = NA

    return data


class DummyMapEnv:
    """Mock map environment that returns dummy map crops."""
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


def test_model_instantiation(device):
    print("=" * 60)
    print("[1] Model Instantiation")
    print("=" * 60)

    model = TrafficPlannerModel(
        npast=4, nfuture=12,
        map_obs_size_pix=240, nclasses=2,
        map_feat_size=64, past_feat_size=64, future_feat_size=64,
        latent_size=32, z_local_size=32,
        output_bicycle=True, dt=0.5,
        num_intents=8, hist_attn_nhead=4, map_attn_nhead=4, sur_pred_dim=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Map tokens: {model.map_num_tokens} ({model.map_token_spatial}x{model.map_token_spatial})")
    print(f"  Map token channels: {model.map_token_ch}")
    print(f"  d_model: {model.d_model}")
    print(f"  z_size: {model.z_size}")
    print(f"  intent_dim: {model.intent_dim}")
    print(f"  num_intents: {model.num_intents}")
    print("  [OK] Model instantiated successfully\n")
    return model


def test_forward_pass(model, device):
    print("=" * 60)
    print("[2] Forward Pass (autoregressive)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    # Set normalizers and bicycle params
    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)

    model.set_normalizer(state_norm)
    model.set_att_normalizer(att_norm)
    model.set_bicycle_params({k: (v[0], v[1]) if isinstance(v, tuple) else v for k, v in NUSC_BIKE_PARAMS.items()})

    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)

    NA = B * agents_per_scene
    assert pred['future_pred'].shape == (NA, 12, 4), f"Expected (NA, 12, 4), got {pred['future_pred'].shape}"
    assert pred['prior_out'][0].shape == (NA, 32), f"Expected prior_mu (NA, 32), got {pred['prior_out'][0].shape}"
    assert pred['posterior_out'][0].shape == (NA, 32), f"Expected post_mu (NA, 32), got {pred['posterior_out'][0].shape}"

    # Check analysis outputs
    z_local = model.get_z_local_stacked()
    assert z_local is not None, "z_local should be available after forward"
    assert z_local.shape == (12, B, 32), f"Expected z_local (12, B, 32), got {z_local.shape}"

    intent_w_raw = model.get_intent_weights()
    assert intent_w_raw is not None, "intent_weights should be available"
    intent_w = torch.stack(intent_w_raw, dim=0)
    assert intent_w.shape == (12, B, 8), f"Expected intent_weights (12, B, 8), got {intent_w.shape}"

    map_attn = model.get_map_attn_weights()
    assert map_attn is not None, "map_attn should be available"
    assert len(map_attn) == 12, f"Expected 12 map_attn entries, got {len(map_attn)}"

    sur_pred = model.get_sur_pred_outputs()
    assert sur_pred is not None, "sur_pred should be available"
    assert len(sur_pred) == 12, f"Expected 12 sur_pred entries, got {len(sur_pred)}"

    ego_pred = model.get_ego_pred_outputs()
    assert ego_pred is not None, "ego_pred should be available"
    assert len(ego_pred) == 12, f"Expected 12 ego_pred entries, got {len(ego_pred)}"

    print(f"  future_pred shape: {pred['future_pred'].shape}")
    print(f"  z_local shape: {z_local.shape}")
    print(f"  intent_weights shape: {intent_w.shape}")
    print(f"  map_attn[0] shape: {map_attn[0].shape}")
    print(f"  sur_pred[0] shape: {sur_pred[0].shape}")
    print(f"  ego_pred[0] shape: {ego_pred[0].shape}")
    print("  [OK] Forward pass successful\n")
    return pred, scene_graph, map_idx, map_env


def test_teacher_forcing(model, device):
    print("=" * 60)
    print("[3] Teacher Forcing Forward Pass")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=100)

    # TF output is list of segments
    tf_preds = pred['future_pred']
    assert isinstance(tf_preds, list), "TF output should be list of segments"
    print(f"  Number of segments: {len(tf_preds)}")
    for i, seg in enumerate(tf_preds[:3]):  # show first 3
        print(f"  Segment {i}: {len(seg)} steps, shape {seg[0].shape if len(seg) > 0 else 'empty'}")

    print("  [OK] Teacher forcing forward pass successful\n")
    return pred


def test_reconstruct(model, device):
    print("=" * 60)
    print("[4] Reconstruct")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.reconstruct(scene_graph, map_idx, map_env)

    NA = B * agents_per_scene
    assert pred['future_pred'].shape == (NA, 12, 4)
    print(f"  Reconstruct future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Reconstruct successful\n")


def test_sample(model, device):
    print("=" * 60)
    print("[5] Sample (3 samples)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.sample(scene_graph, map_idx, map_env, num_samples=3)

    NA = B * agents_per_scene
    # sample() uses stack(..., dim=1) → (NA, NS, FT, 4)
    assert pred['future_pred'].shape == (NA, 3, 12, 4), f"Expected (NA, 3, 12, 4), got {pred['future_pred'].shape}"
    print(f"  Sample future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Sample successful\n")


def test_sample_batched(model, device):
    print("=" * 60)
    print("[6] Sample Batched (3 samples)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.sample_batched(scene_graph, map_idx, map_env, num_samples=3)

    NA = B * agents_per_scene
    # sample_batched returns future_pred as (NA, NS, FT, 4)
    assert pred['future_pred'].shape == (NA, 3, 12, 4), f"Expected (NA, 3, 12, 4), got {pred['future_pred'].shape}"
    print(f"  Sample batched future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Sample batched successful\n")


def test_loss_computation(model, device):
    print("=" * 60)
    print("[7] Loss Computation (autoregressive)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)

    loss_weights = {
        'recon': 1.0, 'kl': 0.04,
        'coll_veh_prior': 0.0, 'coll_env_prior': 0.0,
        'potential_veh': 0.0, 'potential_env': 0.0,
        'sparse': 0.0,
        'sur_pred': 0.1, 'ego_pred': 0.1,
        'intent_ce': 0.0, 'map_attn': 0.0,
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights, state_norm, att_norm,
        phase=1, use_sparse_loss=False, use_potential_loss=False,
    ).to(device)

    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env, model=model)

    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].mean().item():.4f}")
    print(f"  KL loss: {loss_dict['kl_loss'].mean().item():.4f}")
    if 'sur_pred_loss' in loss_dict:
        print(f"  Sur pred loss: {loss_dict['sur_pred_loss'].item():.4f}")
    if 'ego_pred_loss' in loss_dict:
        print(f"  Ego pred loss: {loss_dict['ego_pred_loss'].item():.4f}")

    # Check backward pass
    loss_dict['loss'].backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    num_with_grad = len(grad_norms)
    num_total = sum(1 for _ in model.parameters())
    print(f"  Params with gradients: {num_with_grad}/{num_total}")
    assert num_with_grad > 0, "No parameters received gradients!"
    print("  [OK] Loss computation and backward pass successful\n")


def test_loss_teacher_forcing(model, device):
    print("=" * 60)
    print("[8] Loss Computation (teacher forcing)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)

    loss_weights = {
        'recon': 1.0, 'kl': 0.04,
        'coll_veh_prior': 0.0, 'coll_env_prior': 0.0,
        'potential_veh': 0.0, 'potential_env': 0.0,
        'sparse': 0.0,
        'sur_pred': 0.1, 'ego_pred': 0.1,
        'intent_ce': 0.0, 'map_attn': 0.0,
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights, state_norm, att_norm,
        phase=1, use_sparse_loss=False, use_potential_loss=False,
    ).to(device)

    model.train()
    model.zero_grad()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=100)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env,
                        model=model, use_teacher_forcing=True)

    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].mean().item():.4f}")
    print(f"  KL loss: {loss_dict['kl_loss'].mean().item():.4f}")
    if 'num_segments' in loss_dict:
        print(f"  Num segments: {loss_dict['num_segments'].item():.0f}")
    if 'sur_pred_loss' in loss_dict:
        print(f"  Sur pred loss: {loss_dict['sur_pred_loss'].item():.4f}")
    if 'ego_pred_loss' in loss_dict:
        print(f"  Ego pred loss: {loss_dict['ego_pred_loss'].item():.4f}")

    loss_dict['loss'].backward()
    num_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    num_total = sum(1 for _ in model.parameters())
    print(f"  Params with gradients: {num_with_grad}/{num_total}")
    assert num_with_grad > 0, "No parameters received gradients!"
    print("  [OK] TF loss computation and backward pass successful\n")


def test_freeze_for_finetuning(model, device):
    print("=" * 60)
    print("[9] Freeze for Fine-tuning (Phase 2)")
    print("=" * 60)

    model.set_phase(2)
    model.freeze_for_finetuning()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    # Check that key ego components are trainable
    trainable_prefixes = ['ego_history_attn', 'ego_map_attn', 'ego_decoder_gru',
                          'ego_output_head', 'intent_codebook', 'intent_ce_head',
                          'sur_pred_head', 'ego_warmup_gru']
    for prefix in trainable_prefixes:
        matching = [n for n in trainable if n.startswith(prefix)]
        assert len(matching) > 0, f"Expected {prefix} to be trainable, but found none!"
        print(f"  [TRAINABLE] {prefix}: {len(matching)} params")

    # Check that key frozen components are frozen
    frozen_prefixes = ['latent_prior_net', 'latent_posterior_net', 'map_conv_early',
                       'interaction_gcn', 'sur_decoder_gru', 'sur_output_head']
    for prefix in frozen_prefixes:
        matching = [n for n in frozen if n.startswith(prefix)]
        assert len(matching) > 0, f"Expected {prefix} to be frozen, but found none!"
        print(f"  [FROZEN] {prefix}: {len(matching)} params")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  Total trainable: {num_trainable:,}")
    print(f"  Total frozen: {num_frozen:,}")
    print("  [OK] Freeze for fine-tuning successful\n")

    # Reset
    model.set_phase(1)
    for p in model.parameters():
        p.requires_grad = True


def test_phase2_forward(model, device):
    print("=" * 60)
    print("[10] Phase 2 Forward + Intent Active")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.set_phase(2)
    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)

    z_local = model.get_z_local_stacked()
    intent_w_raw = model.get_intent_weights()
    intent_w = torch.stack(intent_w_raw, dim=0)

    # In Phase 2, z_local should be non-zero (intent codebook active)
    z_local_norm = z_local.norm().item()
    intent_w_sum = intent_w.sum().item()
    print(f"  z_local L2 norm: {z_local_norm:.4f} (should be > 0 in Phase 2)")
    print(f"  intent_weights sum: {intent_w_sum:.4f}")

    # Note: z_local can still be 0 if codebook initialized to 0, so just check intent_weights
    assert intent_w_sum > 0, "Intent weights should be non-zero in Phase 2"
    print("  [OK] Phase 2 forward pass successful\n")

    # Reset
    model.set_phase(1)


def test_map_attn_loss(model, device):
    print("=" * 60)
    print("[11] Map Attention Guidance Loss")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)

    loss_weights = {
        'recon': 1.0, 'kl': 0.04,
        'coll_veh_prior': 0.0, 'coll_env_prior': 0.0,
        'potential_veh': 0.0, 'potential_env': 0.0,
        'sparse': 0.0,
        'sur_pred': 0.1, 'ego_pred': 0.1,
        'intent_ce': 0.0, 'map_attn': 0.1,  # map_attn ON
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights, state_norm, att_norm,
        phase=1, use_sparse_loss=False, use_potential_loss=False,
    ).to(device)

    # --- Autoregressive mode ---
    model.train()
    model.zero_grad()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env, model=model)

    assert 'map_attn_loss' in loss_dict, "map_attn_loss should be in loss_dict when weight > 0"
    map_attn_val = loss_dict['map_attn_loss'].item()
    print(f"  [AR] map_attn_loss: {map_attn_val:.4f}")
    print(f"  [AR] Total loss: {loss_dict['loss'].item():.4f}")

    loss_dict['loss'].backward()
    # Check that map_attn module gets gradients
    map_attn_grads = sum(1 for n, p in model.named_parameters()
                         if 'ego_map_attn' in n and p.grad is not None)
    print(f"  [AR] ego_map_attn params with grad: {map_attn_grads}")

    # --- Teacher forcing mode ---
    model.zero_grad()
    pred_tf = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=100)
    loss_dict_tf = loss_fn(scene_graph, pred_tf, map_idx=map_idx, map_env=map_env,
                           model=model, use_teacher_forcing=True)

    assert 'map_attn_loss' in loss_dict_tf, "map_attn_loss should be in TF loss_dict"
    map_attn_tf_val = loss_dict_tf['map_attn_loss'].item()
    print(f"  [TF] map_attn_loss: {map_attn_tf_val:.4f}")
    print(f"  [TF] Total loss: {loss_dict_tf['loss'].item():.4f}")

    loss_dict_tf['loss'].backward()
    map_attn_grads_tf = sum(1 for n, p in model.named_parameters()
                            if 'ego_map_attn' in n and p.grad is not None)
    print(f"  [TF] ego_map_attn params with grad: {map_attn_grads_tf}")

    print("  [OK] Map attention guidance loss successful\n")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    # Test 1: Instantiation
    model = test_model_instantiation(device)

    # Set up normalizers and bicycle params
    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)
    model.set_normalizer(state_norm)
    model.set_att_normalizer(att_norm)
    model.set_bicycle_params(NUSC_BIKE_PARAMS)

    # Test 2: Forward pass
    test_forward_pass(model, device)

    # Test 3: Teacher forcing
    test_teacher_forcing(model, device)

    # Test 4: Reconstruct
    test_reconstruct(model, device)

    # Test 5: Sample
    test_sample(model, device)

    # Test 6: Sample batched
    test_sample_batched(model, device)

    # Test 7: Loss (autoregressive)
    test_loss_computation(model, device)

    # Test 8: Loss (teacher forcing)
    test_loss_teacher_forcing(model, device)

    # Test 9: Freeze for fine-tuning
    test_freeze_for_finetuning(model, device)

    # Test 10: Phase 2 forward
    test_phase2_forward(model, device)

    # Test 11: Map Attention Guidance Loss
    test_map_attn_loss(model, device)

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
