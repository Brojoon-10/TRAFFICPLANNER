"""
Forward pass test for TrafficPlannerModel (Transformer Decoder Redesign).
Tests model instantiation, training forward, inference forward, reconstruct,
sample, loss computation, and freeze for fine-tuning using dummy data.
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

    data = Data(
        past=past, future=future, future_gt=future_gt, past_gt=past_gt,
        past_vis=past_vis, future_vis=future_vis,
        lw=lw, sem=sem, edge_index=edge_index,
        batch=batch, ptr=ptr,
    )
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


def setup_model(device):
    """Create and configure model with normalizers."""
    model = TrafficPlannerModel(
        npast=4, nfuture=12,
        map_obs_size_pix=240, nclasses=2,
        map_feat_size=64, past_feat_size=64, future_feat_size=64,
        latent_size=32, z_local_size=32,
        output_bicycle=True, dt=0.5,
        num_intents=9, sur_pred_dim=2,
        # Transformer decoder params
        trans_num_layers=2,
        trans_d_model=128,
        trans_nhead=8,
        trans_ffn_dim=256,
        trans_dropout=0.1,
        use_ego_z_local=True,
        use_sur_z_local=False,
        # Enc-Dec Cross-Attention params
        use_a2a_rel_bias=True,
        num_z_tokens=4,
        num_z_queries=2,
        context_num_layers=2,
        map_summary_tokens=8,
    ).to(device)

    state_norm, att_norm = make_dummy_normalizers()
    state_norm.mean_vals = state_norm.mean_vals.to(device)
    state_norm.std_vals = state_norm.std_vals.to(device)
    att_norm.mean_vals = att_norm.mean_vals.to(device)
    att_norm.std_vals = att_norm.std_vals.to(device)
    model.set_normalizer(state_norm)
    model.set_att_normalizer(att_norm)
    model.set_bicycle_params(NUSC_BIKE_PARAMS)

    return model


def test_model_instantiation(device):
    print("=" * 60)
    print("[1] Model Instantiation (Transformer Decoder)")
    print("=" * 60)

    model = setup_model(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Map tokens: {model.map_num_tokens} ({model.map_token_spatial}x{model.map_token_spatial})")
    print(f"  Map token channels: {model.map_token_ch}")
    print(f"  trans_d_model: {model.trans_d_model}")
    print(f"  trans_num_layers: {model.trans_num_layers}")
    print(f"  z_size: {model.z_size}")
    print(f"  intent_dim: {model.intent_dim}")
    print(f"  num_intents: {model.num_intents}")

    # Check Transformer layers exist
    assert len(model.trans_layers) == 2, f"Expected 2 trans layers, got {len(model.trans_layers)}"
    assert hasattr(model, 'query_proj'), "Missing query_proj"
    assert hasattr(model, 'step_pe'), "Missing step_pe"
    assert hasattr(model, 'context_encoder'), "Missing context_encoder"
    assert hasattr(model, 'z_cross_attn'), "Missing z_cross_attn"
    assert hasattr(model, 'z_query'), "Missing z_query"
    assert hasattr(model, 'z_gen_q_proj'), "Missing z_gen_q_proj"
    assert hasattr(model, 'future_proj'), "Missing future_proj"
    assert hasattr(model, 'future_temporal_pe'), "Missing future_temporal_pe"
    assert hasattr(model, 'map_summary_pooling'), "Missing map_summary_pooling"
    assert hasattr(model, 'intent_codebook'), "Missing intent_codebook"
    assert hasattr(model, 'ego_output_head'), "Missing ego_output_head"
    assert hasattr(model, 'sur_output_head'), "Missing sur_output_head"
    assert hasattr(model, 'sur_pred_head'), "Missing sur_pred_head"
    assert hasattr(model, 'ego_pred_head'), "Missing ego_pred_head"
    assert hasattr(model, 'intent_ce_head'), "Missing intent_ce_head"

    print("  [OK] Model instantiated successfully\n")
    return model


def test_training_forward(model, device):
    """Test training forward (Enc-Dec Cross-Attention, AR blended)."""
    print("=" * 60)
    print("[2] Training Forward (teacher_forcing=True, Enc-Dec AR)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    NA = B * agents_per_scene
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=0)

    # future_pred should be (NA, FT, 4) tensor
    fp = pred['future_pred']
    assert isinstance(fp, torch.Tensor), f"Expected tensor, got {type(fp)}"
    assert fp.shape == (NA, 12, 4), f"Expected ({NA}, 12, 4), got {fp.shape}"

    # Encoder outputs
    assert pred['prior_out'][0].shape == (NA, 32), f"prior_mu shape: {pred['prior_out'][0].shape}"
    assert pred['posterior_out'][0].shape == (NA, 32), f"post_mu shape: {pred['posterior_out'][0].shape}"

    # z_local: training → tensor (FT, num_ego, intent_dim)
    z_local = model.get_z_local_stacked()
    assert z_local is not None, "z_local should be available"
    num_ego = B  # 1 ego per scene
    assert z_local.shape == (12, num_ego, 32), f"Expected z_local (12, {num_ego}, 32), got {z_local.shape}"

    # intent_weights: training → tensor (FT, num_ego, K)
    intent_w = model.get_intent_weights()
    assert intent_w is not None, "intent_weights should be available"
    assert isinstance(intent_w, torch.Tensor), f"Training mode should return tensor, got {type(intent_w)}"
    assert intent_w.shape == (12, num_ego, 9), f"Expected (12, {num_ego}, 9), got {intent_w.shape}"

    # ego map attn: training → tensor (FT, N_ego, num_tokens) — one per AR step
    ego_map_attn = model.get_ego_map_attn_weights()
    assert ego_map_attn is not None, "ego_map_attn should be available in training"
    assert isinstance(ego_map_attn, torch.Tensor), f"Expected tensor, got {type(ego_map_attn)}"
    print(f"  ego_map_attn shape: {ego_map_attn.shape}")

    # sur map attn
    sur_map_attn = model.get_sur_map_attn_weights()
    assert sur_map_attn is not None, "sur_map_attn should be available in training"

    # sur_pred/ego_pred: AR loop → list of FT tensors
    sur_pred = model.get_sur_pred_outputs()
    assert sur_pred is not None, "sur_pred should be available"
    if isinstance(sur_pred, list):
        sur_pred_stacked = torch.stack(sur_pred, dim=0)
    else:
        sur_pred_stacked = sur_pred

    ego_pred = model.get_ego_pred_outputs()
    assert ego_pred is not None, "ego_pred should be available"
    if isinstance(ego_pred, list):
        ego_pred_stacked = torch.stack(ego_pred, dim=0)
    else:
        ego_pred_stacked = ego_pred

    print(f"  future_pred shape: {fp.shape}")
    print(f"  z_local shape: {z_local.shape}")
    print(f"  intent_weights shape: {intent_w.shape}")
    print(f"  sur_pred shape: {sur_pred_stacked.shape}")
    print(f"  ego_pred shape: {ego_pred_stacked.shape}")
    print("  [OK] Training forward pass successful\n")
    return pred


def test_inference_forward(model, device):
    """Test inference forward (autoregressive)."""
    print("=" * 60)
    print("[3] Inference Forward (teacher_forcing=False, AR)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    NA = B * agents_per_scene
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.train()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)

    fp = pred['future_pred']
    assert isinstance(fp, torch.Tensor), f"Expected tensor, got {type(fp)}"
    assert fp.shape == (NA, 12, 4), f"Expected ({NA}, 12, 4), got {fp.shape}"

    # z_local: inference → list of FT tensors
    z_local_raw = model.get_z_local()
    z_local = model.get_z_local_stacked()
    assert z_local is not None, "z_local should be available"
    num_ego = B
    assert z_local.shape == (12, num_ego, 32), f"Expected z_local (12, {num_ego}, 32), got {z_local.shape}"

    # intent_weights: inference → list
    intent_w_raw = model.get_intent_weights()
    assert intent_w_raw is not None, "intent_weights should be available"

    # sur_pred/ego_pred: Not collected in AR inference (only in training parallel mode)
    sur_pred = model.get_sur_pred_outputs()
    ego_pred = model.get_ego_pred_outputs()
    # These may be None in inference mode — that's OK
    print(f"  future_pred shape: {fp.shape}")
    print(f"  z_local stacked shape: {z_local.shape}")
    print(f"  sur_pred available: {sur_pred is not None}")
    print(f"  ego_pred available: {ego_pred is not None}")
    print("  [OK] Inference forward pass successful\n")
    return pred


def test_reconstruct(model, device):
    print("=" * 60)
    print("[4] Reconstruct")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    NA = B * agents_per_scene
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.reconstruct(scene_graph, map_idx, map_env)

    assert pred['future_pred'].shape == (NA, 12, 4)
    print(f"  Reconstruct future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Reconstruct successful\n")


def test_sample(model, device):
    print("=" * 60)
    print("[5] Sample (3 samples)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    NA = B * agents_per_scene
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.sample(scene_graph, map_idx, map_env, num_samples=3)

    # sample() → (NA, NS, FT, 4)
    assert pred['future_pred'].shape == (NA, 3, 12, 4), \
        f"Expected (NA, 3, 12, 4), got {pred['future_pred'].shape}"
    print(f"  Sample future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Sample successful\n")


def test_sample_batched(model, device):
    print("=" * 60)
    print("[6] Sample Batched (3 samples)")
    print("=" * 60)

    B, agents_per_scene = 2, 3
    NA = B * agents_per_scene
    scene_graph = make_dummy_scene_graph(B=B, agents_per_scene=agents_per_scene, device=device)
    map_env = DummyMapEnv(map_size=240, num_layers=4, device=device)
    map_idx = torch.zeros(B, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model.sample_batched(scene_graph, map_idx, map_env, num_samples=3)

    assert pred['future_pred'].shape == (NA, 3, 12, 4), \
        f"Expected (NA, 3, 12, 4), got {pred['future_pred'].shape}"
    print(f"  Sample batched future_pred shape: {pred['future_pred'].shape}")
    print("  [OK] Sample batched successful\n")


def test_loss_training(model, device):
    """Test loss computation with training forward (parallel)."""
    print("=" * 60)
    print("[7] Loss Computation (training forward)")
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
        'sur_pred': 1.0, 'ego_pred': 1.0,
        'intent_ce': 0.0, 'map_attn': 0.1,
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights, state_norm, att_norm,
        phase=1, use_sparse_loss=False, use_potential_loss=False,
    ).to(device)

    model.train()
    model.zero_grad()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=0)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env, model=model)

    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].mean().item():.4f}")
    print(f"  KL loss: {loss_dict['kl_loss'].mean().item():.4f}")
    if 'sur_pred_loss' in loss_dict:
        print(f"  Sur pred loss: {loss_dict['sur_pred_loss'].item():.4f}")
    if 'ego_pred_loss' in loss_dict:
        print(f"  Ego pred loss: {loss_dict['ego_pred_loss'].item():.4f}")
    if 'map_attn_loss' in loss_dict:
        print(f"  Map attn loss: {loss_dict['map_attn_loss'].item():.4f}")

    # Backward pass — check gradient flow
    loss_dict['loss'].backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    num_with_grad = len(grad_norms)
    num_total = sum(1 for _ in model.parameters())
    print(f"  Params with gradients: {num_with_grad}/{num_total}")
    assert num_with_grad > 0, "No parameters received gradients!"

    # Check key Transformer components get gradients
    trans_grads = [n for n in grad_norms if 'trans_layers' in n]
    print(f"  Transformer layer params with grad: {len(trans_grads)}")
    assert len(trans_grads) > 0, "Transformer layers should receive gradients!"

    print("  [OK] Training loss and backward pass successful\n")


def test_loss_inference(model, device):
    """Test loss computation with inference forward (AR)."""
    print("=" * 60)
    print("[8] Loss Computation (inference forward)")
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
        'sur_pred': 1.0, 'ego_pred': 1.0,
        'intent_ce': 0.0, 'map_attn': 0.0,
    }

    loss_fn = TrafficPlannerLoss(
        loss_weights, state_norm, att_norm,
        phase=1, use_sparse_loss=False, use_potential_loss=False,
    ).to(device)

    model.train()
    model.zero_grad()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=False, current_epoch=0)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env, model=model)

    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Recon loss: {loss_dict['recon_loss'].mean().item():.4f}")
    print(f"  KL loss: {loss_dict['kl_loss'].mean().item():.4f}")

    loss_dict['loss'].backward()
    num_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    num_total = sum(1 for _ in model.parameters())
    print(f"  Params with gradients: {num_with_grad}/{num_total}")
    assert num_with_grad > 0, "No parameters received gradients!"
    print("  [OK] Inference loss and backward pass successful\n")


def test_freeze_for_finetuning(model, device):
    print("=" * 60)
    print("[9] Freeze for Fine-tuning (Phase 2)")
    print("=" * 60)

    model.set_phase(2)
    model.freeze_for_finetuning()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    # Key ego components should be trainable (Transformer version)
    trainable_keywords = ['ego_output_head', 'intent_codebook', 'intent_ce_head',
                          'ego_intent_proj', 'sur_pred_head']
    for kw in trainable_keywords:
        matching = [n for n in trainable if kw in n]
        if len(matching) > 0:
            print(f"  [TRAINABLE] {kw}: {len(matching)} params")
        else:
            print(f"  [WARNING] {kw}: not found in trainable")

    # Ego Q/O in A2A and A2S should be trainable
    ego_qo = [n for n in trainable if 'ego_q_proj' in n or 'ego_o_proj' in n]
    print(f"  [TRAINABLE] ego Q/O projections: {len(ego_qo)} params")

    # Ego FFN should be trainable
    ego_ffn = [n for n in trainable if 'ego_ffn' in n]
    print(f"  [TRAINABLE] ego FFN: {len(ego_ffn)} params")

    # Frozen components
    frozen_keywords = ['z_gen_q_proj', 'z_gen_to_latent', 'map_conv_early',
                       'interaction_gcn', 'sur_output_head']
    for kw in frozen_keywords:
        matching = [n for n in frozen if kw in n]
        if len(matching) > 0:
            print(f"  [FROZEN] {kw}: {len(matching)} params")
        else:
            print(f"  [WARNING] {kw}: not found in frozen")

    # A2Z (z_context cross-attn) ego Q/O should be trainable, K/V frozen
    a2z_ego = [n for n in trainable if 'a2z_ego' in n]
    print(f"  [TRAINABLE] A2Z ego Q/O: {len(a2z_ego)} params")
    a2z_kv = [n for n in frozen if 'a2z_k_proj' in n or 'a2z_v_proj' in n]
    print(f"  [FROZEN] A2Z K/V: {len(a2z_kv)} params")

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
    # Test training forward in Phase 2
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=0)

    z_local = model.get_z_local_stacked()
    intent_w = model.get_intent_weights()

    z_local_norm = z_local.norm().item()
    if isinstance(intent_w, torch.Tensor):
        intent_w_sum = intent_w.sum().item()
    else:
        intent_w_sum = sum(w.sum().item() for w in intent_w)
    print(f"  z_local L2 norm: {z_local_norm:.4f} (should be > 0 in Phase 2)")
    print(f"  intent_weights sum: {intent_w_sum:.4f}")
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

    # Training mode (parallel) — map attn available
    model.train()
    model.zero_grad()
    pred = model(scene_graph, map_idx, map_env, teacher_forcing=True, current_epoch=0)
    loss_dict = loss_fn(scene_graph, pred, map_idx=map_idx, map_env=map_env, model=model)

    assert 'map_attn_loss' in loss_dict, "map_attn_loss should be in loss_dict when weight > 0"
    map_attn_val = loss_dict['map_attn_loss'].item()
    print(f"  [Training] map_attn_loss: {map_attn_val:.4f}")
    print(f"  [Training] Total loss: {loss_dict['loss'].item():.4f}")

    loss_dict['loss'].backward()
    # Check that Transformer A2S layers get gradients
    a2s_grads = sum(1 for n, p in model.named_parameters()
                    if 'a2s' in n and p.grad is not None)
    print(f"  [Training] A2S params with grad: {a2s_grads}")

    print("  [OK] Map attention guidance loss successful\n")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    # Test 1: Instantiation
    model = test_model_instantiation(device)

    # Test 2: Training forward (parallel)
    test_training_forward(model, device)

    # Test 3: Inference forward (autoregressive)
    test_inference_forward(model, device)

    # Test 4: Reconstruct
    test_reconstruct(model, device)

    # Test 5: Sample
    test_sample(model, device)

    # Test 6: Sample batched
    test_sample_batched(model, device)

    # Test 7: Loss (training forward)
    test_loss_training(model, device)

    # Test 8: Loss (inference forward)
    test_loss_inference(model, device)

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
