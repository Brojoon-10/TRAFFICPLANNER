# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.individual_interaction_net import IndividualSceneInteractionNet
from models.common import MLP, car_dynamics


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for temporal information in sliding window.
    From fit_traffic_model_trans.py
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (Batch, Seq, Feature) [batch_first=True]
        """
        if x.dim() == 3:  # (B, T, D)
            pe_to_add = self.pe[:x.size(1)].permute(1, 0, 2)  # (T, 1, D) -> (1, T, D)
            x = x + pe_to_add
        else:  # (T, B, D)
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ============================================================
# New Decoder Modules (Redesign)
# ============================================================


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: modulates LN output using z_global.
    y = gamma * LayerNorm(x) + beta, where (gamma, beta) = MLP(z_global).
    Identity-initialized so initial behavior matches standard LayerNorm.
    """
    def __init__(self, d_model, z_size):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.adaln_mlp = nn.Sequential(
            nn.Linear(z_size, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        # Identity init: gamma=1, beta=0 at start
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)
        with torch.no_grad():
            self.adaln_mlp[-1].bias[:d_model] = 1.0

    def forward(self, x, z_cond):
        """x: (..., D), z_cond: broadcastable, last dim = z_size"""
        normed = self.norm(x)
        gamma, beta = self.adaln_mlp(z_cond).chunk(2, dim=-1)
        return gamma * normed + beta


class A2ARelativeBias(nn.Module):
    """Compute per-head additive attention bias from pairwise physical features.
    Zero-initialized output so initial behavior matches standard attention.
    """
    def __init__(self, num_features=8, nhead=8, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead),
        )
        # Zero-init: no bias at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, rel_features):
        """rel_features: (BT, N, N, num_features) -> bias: (BT, nhead, N, N)"""
        return self.mlp(rel_features).permute(0, 3, 1, 2)


# ============================================================
# Context Encoder Modules (Enc-Dec Cross-Attention Redesign)
# ============================================================


class MapSummaryPooling(nn.Module):
    """Learnable query pooling to compress map tokens into summary tokens.
    Uses cross-attention: K learned queries attend to 3249 map tokens → K summary tokens.
    """
    def __init__(self, num_queries=8, d_model=128, map_ch=64, nhead=4):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.map_proj = nn.Linear(map_ch, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, map_tokens):
        """
        :param map_tokens: (NA, num_tokens, map_ch) — CNN conv3 spatial features
        :return: (NA, num_queries, d_model) — compressed map summary
        """
        NA = map_tokens.size(0)
        kv = self.map_proj(map_tokens)  # (NA, num_tokens, d_model)
        q = self.queries.unsqueeze(0).expand(NA, -1, -1)  # (NA, num_queries, d_model)
        summary, _ = self.cross_attn(q, kv, kv)  # (NA, num_queries, d_model)
        return summary


class ContextEncoder(nn.Module):
    """Encode past + z_global + map_summary into context tokens via self-attention.
    The self-attention mixes all three information sources, making them inseparable.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=2,
                 gcn_feat_dim=64, z_size=32, num_z_tokens=4,
                 past_len=4, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.num_z_tokens = num_z_tokens
        self.d_model = d_model

        self.past_proj = nn.Linear(gcn_feat_dim, d_model)
        self.z_proj = nn.Linear(z_size, num_z_tokens * d_model)
        self.past_pe = nn.Embedding(past_len, d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, past_gcn_feats, z_global, map_summary):
        """
        :param past_gcn_feats: (NA, PT, gcn_feat_dim) — per-step GCN features
        :param z_global: (NA, z_size)
        :param map_summary: (NA, M, d_model) — from MapSummaryPooling
        :return: context_tokens (NA, PT+K+M, d_model)
        """
        NA = past_gcn_feats.size(0)
        PT = past_gcn_feats.size(1)
        device = past_gcn_feats.device

        # Past tokens with temporal PE
        past_tokens = self.past_proj(past_gcn_feats)  # (NA, PT, d_model)
        past_pe = self.past_pe(torch.arange(PT, device=device))  # (PT, d_model)
        past_tokens = past_tokens + past_pe.unsqueeze(0)  # broadcast

        # z tokens (no PE — order-free latent)
        z_tokens = self.z_proj(z_global).view(NA, self.num_z_tokens, self.d_model)

        # Concatenate: [past, z, map_summary]
        context = torch.cat([past_tokens, z_tokens, map_summary], dim=1)

        # Self-attention mixes all information
        context = self.encoder(context)
        return context


# ============================================================
# Transformer Decoder Modules
# ============================================================

class SharedKVAttention(nn.Module):
    """
    Attention with shared K/V projections and separate ego/sur Q/O projections.
    Used for A2A (agent interaction) and A2S (map cross-attention).
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        # Shared K/V projections
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Ego Q/O projections
        self.ego_q_proj = nn.Linear(d_model, d_model)
        self.ego_o_proj = nn.Linear(d_model, d_model)

        # Sur Q/O projections
        self.sur_q_proj = nn.Linear(d_model, d_model)
        self.sur_o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _attention(self, q, k, v, attn_mask=None, rel_bias=None):
        """
        :param q: (B, nhead, Nq, head_dim)
        :param k: (B, nhead, Nkv, head_dim)
        :param v: (B, nhead, Nkv, head_dim)
        :param attn_mask: (Nq, Nkv) or None
        :param rel_bias: (B, nhead, Nq, Nkv) or None — additive attention bias
        :return: (attn_out, attn_weights)
        """
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nhead, Nq, Nkv)
        if rel_bias is not None:
            attn_scores = attn_scores + rel_bias
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # mask is -inf for blocked positions
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # (B, nhead, Nq, head_dim)
        return attn_out, attn_weights

    def _reshape_to_heads(self, x):
        """(B, N, D) -> (B, nhead, N, head_dim)"""
        B, N, D = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query_tokens, kv_tokens, ego_mask, attn_mask=None,
                return_attn_weights=False, rel_bias=None):
        """
        :param query_tokens: (B, N, D) — agent tokens (query source)
        :param kv_tokens: (B, N_kv, D) — key/value source (same as query for A2A, map_tokens for A2S)
        :param ego_mask: (N,) bool — True for ego agents
        :param attn_mask: optional attention mask
        :param return_attn_weights: whether to return attention weights
        :param rel_bias: (B, nhead, N, N_kv) or None — pairwise attention bias
        :return: output (B, N, D), optionally (ego_attn_weights, sur_attn_weights)
        """
        B, N, D = query_tokens.shape
        N_kv = kv_tokens.shape[1]

        # Shared K/V
        k = self._reshape_to_heads(self.k_proj(kv_tokens))  # (B, nhead, N_kv, head_dim)
        v = self._reshape_to_heads(self.v_proj(kv_tokens))  # (B, nhead, N_kv, head_dim)

        # Separate Q for ego/sur
        ego_idx = ego_mask.nonzero(as_tuple=True)[0]
        sur_idx = (~ego_mask).nonzero(as_tuple=True)[0]

        output = query_tokens.new_zeros(B, N, D)
        ego_attn_w = sur_attn_w = None

        if ego_idx.numel() > 0:
            ego_q_in = query_tokens[:, ego_idx]  # (B, N_ego, D)
            ego_q = self._reshape_to_heads(self.ego_q_proj(ego_q_in))
            ego_bias = rel_bias[:, :, ego_idx, :] if rel_bias is not None else None
            ego_attn_out, ego_attn_w = self._attention(ego_q, k, v, attn_mask, ego_bias)
            # (B, nhead, N_ego, head_dim) -> (B, N_ego, D)
            ego_attn_out = ego_attn_out.transpose(1, 2).contiguous().view(B, ego_idx.numel(), D)
            output[:, ego_idx] = self.ego_o_proj(ego_attn_out)

        if sur_idx.numel() > 0:
            sur_q_in = query_tokens[:, sur_idx]  # (B, N_sur, D)
            sur_q = self._reshape_to_heads(self.sur_q_proj(sur_q_in))
            sur_bias = rel_bias[:, :, sur_idx, :] if rel_bias is not None else None
            sur_attn_out, sur_attn_w = self._attention(sur_q, k, v, attn_mask, sur_bias)
            sur_attn_out = sur_attn_out.transpose(1, 2).contiguous().view(B, sur_idx.numel(), D)
            output[:, sur_idx] = self.sur_o_proj(sur_attn_out)

        if return_attn_weights:
            return output, ego_attn_w, sur_attn_w
        return output


class TransDecoderLayer(nn.Module):
    """
    One Transformer decoder layer: A2A → A2C → A2S → FFN.
    (Enc-Dec Cross-Attention redesign: A2T/A2Z/AdaLN removed)

    A2A: K/V shared + Q/O separate (ego/sur) — agent interaction
    A2C: context cross-attention (shared K/V + ego/sur Q/O) — past+z+map_summary
    A2S: K/V shared + Q/O separate (ego/sur) — high-res map cross-attention
    FFN: ego/sur fully separate
    """
    def __init__(self, d_model, nhead, ffn_dim, map_token_dim, dropout=0.1,
                 use_a2a_rel_bias=False):
        super().__init__()
        self.d_model = d_model
        self.use_a2a_rel_bias = use_a2a_rel_bias

        # A2A: shared K/V + ego/sur separate Q/O
        self.a2a_norm = nn.LayerNorm(d_model)
        self.a2a_attn = SharedKVAttention(d_model, nhead, dropout)

        # A2A relative physical bias
        if use_a2a_rel_bias:
            self.a2a_rel_bias_module = A2ARelativeBias(
                num_features=8, nhead=nhead, hidden_dim=32)

        # A2C: context cross-attention (shared K/V + ego/sur Q/O)
        self.a2c_norm = nn.LayerNorm(d_model)
        self.a2c_ctx_k_proj = nn.Linear(d_model, d_model)
        self.a2c_ctx_v_proj = nn.Linear(d_model, d_model)
        self.a2c_ego_q_proj = nn.Linear(d_model, d_model)
        self.a2c_ego_o_proj = nn.Linear(d_model, d_model)
        self.a2c_sur_q_proj = nn.Linear(d_model, d_model)
        self.a2c_sur_o_proj = nn.Linear(d_model, d_model)
        self.a2c_nhead = nhead
        self.a2c_head_dim = d_model // nhead
        self.a2c_scale = self.a2c_head_dim ** -0.5
        self.a2c_dropout = nn.Dropout(dropout)

        # A2S: shared K/V (map) + ego/sur separate Q/O
        self.a2s_norm = nn.LayerNorm(d_model)
        self.a2s_map_k_proj = nn.Linear(map_token_dim, d_model)
        self.a2s_map_v_proj = nn.Linear(map_token_dim, d_model)
        self.a2s_ego_q_proj = nn.Linear(d_model, d_model)
        self.a2s_ego_o_proj = nn.Linear(d_model, d_model)
        self.a2s_sur_q_proj = nn.Linear(d_model, d_model)
        self.a2s_sur_o_proj = nn.Linear(d_model, d_model)
        self.a2s_nhead = nhead
        self.a2s_head_dim = d_model // nhead
        self.a2s_scale = self.a2s_head_dim ** -0.5
        self.a2s_dropout = nn.Dropout(dropout)

        # FFN: ego/sur fully separate
        self.ego_ffn_norm = nn.LayerNorm(d_model)
        self.ego_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.sur_ffn_norm = nn.LayerNorm(d_model)
        self.sur_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def _a2c_attention(self, query_tokens, context, ego_mask):
        """
        A2C: context cross-attention with shared K/V and separate ego/sur Q/O.
        :param query_tokens: (BT, N, D) — decoder tokens
        :param context: (N, C, D) — context encoder output (fixed across steps)
        :return: output (BT, N, D)
        """
        BT, N, D = query_tokens.shape
        nhead = self.a2c_nhead
        head_dim = self.a2c_head_dim
        C = context.size(1)

        # Shared K/V from context: (N, C, D) → project → broadcast over BT
        ctx_k = self.a2c_ctx_k_proj(context)  # (N, C, D)
        ctx_v = self.a2c_ctx_v_proj(context)  # (N, C, D)

        ego_idx = ego_mask.nonzero(as_tuple=True)[0]
        sur_idx = (~ego_mask).nonzero(as_tuple=True)[0]

        output = query_tokens.new_zeros(BT, N, D)

        def _do_a2c(agent_idx, q_proj, o_proj):
            if agent_idx.numel() == 0:
                return
            # Q from decoder tokens: (BT, N_sel, D)
            q_in = query_tokens[:, agent_idx]
            q = q_proj(q_in).view(BT, agent_idx.numel(), nhead, head_dim).transpose(1, 2)
            # (BT, nhead, N_sel, head_dim)

            # K/V from context: (N_sel, C, D) → broadcast over BT
            k_sel = ctx_k[agent_idx]  # (N_sel, C, D)
            v_sel = ctx_v[agent_idx]
            k_sel = k_sel.view(agent_idx.numel(), C, nhead, head_dim)
            k_sel = k_sel.permute(2, 0, 1, 3).unsqueeze(0).expand(BT, -1, -1, -1, -1)
            # (BT, nhead, N_sel, C, head_dim)
            v_sel = v_sel.view(agent_idx.numel(), C, nhead, head_dim)
            v_sel = v_sel.permute(2, 0, 1, 3).unsqueeze(0).expand(BT, -1, -1, -1, -1)

            # Attention: Q(BT, nhead, N_sel, 1, head_dim) @ K^T(BT, nhead, N_sel, head_dim, C)
            q = q.unsqueeze(3)  # (BT, nhead, N_sel, 1, head_dim)
            scores = torch.matmul(q, k_sel.transpose(-2, -1)) * self.a2c_scale
            # (BT, nhead, N_sel, 1, C)
            attn_w = F.softmax(scores, dim=-1)
            attn_w = self.a2c_dropout(attn_w)
            attn_out = torch.matmul(attn_w, v_sel)  # (BT, nhead, N_sel, 1, head_dim)
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().view(BT, agent_idx.numel(), D)
            output[:, agent_idx] = o_proj(attn_out)

        _do_a2c(ego_idx, self.a2c_ego_q_proj, self.a2c_ego_o_proj)
        _do_a2c(sur_idx, self.a2c_sur_q_proj, self.a2c_sur_o_proj)

        return output

    def _a2s_attention(self, query_tokens, map_tokens, ego_mask):
        """
        A2S: map cross-attention with shared K/V and separate ego/sur Q/O.
        :param query_tokens: (BT, N, D)
        :param map_tokens: (N, num_tokens, map_dim) — per-agent map crop
        :return: output (BT, N, D), ego_attn_weights, sur_attn_weights
        """
        BT, N, D = query_tokens.shape
        nhead = self.a2s_nhead
        head_dim = self.a2s_head_dim

        # Map tokens: (N, num_tokens, ch) → project → broadcast
        map_k = self.a2s_map_k_proj(map_tokens)  # (N, num_tokens, D)
        map_v = self.a2s_map_v_proj(map_tokens)
        map_k = map_k.unsqueeze(0).expand(BT, -1, -1, -1)  # (BT, N, num_tokens, D)
        map_v = map_v.unsqueeze(0).expand(BT, -1, -1, -1)
        num_tokens = map_k.shape[2]

        ego_idx = ego_mask.nonzero(as_tuple=True)[0]
        sur_idx = (~ego_mask).nonzero(as_tuple=True)[0]

        output = query_tokens.new_zeros(BT, N, D)
        ego_attn_w = sur_attn_w = None

        def _do_a2s(agent_idx, q_proj, o_proj):
            if agent_idx.numel() == 0:
                return None
            q_in = query_tokens[:, agent_idx]
            q = q_proj(q_in).view(BT, agent_idx.numel(), nhead, head_dim).transpose(1, 2)
            k_sel = map_k[:, agent_idx].view(BT, agent_idx.numel(), num_tokens, nhead, head_dim).permute(0, 3, 1, 2, 4)
            v_sel = map_v[:, agent_idx].view(BT, agent_idx.numel(), num_tokens, nhead, head_dim).permute(0, 3, 1, 2, 4)

            q = q.unsqueeze(3)  # (BT, nhead, N_sel, 1, head_dim)
            scores = torch.matmul(q, k_sel.transpose(-2, -1)) * self.a2s_scale
            attn_w = F.softmax(scores, dim=-1)
            attn_w = self.a2s_dropout(attn_w)
            attn_out = torch.matmul(attn_w, v_sel)
            attn_out = attn_out.squeeze(3).transpose(1, 2).contiguous().view(BT, agent_idx.numel(), D)
            output[:, agent_idx] = o_proj(attn_out)
            return attn_w.squeeze(3).mean(dim=1)

        ego_attn_w = _do_a2s(ego_idx, self.a2s_ego_q_proj, self.a2s_ego_o_proj)
        sur_attn_w = _do_a2s(sur_idx, self.a2s_sur_q_proj, self.a2s_sur_o_proj)

        return output, ego_attn_w, sur_attn_w

    def forward(self, x, ego_mask, context, map_tokens,
                return_a2a_output=False, return_a2s_weights=False,
                agent_states=None):
        """
        :param x: (B, T, N, D) — decoder tokens (T=1 for AR single-step)
        :param ego_mask: (N,) bool — True for ego
        :param context: (N, C, D) — context encoder output
        :param map_tokens: (N, num_tokens, map_dim) — per-agent map crop
        :param return_a2a_output: return tokens after A2A (for pred loss)
        :param return_a2s_weights: return A2S attention weights (for map guidance loss)
        :param agent_states: (B*T, N, 6) or None — for A2A relative bias
        :return: x, extras_dict
        """
        B, T, N, D = x.shape
        extras = {}

        # --- A2A: agent interaction (shared K/V + ego/sur Q/O) ---
        x_a2a = x.reshape(B * T, N, D)
        x_norm = self.a2a_norm(x_a2a)

        # Compute A2A relative bias from agent states
        a2a_rel_bias = None
        if self.use_a2a_rel_bias and agent_states is not None:
            rel_features = self._compute_rel_features_static(agent_states)
            a2a_rel_bias = self.a2a_rel_bias_module(rel_features)

        a2a_out = self.a2a_attn(x_norm, x_norm, ego_mask, rel_bias=a2a_rel_bias)
        x_a2a = x_a2a + a2a_out

        if return_a2a_output:
            extras['a2a_output'] = x_a2a.view(B, T, N, D)

        # --- A2C: context cross-attention ---
        x_a2c = x_a2a  # (BT, N, D) — stays flat
        x_norm = self.a2c_norm(x_a2c)
        a2c_out = self._a2c_attention(x_norm, context, ego_mask)
        x_a2c = x_a2c + a2c_out

        # --- A2S: map cross-attention ---
        x_a2s = x_a2c  # (BT, N, D)
        x_norm = self.a2s_norm(x_a2s)
        a2s_out, ego_map_w, sur_map_w = self._a2s_attention(x_norm, map_tokens, ego_mask)
        x_a2s = x_a2s + a2s_out

        if return_a2s_weights:
            extras['ego_map_attn_weights'] = ego_map_w
            extras['sur_map_attn_weights'] = sur_map_w

        x = x_a2s.view(B, T, N, D)

        # --- FFN: ego/sur separate ---
        ego_idx = ego_mask.nonzero(as_tuple=True)[0]
        sur_idx = (~ego_mask).nonzero(as_tuple=True)[0]

        if ego_idx.numel() > 0:
            ego_tokens = x[:, :, ego_idx]
            ego_normed = self.ego_ffn_norm(ego_tokens)
            x[:, :, ego_idx] = ego_tokens + self.ego_ffn(ego_normed)

        if sur_idx.numel() > 0:
            sur_tokens = x[:, :, sur_idx]
            sur_normed = self.sur_ffn_norm(sur_tokens)
            x[:, :, sur_idx] = sur_tokens + self.sur_ffn(sur_normed)

        return x, extras

    @staticmethod
    def _compute_rel_features_static(states):
        """Compute 8 pairwise relative physical features from agent states.
        :param states: (BT, N, 6) — (x, y, hx, hy, speed, hdot)
        :return: (BT, N, N, 8)
        """
        pos = states[:, :, :2]           # (BT, N, 2)
        hd = states[:, :, 2:4]           # (BT, N, 2) = (hx, hy) unit vec
        speed = states[:, :, 4:5]        # (BT, N, 1)

        # Position difference: j - i for each pair
        dp = pos.unsqueeze(1) - pos.unsqueeze(2)  # (BT, N, N, 2)

        # Rotate dp into query agent i's heading frame
        hx_i = hd[:, :, 0:1].unsqueeze(2)  # (BT, N, 1, 1)
        hy_i = hd[:, :, 1:2].unsqueeze(2)
        rel_long = dp[..., 0:1] * hx_i + dp[..., 1:2] * hy_i
        rel_lat = -dp[..., 0:1] * hy_i + dp[..., 1:2] * hx_i

        dist = (dp ** 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt()

        # Velocity vectors and relative velocity
        vel = hd * speed  # (BT, N, 2)
        dv = vel.unsqueeze(1) - vel.unsqueeze(2)  # (BT, N, N, 2)
        rel_v_long = dv[..., 0:1] * hx_i + dv[..., 1:2] * hy_i
        rel_v_lat = -dv[..., 0:1] * hy_i + dv[..., 1:2] * hx_i

        # Closing speed: positive = approaching
        closing = -(dp * dv).sum(-1, keepdim=True) / (dist + 1e-6)
        # TTC: clamp closing to avoid division by zero/negative
        ttc = (dist / closing.clamp(min=0.1)).clamp(0, 10.0)

        # Relative heading via cross/dot product of heading vectors
        hx_j = hd[:, :, 0:1].unsqueeze(1)  # (BT, 1, N, 1)
        hy_j = hd[:, :, 1:2].unsqueeze(1)
        cross = hx_i * hy_j - hy_i * hx_j
        dot = hx_i * hx_j + hy_i * hy_j
        rel_heading = torch.atan2(cross, dot)

        return torch.cat([rel_long, rel_lat, dist, rel_v_long,
                          rel_v_lat, closing, ttc, rel_heading], dim=-1)


class IntentCodebook(nn.Module):
    """
    Discrete intent codebook for z_local.
    K learnable intent vectors, selected via Gumbel-Softmax.
    Always active — on/off controlled by use_ego_z_local / use_sur_z_local flags in model.
    """
    def __init__(self, num_intents=8, intent_dim=32, input_dim=73):
        super().__init__()
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.codebook = nn.Embedding(num_intents, intent_dim)
        self.intent_predictor = MLP([input_dim, 64, num_intents])

    def forward(self, situation, temperature=1.0):
        """
        :param situation: (N, input_dim) — token features
        :param temperature: Gumbel-Softmax temperature
        :return: (z_local, intent_weights)
            z_local: (N, intent_dim)
            intent_weights: (N, K)
        """
        logits = self.intent_predictor(situation)  # (N, K)

        if self.training:
            intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            intent_weights = F.softmax(logits / temperature, dim=-1)

        z_local = intent_weights @ self.codebook.weight  # (N, intent_dim)
        return z_local, intent_weights


from datasets.utils import normalize_scene_graph
from utils.transforms import transform2frame, kinematics2angle, kinematics2vec
from utils.torch import calc_conv_out
from utils.logger import throw_err, Logger

class TrafficPlannerModel(nn.Module):
    """
    Traffic + Planner Model (Redesign).

    Key changes from v4:
    - Symmetric ego/sur decoder: both get History Attention + Map Cross-Attention + GRU
    - Single interaction GCN for history buffer (replaces decoder_net + z_local_gcn)
    - Intent Codebook for z_local (discrete, ego only, Phase 2 active)
    - Map tokens: conv3 spatial features for cross-attention
    - Auxiliary loss heads: sur_pred, ego_pred, map_attn_guidance, intent_ce
    """
    def __init__(self, npast, nfuture, map_obs_size_pix, nclasses,
                 map_feat_size=64,
                 past_feat_size=64,
                 future_feat_size=64,
                 latent_size=32,          # z_global size
                 z_local_size=32,         # intent_dim
                 output_bicycle=True,
                 dt=0.5,
                 gcn_hidden_dim=64,
                 transformer_nhead=8,
                 transformer_nlayer=3,
                 conv_channel_in=4,
                 conv_kernel_list=[7, 5, 5, 3, 3, 3],
                 conv_stride_list=[2, 2, 2, 2, 2, 2],
                 conv_filter_list=[16, 32, 64, 64, 128, 128],
                 # New redesign params
                 num_intents=8,           # K for intent codebook
                 sur_pred_dim=2,          # predicted sur delta dim (dx, dy)
                 map_recrop=False,        # re-crop map tokens every decode step
                 # Transformer decoder params
                 trans_num_layers=4,
                 trans_d_model=128,
                 trans_nhead=8,
                 trans_ffn_dim=512,
                 trans_dropout=0.1,
                 use_ego_z_local=True,
                 use_sur_z_local=False,
                 # Enc-Dec Cross-Attention params
                 use_a2a_rel_bias=False,
                 num_z_tokens=4,
                 enc_dropout=0.1,
                 context_num_layers=2,
                 map_summary_tokens=8,
                 ):
        super(TrafficPlannerModel, self).__init__()
        self.normalizer = self.att_normalizer = None
        self.PT = npast
        self.FT = nfuture

        self.dt = dt
        print(f'time interval : {self.dt}')

        self.NC = nclasses
        self.output_bicycle = output_bicycle
        if self.output_bicycle:
            self.bicycle_params = None
            Logger.log('Using bicycle model as output parameterization of model...')

        self.state_size = 6  # (x,y,hx,hy,s,hdot)
        self.att_feat_size = 2  # (l,w)
        self.gcn_hidden_dim = gcn_hidden_dim
        self.past_feat_size = past_feat_size  # D = 64
        self.d_model = gcn_hidden_dim  # D = 64 throughout

        # Store redesign params
        self.num_intents = num_intents
        self.sur_pred_dim = sur_pred_dim
        self.intent_dim = z_local_size  # 32

        #
        # Map encoding — split into early (conv1-3, for tokens) and late (conv4-6, for encoder feat)
        #
        self.mapH = map_obs_size_pix
        self.mapW = map_obs_size_pix
        self.map_obs_size_pix = map_obs_size_pix

        # Build conv layers, tracking intermediate sizes
        assert len(conv_kernel_list) == len(conv_stride_list) == len(conv_filter_list)
        conv_filter_list_full = [conv_channel_in] + conv_filter_list

        # conv1-3: early layers (for map tokens)
        early_layers = []
        self.map_token_spatial = map_obs_size_pix
        for lidx in range(3):
            cur_conv = nn.Conv2d(conv_filter_list_full[lidx],
                                 conv_filter_list_full[lidx+1],
                                 kernel_size=conv_kernel_list[lidx],
                                 stride=conv_stride_list[lidx],
                                 padding=0)
            cur_gn = nn.GroupNorm(1, conv_filter_list_full[lidx+1])
            early_layers.extend([cur_conv, cur_gn, nn.ReLU()])
            self.map_token_spatial = calc_conv_out(self.map_token_spatial, conv_kernel_list[lidx], conv_stride_list[lidx])

        self.map_conv_early = nn.Sequential(*early_layers)
        self.map_token_ch = conv_filter_list_full[3]  # 64
        self.map_num_tokens = self.map_token_spatial * self.map_token_spatial

        # Receptive field params for soft label coordinate mapping
        # token[m] RF center pixel = m * map_rf_stride + map_rf_offset
        _rf_stride = 1
        _rf_offset = 0.0
        for lidx in range(3):
            _rf_offset += (conv_kernel_list[lidx] - 1) / 2.0 * _rf_stride
            _rf_stride *= conv_stride_list[lidx]
        self.map_rf_stride = _rf_stride
        self.map_rf_offset = _rf_offset

        print(f'Map tokens: {self.map_token_spatial}x{self.map_token_spatial} = {self.map_num_tokens} tokens, ch={self.map_token_ch}, RF stride={_rf_stride}, offset={_rf_offset}')

        # conv4-6: late layers (for encoder 64-dim feature)
        late_layers = []
        final_conv_out = self.map_token_spatial
        for lidx in range(3, len(conv_kernel_list)):
            cur_conv = nn.Conv2d(conv_filter_list_full[lidx],
                                 conv_filter_list_full[lidx+1],
                                 kernel_size=conv_kernel_list[lidx],
                                 stride=conv_stride_list[lidx],
                                 padding=0)
            cur_gn = nn.GroupNorm(1, conv_filter_list_full[lidx+1])
            late_layers.extend([cur_conv, cur_gn, nn.ReLU()])
            final_conv_out = calc_conv_out(final_conv_out, conv_kernel_list[lidx], conv_stride_list[lidx])

        self.map_conv_late = nn.Sequential(*late_layers)
        self.map_feat_in_size = conv_filter_list_full[-1] * final_conv_out * final_conv_out
        self.map_feat_out_size = map_feat_size
        self.map_feature = nn.Linear(self.map_feat_in_size, self.map_feat_out_size)

        #
        # Temporal GCN Encoder (per-step GCN → temporal encoding) — UNCHANGED
        #
        step_input_size = self.state_size + self.att_feat_size + 1 + self.NC
        self.step_feature_extractor = MLP([step_input_size, 128, gcn_hidden_dim])

        self.temporal_gcn_encoder = IndividualSceneInteractionNet(
            gcn_hidden_dim, self.NC, 4, 128, gcn_hidden_dim,
        )

        # Prior temporal encoding: GRU
        self.prior_temporal_gru = nn.GRU(
            gcn_hidden_dim, gcn_hidden_dim, 2, batch_first=True,
        )

        # Posterior temporal encoding: PE + Transformer
        self.positional_encoding = PositionalEncoding(gcn_hidden_dim, dropout=enc_dropout, max_len=max(self.PT, self.FT))
        encoder_layer = TransformerEncoderLayer(
            d_model=gcn_hidden_dim, nhead=transformer_nhead, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_nlayer)

        #
        # Latent variable sizes
        #
        self.z_size = latent_size  # 32
        self.z_local_size = z_local_size  # 32 (intent_dim)

        #
        # Map recrop
        #
        self.map_recrop = map_recrop

        #
        # Prior/Posterior latent networks — UNCHANGED
        #
        self.latent_prior_net = MLP([
            gcn_hidden_dim + self.map_feat_out_size + self.NC,
            128,
            self.z_size * 2
        ])
        self.latent_posterior_net = MLP([
            gcn_hidden_dim * 2 + self.map_feat_out_size + self.NC,
            128,
            self.z_size * 2
        ])

        if self.output_bicycle:
            self.traj_out_size = 2  # (a, hdot)
        else:
            self.traj_out_size = 4  # (x, y, hx, hy)

        #
        # =============================================
        # TRANSFORMER DECODER COMPONENTS (Enc-Dec Cross-Attention)
        # =============================================
        #
        self.trans_d_model = trans_d_model  # 128
        self.trans_num_layers = trans_num_layers
        self.use_ego_z_local = use_ego_z_local
        self.use_sur_z_local = use_sur_z_local
        self.use_a2a_rel_bias = use_a2a_rel_bias

        # 1. Interaction GCN: per-step agent interaction -> token features (64dim)
        interaction_gcn_in = self.state_size + self.att_feat_size + self.NC
        self.interaction_gcn = IndividualSceneInteractionNet(
            interaction_gcn_in, self.NC, 4, 64, self.gcn_hidden_dim,
        )

        # 2. MapSummaryPooling: map_tokens(3249, 64) → summary(M, 128)
        self.map_summary_pooling = MapSummaryPooling(
            num_queries=map_summary_tokens,
            d_model=trans_d_model,
            map_ch=self.map_token_ch,
            nhead=4,
        )

        # 3. Context Encoder: past_gcn(4,64) + z(K,128) + map_summary(M,128) → context
        self.context_encoder = ContextEncoder(
            d_model=trans_d_model,
            nhead=trans_nhead,
            num_layers=context_num_layers,
            gcn_feat_dim=self.gcn_hidden_dim,
            z_size=self.z_size,
            num_z_tokens=num_z_tokens,
            past_len=self.PT,
            ffn_dim=trans_d_model * 2,
            dropout=trans_dropout,
        )

        # 4. Decoder query construction: GCN(64) + lw(2) + sem(NC) → d_model(128)
        query_input_dim = self.gcn_hidden_dim + self.att_feat_size + self.NC
        self.query_proj = nn.Linear(query_input_dim, trans_d_model)

        # 5. Decoder step PE: nn.Embedding(FT=12, d_model)
        self.step_pe = nn.Embedding(self.FT, trans_d_model)

        # 6. Transformer decoder layers: A2A → A2C → A2S → FFN
        self.trans_layers = nn.ModuleList([
            TransDecoderLayer(trans_d_model, trans_nhead, trans_ffn_dim,
                              self.map_token_ch, trans_dropout,
                              use_a2a_rel_bias=use_a2a_rel_bias)
            for _ in range(trans_num_layers)
        ])

        # 5. Intent Codebook (ego, between Layer 0 and Layer 1)
        self.intent_codebook = IntentCodebook(
            num_intents=num_intents,
            intent_dim=self.intent_dim,
            input_dim=trans_d_model,  # 128 (Layer 0 output)
        )
        # z_local concat projection: (128 + 32) → 128
        self.ego_intent_proj = nn.Linear(trans_d_model + self.intent_dim, trans_d_model)

        # Sur Intent Codebook (optional, controlled by use_sur_z_local)
        if use_sur_z_local:
            self.sur_intent_codebook = IntentCodebook(
                num_intents=num_intents,
                intent_dim=self.intent_dim,
                input_dim=trans_d_model,
            )
            self.sur_intent_proj = nn.Linear(trans_d_model + self.intent_dim, trans_d_model)
            self.sur_intent_ce_head = nn.Linear(self.intent_dim, num_intents)

        # 6. Output heads: d_model → (acc, yaw_rate)
        self.ego_output_head = nn.Linear(trans_d_model, self.traj_out_size)
        self.sur_output_head = nn.Linear(trans_d_model, self.traj_out_size)

        # 7. Auxiliary loss heads (Linear — force upstream quality)
        self.sur_pred_head = nn.Linear(trans_d_model, self.sur_pred_dim)  # ego→sur delta
        self.ego_pred_head = nn.Linear(trans_d_model, self.sur_pred_dim)  # sur→ego delta
        self.intent_ce_head = nn.Linear(self.intent_dim, self.num_intents)  # ego intent CE

        # 8. z_global auxiliary decoder: z + step_embed -> action prediction (training only)
        self.z_aux_step_embed = nn.Embedding(self.FT, self.z_size)  # 12 steps -> 32-dim
        self.z_aux_decoder = nn.Sequential(
            nn.Linear(self.z_size * 2, 128),  # z(32) + step_embed(32) = 64 -> 128
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.traj_out_size),  # -> 2 (acc, ddh)
        )

        # Phase control
        self.phase = 1
        self.gumbel_temperature = 1.0
        # use_ego_z_local / use_sur_z_local are set above


    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def get_normalizer(self):
        return self.normalizer

    def set_att_normalizer(self, normalizer):
        self.att_normalizer = normalizer

    def get_att_normalizer(self):
        return self.att_normalizer

    def set_bicycle_params(self, bicycle_params):
        self.bicycle_params = bicycle_params

    # ============================================================
    # Forward / Reconstruct / Sample — public API (unchanged interface)
    # ============================================================

    def forward(self, scene_graph, map_idx, map_env,
                use_post_mean=False,
                future_sample=False,
                teacher_forcing=False,
                current_epoch=0,
                blend_alpha=0.0):
        # Reset per-step pooled raster (will be set by _recompute_map_tokens_batched if map_recrop)
        self._map_raster_pooled_steps = None

        # Map encoding (encoder-level 64dim + decoder-level tokens)
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        # PRIOR
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)

        # POSTERIOR
        past_context = past_seq_out[:, -1, :]
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_context)

        # DECODER (Transformer-based)
        if use_post_mean:
            z_samp = post_mu
        else:
            z_samp = self.rsample(post_mu, post_var)

        # Store health monitoring intermediates
        self._last_z_samp = z_samp.detach()
        self._last_prior_out = (prior_mu.detach(), prior_var.detach())
        self._last_posterior_out = (post_mu.detach(), post_var.detach())

        # z_global auxiliary decoder: predict actions from z alone
        self._z_aux_traj = None
        if hasattr(self, 'z_aux_decoder'):
            ego_mask_aux = torch.zeros(z_samp.size(0), dtype=torch.bool, device=z_samp.device)
            ego_mask_aux[scene_graph.ptr[:-1]] = True
            z_ego = z_samp[ego_mask_aux]  # (N_ego, 32)

            step_ids = torch.arange(self.FT, device=z_samp.device)
            step_emb = self.z_aux_step_embed(step_ids)  # (FT, 32)

            z_exp = z_ego.unsqueeze(1).expand(-1, self.FT, -1)  # (N_ego, FT, 32)
            s_exp = step_emb.unsqueeze(0).expand(z_ego.size(0), -1, -1)  # (N_ego, FT, 32)
            z_input = torch.cat([z_exp, s_exp], dim=-1)  # (N_ego, FT, 64)
            self._z_aux_traj = self.z_aux_decoder(z_input)  # (N_ego, FT, 2)

        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp, map_idx, map_env,
                                   map_tokens=map_tokens,
                                   teacher_forcing=teacher_forcing,
                                   blend_alpha=blend_alpha)

        net_out = {
            'prior_out': (prior_mu, prior_var),
            'posterior_out': (post_mu, post_var),
            'future_pred': future_pred,
            'past_seq_out': past_seq_out,
        }

        if future_sample:
            prior_samp = self.rsample(prior_mu, prior_var)
            future_samp = self.decoder(scene_graph, map_feat, past_seq_out, prior_samp, map_idx, map_env,
                                       map_tokens=map_tokens)
            net_out['future_samp'] = future_samp

        return net_out

    def compute_health_metrics(self):
        """Compute module health metrics for TensorBoard logging."""
        metrics = {}

        # z_global health
        if hasattr(self, '_last_z_samp') and self._last_z_samp is not None:
            z = self._last_z_samp
            metrics['z_global/norm'] = z.norm(dim=-1).mean().item()
            metrics['z_global/std'] = z.std(dim=0).mean().item()
            # Per-dim std: check if any dims are dead (std ≈ 0)
            per_dim_std = z.std(dim=0)  # (z_size,)
            metrics['z_global/min_dim_std'] = per_dim_std.min().item()
            metrics['z_global/active_dims'] = (per_dim_std > 0.01).sum().item()

            # Per-dim KL
            pm, pv = self._last_prior_out
            qm, qv = self._last_posterior_out
            kl_per_dim = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm) ** 2 / pv - 1)
            metrics['z_global/kl_per_dim_mean'] = kl_per_dim.mean().item()
            metrics['z_global/kl_per_dim_max'] = kl_per_dim.max().item()

            # Prior/posterior gap: how far posterior moves from prior
            metrics['z_global/prior_post_mean_gap'] = (qm - pm).abs().mean().item()
            metrics['z_global/posterior_var_mean'] = qv.mean().item()

        # Intent health
        if hasattr(self, '_intent_weights_outputs') and isinstance(self._intent_weights_outputs, torch.Tensor):
            iw = self._intent_weights_outputs
            probs = iw.softmax(dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean()
            metrics['intent/entropy'] = entropy.item()
            metrics['intent/max_prob'] = probs.max(dim=-1)[0].mean().item()
            metrics['intent/unique_slots'] = probs.max(dim=-1)[1].unique().numel()
            # Per-slot usage frequency
            slot_usage = probs.argmax(dim=-1).float()
            for k in range(self.num_intents):
                metrics[f'intent/slot_{k}_frac'] = (slot_usage == k).float().mean().item()

        # Map attention health
        if hasattr(self, '_ego_map_attn_weights_outputs') and self._ego_map_attn_weights_outputs is not None:
            mw = self._ego_map_attn_weights_outputs
            if isinstance(mw, torch.Tensor) and mw.numel() > 0:
                entropy = -(mw * (mw + 1e-8).log()).sum(-1).mean()
                metrics['map_attn/entropy'] = entropy.item()
                metrics['map_attn/peak'] = mw.max(dim=-1)[0].mean().item()
                # Top-5 concentration: what fraction of attention is in top 5 tokens
                if mw.size(-1) > 5:
                    top5, _ = mw.topk(5, dim=-1)
                    metrics['map_attn/top5_mass'] = top5.sum(dim=-1).mean().item()

        # z_aux decoder health (training only)
        if hasattr(self, '_z_aux_traj') and self._z_aux_traj is not None:
            z_aux = self._z_aux_traj.detach()
            metrics['z_aux/pred_acc_std'] = z_aux[:, :, 0].std().item()
            metrics['z_aux/pred_ddh_std'] = z_aux[:, :, 1].std().item()

        return metrics

    def compute_per_module_grad_norms(self):
        """Compute gradient norms per module for TensorBoard logging."""
        norms = {}
        module_groups = {
            'encoder_gcn': [self.temporal_gcn_encoder, self.step_feature_extractor],
            'prior': [self.latent_prior_net, self.prior_temporal_gru],
            'posterior': [self.latent_posterior_net, self.transformer_encoder],
            'interaction_gcn': [self.interaction_gcn],
            'query_proj': [self.query_proj],
            'context_encoder': [self.context_encoder, self.map_summary_pooling],
            'trans_layers': list(self.trans_layers),
            'output_heads': [self.ego_output_head, self.sur_output_head],
            'intent': [self.intent_codebook, self.ego_intent_proj],
            'map_cnn': [self.map_conv_early, self.map_conv_late],
        }
        if hasattr(self, 'z_aux_decoder'):
            module_groups['z_aux'] = [self.z_aux_decoder, self.z_aux_step_embed]

        for name, modules in module_groups.items():
            total = 0.0
            count = 0
            for m in modules:
                for p in m.parameters():
                    if p.grad is not None:
                        total += p.grad.data.norm(2).item() ** 2
                        count += 1
            norms[name] = total ** 0.5 if count > 0 else 0.0
        return norms

    def reconstruct(self, scene_graph, map_idx, map_env, sur_gt_replay=False):
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        past_context = past_seq_out[:, -1, :]
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_context)

        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, post_mu, map_idx, map_env,
                                   map_tokens=map_tokens,
                                   sur_gt_replay=sur_gt_replay)

        return {
            'prior_out': (prior_mu, prior_var),
            'posterior_out': (post_mu, post_var),
            'future_pred': future_pred,
        }

    def sample(self, scene_graph, map_idx, map_env, num_samples,
               include_mean=False, nfuture=None, sur_gt_replay=False):
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        prior_distrib = Normal(prior_mu, torch.sqrt(prior_var))

        net_out = {
            'prior_out': (prior_mu, prior_var),
            'z_samp': [],
            'z_logprob': [],
            'z_mdist': [],
            'future_pred': [],
        }
        for sidx in range(num_samples):
            if include_mean and sidx == (num_samples - 1):
                z_samp = prior_mu
            else:
                z_samp = self.rsample(prior_mu, prior_var)
            z_logprob = prior_distrib.log_prob(z_samp).sum(dim=-1)
            z_mdist = torch.norm((z_samp - prior_mu) / torch.sqrt(prior_var), dim=-1)
            future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp, map_idx, map_env,
                                       map_tokens=map_tokens,
                                       nfuture=nfuture, sur_gt_replay=sur_gt_replay)
            net_out['z_samp'].append(z_samp)
            net_out['z_logprob'].append(z_logprob)
            net_out['z_mdist'].append(z_mdist)
            net_out['future_pred'].append(future_pred)

        net_out['z_samp'] = torch.stack(net_out['z_samp'], dim=1)
        net_out['z_logprob'] = torch.stack(net_out['z_logprob'], dim=1)
        net_out['z_mdist'] = torch.stack(net_out['z_mdist'], dim=1)
        net_out['future_pred'] = torch.stack(net_out['future_pred'], dim=1)
        return net_out

    def sample_batched(self, scene_graph, map_idx, map_env, num_samples,
                       include_mean=False, nfuture=None, sur_gt_replay=False):
        NA = scene_graph.past.size(0)
        NS = num_samples
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)

        samp_mu = prior_mu.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        samp_var = prior_var.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        prior_distrib = Normal(samp_mu, torch.sqrt(samp_var))

        z_samp = self.rsample(samp_mu, samp_var)
        if include_mean:
            z_samp[-1, :, :] = prior_mu
        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp.transpose(0, 1), map_idx, map_env,
                                   map_tokens=map_tokens,
                                   nfuture=nfuture, sur_gt_replay=sur_gt_replay)
        net_out = {
            'prior_out': (prior_mu, prior_var),
            'z_samp': z_samp.view(NS, NA, self.z_size).transpose(0, 1),
            'future_pred': future_pred,
        }
        z_logprob = prior_distrib.log_prob(z_samp.view(NS, NA, self.z_size)).sum(dim=-1).transpose(0, 1)
        z_mdist = torch.norm((z_samp.view(NS, NA, self.z_size) - samp_mu) / torch.sqrt(samp_var), dim=-1).transpose(0, 1)
        net_out['z_logprob'] = z_logprob
        net_out['z_mdist'] = z_mdist
        return net_out

    def embed(self, scene_graph, map_idx, map_env):
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        embed_out = {
            'prior_out': (prior_mu, prior_var),
            'map_feat': map_feat,
            'map_tokens': map_tokens,
            'past_seq_out': past_seq_out,
        }
        if 'future' in scene_graph:
            past_context = past_seq_out[:, -1, :]
            post_mu, post_var = self.encoder(scene_graph, map_feat, past_context)
            embed_out['posterior_out'] = (post_mu, post_var)
        return embed_out

    def decode_embedding(self, z, embed_out, scene_graph, map_idx, map_env,
                         ext_future=None, nfuture=None):
        future_pred = self.decoder(scene_graph, embed_out['map_feat'], embed_out['past_seq_out'],
                                   z, map_idx, map_env,
                                   map_tokens=embed_out.get('map_tokens'),
                                   ext_future=ext_future, nfuture=nfuture)
        return {'future_pred': future_pred}

    # ============================================================
    # Encoder methods — UNCHANGED
    # ============================================================

    def _run_temporal_encoder(self, scene_graph, traj_data, vis_data, T, use_transformer=True):
        g_in_data = scene_graph
        ego_mask = self._get_ego_mask(scene_graph)
        all_gcn_features = []

        # Transform to local frame of last past step (same as original STRIVE encode_past/future).
        # This ensures MLP input has relative displacements (~0.2) instead of
        # absolute coordinates (~30), keeping all feature dimensions balanced.
        ref_frame = scene_graph.past[:, -1, :4]  # (NA, 4) normalized global
        local_kin = transform2frame(ref_frame, traj_data[:, :, :4])  # (NA, T, 4)
        local_traj = torch.cat([local_kin, traj_data[:, :, 4:]], dim=2)  # (NA, T, 6)

        for t in range(T):
            cur_state_local = local_traj[:, t, :]  # (NA, 6) in local frame
            cur_vis = vis_data[:, t].unsqueeze(-1)
            cur_lw = scene_graph.lw
            cur_sem = scene_graph.sem

            step_in_feat = torch.cat([cur_state_local, cur_lw, cur_vis, cur_sem], dim=-1)
            gcn_node_in = self.step_feature_extractor(step_in_feat)

            g_in_data.x = gcn_node_in
            g_in_data.pos = traj_data[:, t, :4]  # GCN edges use global coords (relative transform inside GCN)

            ego_feat_t, other_feat_t = self.temporal_gcn_encoder(g_in_data, ego_mask)
            gcn_feat_t = self._merge_ego_other_feat(ego_feat_t, other_feat_t, ego_mask)
            all_gcn_features.append(gcn_feat_t)

        sequence_features = torch.stack(all_gcn_features, dim=1)

        if use_transformer:
            sequence_features = self.positional_encoding(sequence_features)
            padding_mask = (vis_data == 0)
            transformer_out = self.transformer_encoder(sequence_features, src_key_padding_mask=padding_mask)
            context_vector = transformer_out[:, -1, :]
            return transformer_out, context_vector
        else:
            gru_out, _ = self.prior_temporal_gru(sequence_features)
            context_vector = gru_out[:, -1, :]
            return gru_out, context_vector

    def encode_map(self, scene_graph, map_idx, map_env, return_tokens=False):
        """
        Encode map features. If return_tokens=True, also return spatial tokens from conv3.

        :return: map_feat (NA, 64) or (map_feat, map_tokens) if return_tokens=True
                 map_tokens: (NA, num_tokens, map_token_ch)
        """
        NA = scene_graph.pos.size(0)
        NS = None if len(scene_graph.pos.size()) != 3 else scene_graph.pos.size(1)

        normalize_scene_graph(scene_graph, self.normalizer, self.att_normalizer, unnorm=True)
        map_obs = map_env.get_map_crop(scene_graph, map_idx).to(torch.float)  # NA x C x H x W
        normalize_scene_graph(scene_graph, self.normalizer, self.att_normalizer, unnorm=False)

        bsize = NA if NS is None else NA * NS
        if NS is not None:
            map_obs = map_obs.reshape(bsize, map_obs.size(-3), map_obs.size(-2), map_obs.size(-1))

        # Pool raw raster to token grid for potential-weighted soft labels
        # map_obs: (bsize, C, 256, 256) → (bsize, C, 29, 29)
        with torch.no_grad():
            self._map_raster_pooled = nn.functional.adaptive_avg_pool2d(
                map_obs, self.map_token_spatial)  # (bsize, C, 29, 29)

        # Early conv (conv1-3): spatial tokens
        map_early = self.map_conv_early(map_obs)  # (bsize, ch, H', W')

        # Late conv (conv4-6): encoder feature
        map_late = self.map_conv_late(map_early)
        map_feat = self.map_feature(map_late.reshape(bsize, self.map_feat_in_size))

        if NS is not None:
            map_feat = map_feat.reshape(NA, NS, -1)

        if return_tokens:
            # Flatten spatial dims: (bsize, ch, H', W') → (bsize, H'*W', ch)
            map_tokens = map_early.flatten(2).permute(0, 2, 1)  # (bsize, num_tokens, ch)
            if NS is not None:
                map_tokens = map_tokens.reshape(NA, NS, self.map_num_tokens, self.map_token_ch)
            return map_feat, map_tokens
        return map_feat

    def encoder(self, scene_graph, map_feat, past_context):
        """Posterior: future → z_global."""
        _, future_context = self._run_temporal_encoder(
            scene_graph, scene_graph.future, scene_graph.future_vis, self.FT, use_transformer=True
        )
        posterior_in = torch.cat([past_context, future_context, map_feat, scene_graph.sem], dim=-1)
        posterior_z = self.latent_posterior_net(posterior_in)
        mean, logvar = posterior_z[:, :self.z_size], posterior_z[:, self.z_size:]
        var = torch.exp(logvar)
        return mean, var

    def prior(self, scene_graph, map_feat):
        """Prior: past → z_global."""
        past_seq_out, past_context = self._run_temporal_encoder(
            scene_graph, scene_graph.past, scene_graph.past_vis, self.PT, use_transformer=False
        )
        prior_in = torch.cat([past_context, map_feat, scene_graph.sem], dim=-1)
        prior_z = self.latent_prior_net(prior_in)
        mean, logvar = prior_z[:, :self.z_size], prior_z[:, self.z_size:]
        var = torch.exp(logvar)
        return mean, var, past_seq_out

    # ============================================================
    # Decoder (Transformer-based)
    # ============================================================

    def _run_interaction_gcn_parallel(self, states, scene_graph, ego_mask, T):
        """
        Run interaction GCN for T timesteps in parallel (using GT states).

        :param states: (NA, T, state_dim) — states for each timestep (global frame)
        :param scene_graph: for lw, sem, batch structure
        :param ego_mask: (NA,) bool
        :param T: number of timesteps
        :return: gcn_feats_merged (NA, T, gcn_hidden_dim)
        """
        NA = states.size(0)
        cur_lw = scene_graph.lw
        cur_sem = scene_graph.sem

        # Transform to local frame of last past step for node features (same as encoder).
        # GCN edge features use global coords internally via transform2frame.
        ref_frame = scene_graph.past[:, -1, :4]  # (NA, 4)
        local_kin = transform2frame(ref_frame, states[:, :, :4])  # (NA, T, 4)
        local_states = torch.cat([local_kin, states[:, :, 4:]], dim=2)  # (NA, T, 6)

        all_gcn_features = []
        for t in range(T):
            cur_state_global = states[:, t, :]
            cur_state_6d_global = self._get_6d_state(cur_state_global, False, NA, None)
            cur_state_local = local_states[:, t, :]
            cur_state_6d_local = self._get_6d_state(cur_state_local, False, NA, None)

            # Node features use local coords for balanced MLP input scale
            gcn_in = torch.cat([cur_state_6d_local, cur_lw, cur_sem], dim=-1)
            scene_graph.x = gcn_in
            # Edge features use global coords (transform2frame inside GCN)
            scene_graph.pos = cur_state_6d_global[:, :4]

            ego_feat_t, sur_feat_t = self.interaction_gcn(scene_graph, ego_mask)
            merged_t = self._merge_ego_other_feat(ego_feat_t, sur_feat_t, ego_mask)
            all_gcn_features.append(merged_t)

        return torch.stack(all_gcn_features, dim=1)  # (NA, T, gcn_hidden_dim)

    def decoder(self, scene_graph, map_feat, past_seq_out, z, map_idx, map_env,
                map_tokens=None,
                ext_future=None, nfuture=None,
                teacher_forcing=False,
                sur_gt_replay=False,
                blend_alpha=0.0):
        """
        Decoder dispatcher (Enc-Dec Cross-Attention).
        Training: AR loop with GT action interpolation (blend_alpha ∈ [0,1])
        Inference: autoregressive (teacher_forcing=False)
        """
        if map_tokens is None:
            scene_graph_pos_backup = scene_graph.pos.clone()
            scene_graph.pos = scene_graph.past[:, -1, :4]
            _, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)
            scene_graph.pos = scene_graph_pos_backup

        if teacher_forcing:
            return self.transformer_decoder_training_blended(
                scene_graph, z, map_tokens, map_idx, map_env, blend_alpha)
        else:
            return self.transformer_decoder_inference(
                scene_graph, z, map_tokens, map_idx, map_env,
                ext_future=ext_future, nfuture=nfuture,
                sur_gt_replay=sur_gt_replay)

    def transformer_decoder_training_blended(self, scene_graph, z, map_tokens, map_idx, map_env,
                                              blend_alpha=0.5):
        """
        Enc-Dec Cross-Attention decoder — blended training mode.

        Architecture:
          1. Context = ContextEncoder(past_gcn + z_global + map_summary) — computed once
          2. AR loop (12 steps):
             a. GCN(current state) → interaction_feat
             b. query = query_proj(gcn ⊕ z_local ⊕ lw ⊕ sem) + step_PE(t)
             c. TransDecoderLayer ×2: A2A → A2C(context) → A2S(map) → FFN
             d. output_head → action → blend with GT → bicycle model

        :param blend_alpha: interpolation weight (0=pure GT, 1=pure predicted)
        :return: traj_out (NA, FT, 4)
        """
        NA = z.size(0)
        FT = self.FT
        PT = self.PT
        device = z.device

        ego_mask = self._get_ego_mask(scene_graph)
        num_ego = int(ego_mask.sum())

        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)

        # Precompute GT actions for blending
        gt_actions_all = self._compute_gt_actions(scene_graph, ego_only=False)  # (NA, FT, 2)

        # Initialize analysis storage
        self._z_local_outputs = []
        self._intent_weights_outputs = []
        self._ego_map_attn_weights_outputs = []
        self._sur_map_attn_weights_outputs = []
        self._sur_pred_outputs = []
        self._ego_pred_outputs = []

        # ================================================================
        # 1. Build Context (once, before AR loop)
        # ================================================================
        gt_past = scene_graph.past  # (NA, PT, state_dim)
        past_gcn_feats = self._run_interaction_gcn_parallel(
            gt_past, scene_graph, ego_mask, PT)  # (NA, PT, 64)

        # Map summary: 3249 tokens → M summary tokens
        map_summary = self.map_summary_pooling(map_tokens)  # (NA, M, d_model)

        # Context encoder: past + z + map_summary → context tokens
        context = self.context_encoder(past_gcn_feats, z, map_summary)
        # context: (NA, PT+K+M, d_model) — fixed across all AR steps

        # Store for health metrics
        self._last_context = context.detach()

        # ================================================================
        # 2. AR loop
        # ================================================================
        prev_state = scene_graph.past[:, -1, :]  # (NA, state_dim)
        traj_out = torch.zeros(NA, FT, 4, device=device)

        for t in range(FT):
            # --- GCN for current state ---
            cur_state_6d = self._get_6d_state(prev_state, False, NA, None)

            # Local frame for GCN node features
            ref_frame = scene_graph.past[:, -1, :4]
            local_kin = transform2frame(ref_frame, prev_state[:, :4].unsqueeze(1))[:, 0]
            local_state = torch.cat([local_kin, prev_state[:, 4:]], dim=1)
            local_6d = self._get_6d_state(local_state, False, NA, None)

            gcn_in = torch.cat([local_6d, scene_graph.lw, scene_graph.sem], dim=-1)
            scene_graph.x = gcn_in
            scene_graph.pos = cur_state_6d[:, :4]  # global for edge transform

            ego_gcn, sur_gcn = self.interaction_gcn(scene_graph, ego_mask)
            gcn_feat = self._merge_ego_other_feat(ego_gcn, sur_gcn, ego_mask)
            # gcn_feat: (NA, 64)

            # --- Build query token ---
            query_input = torch.cat([gcn_feat, scene_graph.lw, scene_graph.sem], dim=-1)
            query = self.query_proj(query_input)  # (NA, d_model)
            query = query + self.step_pe(torch.tensor(t, device=device))  # step PE
            # Reshape: (1, 1, NA, D) for TransDecoderLayer
            x = query.unsqueeze(0).unsqueeze(0)

            # --- Map tokens (recrop if needed) ---
            if self.map_recrop and t > 0:
                cur_map = self._recompute_map_tokens(
                    prev_state[:, :4], map_idx, map_env, scene_graph)
            else:
                cur_map = map_tokens

            # --- Agent states for A2A relative bias ---
            agent_states_for_bias = None
            if self.use_a2a_rel_bias:
                agent_states_for_bias = cur_state_6d.unsqueeze(0)  # (1, NA, 6)

            # --- Run through Transformer decoder layers ---
            for layer_idx, layer in enumerate(self.trans_layers):
                is_first_layer = (layer_idx == 0)
                x_out, extras = layer(
                    x, ego_mask, context, cur_map,
                    return_a2a_output=is_first_layer,
                    return_a2s_weights=is_first_layer,
                    agent_states=agent_states_for_bias,
                )
                x = x_out

                # Intent between Layer 0 and Layer 1
                if is_first_layer:
                    # Collect map attn weights
                    if 'ego_map_attn_weights' in extras and extras['ego_map_attn_weights'] is not None:
                        self._ego_map_attn_weights_outputs.append(extras['ego_map_attn_weights'])
                    if 'sur_map_attn_weights' in extras and extras['sur_map_attn_weights'] is not None:
                        self._sur_map_attn_weights_outputs.append(extras['sur_map_attn_weights'])

                    # A2A aux outputs
                    if 'a2a_output' in extras:
                        a2a_out = extras['a2a_output']  # (1, 1, NA, D)
                        a2a_single = a2a_out[:, 0, :, :]  # (1, NA, D)
                        self._sur_pred_outputs.append(
                            self.sur_pred_head(a2a_single[:, ego_mask, :]).squeeze(0))
                        self._ego_pred_outputs.append(
                            self.ego_pred_head(a2a_single[:, ~ego_mask, :]).squeeze(0))

                    # Intent codebook (ego)
                    ego_token = x[:, 0, ego_mask, :]  # (1, num_ego, D)
                    ego_flat = ego_token.reshape(-1, self.trans_d_model)

                    z_local, intent_w = self.intent_codebook(
                        ego_flat, temperature=self.gumbel_temperature)
                    if not self.use_ego_z_local:
                        z_local = torch.zeros_like(z_local)

                    self._z_local_outputs.append(z_local.detach())
                    self._intent_weights_outputs.append(intent_w.detach())

                    ego_with_z = torch.cat([ego_flat, z_local], dim=-1)
                    ego_proj = self.ego_intent_proj(ego_with_z)
                    x = x.clone()
                    x[:, 0, ego_mask, :] = ego_proj.view(1, -1, self.trans_d_model)

                    if self.use_sur_z_local:
                        sur_token = x[:, 0, ~ego_mask, :]
                        sur_flat = sur_token.reshape(-1, self.trans_d_model)
                        sur_z_local, _ = self.sur_intent_codebook(
                            sur_flat, temperature=self.gumbel_temperature)
                        sur_with_z = torch.cat([sur_flat, sur_z_local], dim=-1)
                        sur_proj = self.sur_intent_proj(sur_with_z)
                        x[:, 0, ~ego_mask, :] = sur_proj.view(1, -1, self.trans_d_model)

            # --- Output head → action → blend → bicycle model ---
            last_tokens = x[:, 0, :, :]  # (1, NA, D)

            ego_last = last_tokens[:, ego_mask, :]
            sur_last = last_tokens[:, ~ego_mask, :]
            ego_out = self.ego_output_head(ego_last).squeeze(0)
            sur_out = self.sur_output_head(sur_last).squeeze(0)

            pred_action = torch.zeros(NA, self.traj_out_size, device=device)
            pred_action[ego_mask] = ego_out
            pred_action[~ego_mask] = sur_out

            # Blend with GT action
            gt_action_t = gt_actions_all[:, t, :]
            blended_action = (1.0 - blend_alpha) * gt_action_t + blend_alpha * pred_action

            # Bicycle model
            cur_state_global, _, cur_bike_state = self._apply_dynamics(
                blended_action, prev_state, cur_veh_len, NA, None, False)

            traj_out[:, t, :] = cur_state_global

            # Update prev_state (gradient flows through!)
            if self.output_bicycle and cur_bike_state is not None:
                prev_state = cur_bike_state
            else:
                prev_state = cur_state_global

        # Stack collected outputs
        if self._ego_map_attn_weights_outputs:
            self._ego_map_attn_weights_outputs = torch.cat(
                self._ego_map_attn_weights_outputs, dim=0)
        if self._sur_map_attn_weights_outputs:
            self._sur_map_attn_weights_outputs = torch.cat(
                self._sur_map_attn_weights_outputs, dim=0)

        if self._z_local_outputs:
            self._z_local_outputs = torch.stack(self._z_local_outputs, dim=0)
        if self._intent_weights_outputs:
            self._intent_weights_outputs = torch.stack(self._intent_weights_outputs, dim=0)

        return traj_out

    def transformer_decoder_inference(self, scene_graph, z, map_tokens, map_idx, map_env,
                                       ext_future=None, nfuture=None, sur_gt_replay=False):
        """
        Enc-Dec Cross-Attention decoder — inference mode (autoregressive).

        Same architecture as training: context + single-query AR loop.
        Supports multi-sample (z: NA, NS, z_size).

        :return: traj_out (NA, FT, 4) or (NA, NS, FT, 4) for multi-sample
        """
        NA = z.size(0)
        FT = self.FT if nfuture is None else nfuture
        PT = self.PT
        device = z.device

        ego_inds = scene_graph.ptr[:-1]
        ego_mask = self._get_ego_mask(scene_graph)
        num_ego = int(ego_mask.sum())
        B = map_idx.size(0)

        # Multi-sample check
        mult_samp = z.dim() == 3
        NS = z.size(1) if mult_samp else None

        # Vehicle lengths
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)

        # Initialize analysis storage
        self._z_local_outputs = []
        self._intent_weights_outputs = []
        self._ego_map_attn_weights_outputs = []
        self._sur_map_attn_weights_outputs = []
        self._sur_pred_outputs = []
        self._ego_pred_outputs = []

        if mult_samp:
            cur_veh_len = cur_veh_len.unsqueeze(1).expand(NA, NS, 1).reshape(NA * NS, 1)
            z_flat = z.reshape(NA * NS, self.z_size)
            map_tokens_flat = map_tokens.unsqueeze(1).expand(-1, NS, -1, -1).reshape(
                NA * NS, self.map_num_tokens, self.map_token_ch)
        else:
            z_flat = z
            map_tokens_flat = map_tokens

        NA_eff = NA * NS if mult_samp else NA

        # ================================================================
        # 1. Build Context (once)
        # ================================================================
        gt_past = scene_graph.past  # (NA, PT, state_dim)
        past_gcn_feats = self._run_interaction_gcn_parallel(
            gt_past, scene_graph, ego_mask, PT)  # (NA, PT, 64)

        if mult_samp:
            past_gcn_feats_exp = past_gcn_feats.unsqueeze(1).expand(-1, NS, -1, -1).reshape(
                NA_eff, PT, self.gcn_hidden_dim)
            map_summary = self.map_summary_pooling(map_tokens_flat)  # (NA_eff, M, d_model)
            context = self.context_encoder(past_gcn_feats_exp, z_flat, map_summary)
        else:
            map_summary = self.map_summary_pooling(map_tokens)  # (NA, M, d_model)
            context = self.context_encoder(past_gcn_feats, z_flat, map_summary)
        # context: (NA_eff, C, d_model)

        # ================================================================
        # 2. AR loop
        # ================================================================
        prev_state = scene_graph.past[:, -1, :]  # (NA, state_dim)
        if mult_samp:
            prev_state = prev_state.unsqueeze(1).expand(NA, NS, -1).reshape(NA_eff, -1)
            if ext_future is not None:
                ext_future = ext_future.unsqueeze(1).expand(NA, NS, -1, 4).reshape(NA_eff, -1, 4)

        traj_out = torch.zeros(NA_eff, FT, 4, device=device)
        cur_ego = ego_mask if not mult_samp else self._expand_ego_mask(ego_mask, NS)

        for t in range(FT):
            # --- GCN for current state ---
            cur_state_6d = self._get_6d_state(prev_state, False, NA_eff, None)

            ref_frame = scene_graph.past[:, -1, :4]  # (NA, 4)
            if mult_samp:
                ref_frame_exp = ref_frame.unsqueeze(1).expand(-1, NS, -1).reshape(NA_eff, 4)
            else:
                ref_frame_exp = ref_frame
            local_kin = transform2frame(ref_frame_exp, prev_state[:, :4].unsqueeze(1))[:, 0]
            local_state = torch.cat([local_kin, prev_state[:, 4:]], dim=1)
            local_6d = self._get_6d_state(local_state, False, NA_eff, None)

            if mult_samp:
                local_for_gcn = local_6d.reshape(NA, NS, -1)
                scene_graph.x = torch.cat([local_for_gcn,
                                            scene_graph.lw.unsqueeze(1).expand(-1, NS, -1),
                                            scene_graph.sem.unsqueeze(1).expand(-1, NS, -1)], dim=-1)
                scene_graph.pos = cur_state_6d.reshape(NA, NS, -1)[..., :4]
            else:
                gcn_in = torch.cat([local_6d, scene_graph.lw, scene_graph.sem], dim=-1)
                scene_graph.x = gcn_in
                scene_graph.pos = cur_state_6d[:, :4]

            ego_gcn, sur_gcn = self.interaction_gcn(scene_graph, ego_mask)
            gcn_feat = self._merge_ego_other_feat(ego_gcn, sur_gcn, ego_mask)
            if mult_samp:
                gcn_feat = gcn_feat.reshape(NA_eff, self.gcn_hidden_dim)

            # --- Build query token ---
            if mult_samp:
                lw_exp = scene_graph.lw.unsqueeze(1).expand(-1, NS, -1).reshape(NA_eff, -1)
                sem_exp = scene_graph.sem.unsqueeze(1).expand(-1, NS, -1).reshape(NA_eff, -1)
            else:
                lw_exp = scene_graph.lw
                sem_exp = scene_graph.sem

            query_input = torch.cat([gcn_feat, lw_exp, sem_exp], dim=-1)
            query = self.query_proj(query_input) + self.step_pe(torch.tensor(t, device=device))
            x = query.unsqueeze(0).unsqueeze(0)  # (1, 1, NA_eff, D)

            # --- Map tokens ---
            if self.map_recrop and t > 0:
                cur_map = self._recompute_map_tokens(
                    prev_state[:, :4], map_idx, map_env, scene_graph,
                    mult_samp=mult_samp, NS=NS)
            else:
                cur_map = map_tokens_flat

            # --- A2A bias ---
            agent_states_for_bias = None
            if self.use_a2a_rel_bias:
                agent_states_for_bias = cur_state_6d.unsqueeze(0)  # (1, NA_eff, 6)

            # --- Transformer decoder layers ---
            for layer_idx, layer in enumerate(self.trans_layers):
                is_first_layer = (layer_idx == 0)
                x_out, extras = layer(
                    x, cur_ego, context, cur_map,
                    return_a2a_output=is_first_layer,
                    return_a2s_weights=is_first_layer,
                    agent_states=agent_states_for_bias,
                )
                x = x_out

                # Intent between Layer 0 and Layer 1
                if is_first_layer:
                    if 'ego_map_attn_weights' in extras and extras['ego_map_attn_weights'] is not None:
                        self._ego_map_attn_weights_outputs.append(extras['ego_map_attn_weights'])
                    if 'sur_map_attn_weights' in extras and extras['sur_map_attn_weights'] is not None:
                        self._sur_map_attn_weights_outputs.append(extras['sur_map_attn_weights'])

                    # A2A aux outputs (same as training: collect from A2A output)
                    if 'a2a_output' in extras:
                        a2a_out = extras['a2a_output']  # (1, 1, NA_eff, D)
                        a2a_single = a2a_out[:, 0, :, :]  # (1, NA_eff, D)
                        self._sur_pred_outputs.append(
                            self.sur_pred_head(a2a_single[:, cur_ego, :]).squeeze(0))
                        self._ego_pred_outputs.append(
                            self.ego_pred_head(a2a_single[:, ~cur_ego, :]).squeeze(0))

                    ego_token = x[:, 0, cur_ego, :]
                    ego_flat = ego_token.reshape(-1, self.trans_d_model)

                    z_local, intent_w = self.intent_codebook(
                        ego_flat, temperature=self.gumbel_temperature)
                    if not self.use_ego_z_local:
                        z_local = torch.zeros_like(z_local)

                    self._z_local_outputs.append(z_local.detach())
                    self._intent_weights_outputs.append(intent_w.detach())

                    ego_with_z = torch.cat([ego_flat, z_local], dim=-1)
                    ego_proj = self.ego_intent_proj(ego_with_z)
                    x = x.clone()
                    x[:, 0, cur_ego, :] = ego_proj.view(1, -1, self.trans_d_model)

                    if self.use_sur_z_local:
                        sur_token = x[:, 0, ~cur_ego, :]
                        sur_flat = sur_token.reshape(-1, self.trans_d_model)
                        sur_z_local, _ = self.sur_intent_codebook(
                            sur_flat, temperature=self.gumbel_temperature)
                        sur_with_z = torch.cat([sur_flat, sur_z_local], dim=-1)
                        sur_proj = self.sur_intent_proj(sur_with_z)
                        x[:, 0, ~cur_ego, :] = sur_proj.view(1, -1, self.trans_d_model)

            # --- Output head → bicycle model ---
            last_tokens = x[:, 0, :, :]  # (1, NA_eff, D)

            ego_last = last_tokens[:, cur_ego, :]
            sur_last = last_tokens[:, ~cur_ego, :]

            ego_out = self.ego_output_head(ego_last).squeeze(0)
            sur_out = self.sur_output_head(sur_last).squeeze(0)

            decoder_out = torch.zeros(NA_eff, self.traj_out_size, device=device)
            decoder_out[cur_ego] = ego_out
            decoder_out[~cur_ego] = sur_out

            cur_state_global, cur_state_local, cur_bike_state = self._apply_dynamics(
                decoder_out, prev_state, cur_veh_len, NA_eff, None, False)

            # Handle external future
            if ext_future is not None:
                cur_state_global = cur_state_global.clone()
                if mult_samp:
                    ext_ego_inds = ego_inds.unsqueeze(1).expand(B, NS).reshape(B * NS)
                    cur_state_global[ext_ego_inds] = ext_future[:, t]
                else:
                    cur_state_global[ego_inds] = ext_future[:, t]

            # Handle sur GT replay
            if sur_gt_replay and hasattr(scene_graph, 'future_gt'):
                num_sur = NA_eff - int(cur_ego.sum())
                cur_state_global, cur_state_local, cur_bike_state = self._apply_sur_gt_replay(
                    scene_graph, cur_state_global, cur_state_local, cur_bike_state,
                    prev_state, cur_ego, t, False, NA_eff, None, num_sur)

            traj_out[:, t, :] = cur_state_global

            if self.output_bicycle and cur_bike_state is not None:
                prev_state = cur_bike_state
            else:
                prev_state = cur_state_global

        if mult_samp:
            traj_out = traj_out.reshape(NA, NS, FT, 4)
        return traj_out

    def _expand_ego_mask(self, ego_mask, NS):
        """Expand ego_mask (NA,) → (NA*NS,) for multi-sample."""
        return ego_mask.unsqueeze(1).expand(-1, NS).reshape(-1)

    # ============================================================
    # GT action computation (shared by z_aux loss and action blending)
    # ============================================================

    def _compute_gt_actions(self, scene_graph, ego_only=True):
        """Compute GT actions (acc, ddh) in normalized action space from consecutive states.

        acc = (speed_next - speed_cur) / dt
        ddh = (hdot_next - hdot_cur) / dt
        Then normalize: (acc - a_mean) / a_std, (ddh - ddh_mean) / ddh_std

        :param scene_graph: with past (NA, PT, 6) and future_gt (NA, FT, 6)
        :param ego_only: if True, return only ego actions
        :return: (N, FT, 2) normalized actions — N is N_ego if ego_only else NA
        """
        device = scene_graph.past.device
        dt = self.bicycle_params['dt']
        a_mean, a_std = self.bicycle_params['a_stats']
        ddh_mean, ddh_std = self.bicycle_params['ddh_stats']

        gt_future = scene_graph.future_gt  # (NA, FT, 6)
        last_past = scene_graph.past[:, -1, :]  # (NA, 6)

        if ego_only:
            ego_inds = scene_graph.ptr[:-1]
            gt_future = gt_future[ego_inds]   # (N_ego, FT, 6)
            last_past = last_past[ego_inds]   # (N_ego, 6)

        # Unnormalize speed (dim 4) and hdot (dim 5)
        s_mean = self.normalizer.mean_vals[4].to(device)
        s_std = self.normalizer.std_vals[4].to(device)
        h_mean = self.normalizer.mean_vals[5].to(device)
        h_std = self.normalizer.std_vals[5].to(device)

        # Build speed sequence: [last_past_speed, future_speed_0, ..., future_speed_{FT-1}]
        prev_speed = last_past[:, 4:5] * s_std + s_mean  # (N, 1)
        future_speed = gt_future[:, :, 4] * s_std + s_mean  # (N, FT)
        speed_seq = torch.cat([prev_speed, future_speed], dim=1)  # (N, FT+1)

        # Build hdot sequence
        prev_hdot = last_past[:, 5:6] * h_std + h_mean  # (N, 1)
        future_hdot = gt_future[:, :, 5] * h_std + h_mean  # (N, FT)
        hdot_seq = torch.cat([prev_hdot, future_hdot], dim=1)  # (N, FT+1)

        # Compute raw actions
        raw_acc = (speed_seq[:, 1:] - speed_seq[:, :-1]) / dt  # (N, FT)
        raw_ddh = (hdot_seq[:, 1:] - hdot_seq[:, :-1]) / dt   # (N, FT)

        # Normalize to action space
        norm_acc = (raw_acc - a_mean) / a_std   # (N, FT)
        norm_ddh = (raw_ddh - ddh_mean) / ddh_std  # (N, FT)

        return torch.stack([norm_acc, norm_ddh], dim=-1)  # (N, FT, 2)

    # ============================================================
    # Dynamics helpers
    # ============================================================

    def _get_6d_state(self, prev_state, mult_samp, NA, NS):
        """Get 6-dim state (pad if needed)."""
        if prev_state.size(-1) >= 6:
            return prev_state
        pad = torch.zeros(prev_state.size(0), 6 - prev_state.size(-1), device=prev_state.device)
        return torch.cat([prev_state, pad], dim=-1)

    def _apply_dynamics(self, decoder_out, prev_state, cur_veh_len, NA, NS, mult_samp):
        """Apply bicycle model and return (global_state, local_state, bike_state)."""
        bsize = NA if not mult_samp else NA * NS

        if self.output_bicycle:
            dynamics_out = decoder_out.view(bsize, 1, 1, 2)
            a_out = dynamics_out[:, :, :, 0] * self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
            ddh_out = dynamics_out[:, :, :, 1] * self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
            init_state = self.normalizer.unnormalize(prev_state)
            cur_bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, cur_veh_len)[:, 0, 0]
            cur_bike_state = self.normalizer.normalize(cur_bike_state)
            cur_state_global = cur_bike_state[:, :4]
            cur_state_local = transform2frame(prev_state[:, :4], cur_state_global.unsqueeze(1))[:, 0]
            return cur_state_global, cur_state_local, cur_bike_state
        else:
            heading_mag = torch.norm(decoder_out[:, 2:], dim=-1, keepdim=True)
            cur_state_local = torch.cat([decoder_out[:, :2], decoder_out[:, 2:] / heading_mag], dim=-1)
            cur_state_global = transform2frame(prev_state, cur_state_local.unsqueeze(1), inverse=True)[:, 0, :]
            return cur_state_global, cur_state_local, None

    def _apply_dynamics_single(self, traj_out, prev_state, veh_len):
        """Apply dynamics for a single agent group (ego or sur only)."""
        N = traj_out.size(0)
        if self.output_bicycle:
            dynamics_out = traj_out.view(N, 1, 1, 2)
            a_out = dynamics_out[:, :, :, 0] * self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
            ddh_out = dynamics_out[:, :, :, 1] * self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
            init_state = self.normalizer.unnormalize(prev_state)
            bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, veh_len)[:, 0, 0]
            bike_state = self.normalizer.normalize(bike_state)
            state_global = bike_state[:, :4]
            return state_global, bike_state
        else:
            heading_mag = torch.norm(traj_out[:, 2:], dim=-1, keepdim=True).clamp(min=1e-6)
            state_local = torch.cat([traj_out[:, :2], traj_out[:, 2:] / heading_mag], dim=-1)
            state_global = transform2frame(prev_state, state_local.unsqueeze(1), inverse=True)[:, 0, :]
            return state_global, None

    def _apply_sur_gt_replay(self, scene_graph, cur_state_global, cur_state_local, cur_bike_state,
                              prev_state, ego_mask, t, mult_samp, NA, NS, num_sur):
        """Override sur predictions with GT future for testing."""
        gt_future = scene_graph.future_gt
        sur_gt_state = gt_future[:, t, :4][~ego_mask]
        cur_state_global = cur_state_global.clone()
        cur_state_local = cur_state_local.clone()

        if mult_samp:
            cur_state_global_reshaped = cur_state_global.reshape(NA, NS, -1)
            cur_state_global_reshaped[~ego_mask] = sur_gt_state.unsqueeze(1).expand(-1, NS, -1)
            cur_state_global = cur_state_global_reshaped.reshape(NA * NS, -1)
            cur_state_local_reshaped = cur_state_local.reshape(NA, NS, -1)
            sur_prev = prev_state.reshape(NA, NS, -1)[~ego_mask].reshape(num_sur * NS, -1)
            sur_gt_expanded = sur_gt_state.unsqueeze(1).expand(-1, NS, -1).reshape(num_sur * NS, -1)
            sur_local = transform2frame(sur_prev[:, :4], sur_gt_expanded.unsqueeze(1))[:, 0, :]
            cur_state_local_reshaped[~ego_mask] = sur_local.reshape(num_sur, NS, -1)
            cur_state_local = cur_state_local_reshaped.reshape(NA * NS, -1)
        else:
            cur_state_global[~ego_mask] = sur_gt_state
            sur_prev = prev_state[~ego_mask]
            cur_state_local[~ego_mask] = transform2frame(sur_prev[:, :4], sur_gt_state.unsqueeze(1))[:, 0, :]

        if self.output_bicycle and cur_bike_state is not None:
            sur_gt_6d = gt_future[:, t, :][~ego_mask]
            cur_bike_state = cur_bike_state.clone()
            if mult_samp:
                cur_bike_state_reshaped = cur_bike_state.reshape(NA, NS, -1)
                cur_bike_state_reshaped[~ego_mask] = sur_gt_6d.unsqueeze(1).expand(-1, NS, -1)
                cur_bike_state = cur_bike_state_reshaped.reshape(NA * NS, -1)
            else:
                cur_bike_state[~ego_mask] = sur_gt_6d

        return cur_state_global, cur_state_local, cur_bike_state

    # ============================================================
    # Utility methods
    # ============================================================

    def _get_ego_mask(self, scene_graph):
        """Boolean mask for ego agents (first in each batch)."""
        NA = scene_graph.x.size(0) if hasattr(scene_graph, 'x') and scene_graph.x is not None else scene_graph.past.size(0)
        ego_mask = torch.zeros(NA, dtype=torch.bool, device=scene_graph.past.device)
        ego_inds = scene_graph.ptr[:-1]
        ego_mask[ego_inds] = True
        return ego_mask

    def _merge_ego_other_feat(self, ego_feat, other_feat, ego_mask):
        """Merge ego and other features back into (NA, ...) tensor."""
        NA = ego_mask.size(0)
        device = ego_feat.device

        if len(ego_feat.size()) == 3:
            NS = ego_feat.size(1)
            D = ego_feat.size(-1)
            merged = torch.zeros(NA, NS, D, device=device)
            merged[ego_mask] = ego_feat
            merged[~ego_mask] = other_feat
        else:
            D = ego_feat.size(-1)
            merged = torch.zeros(NA, D, device=device)
            merged[ego_mask] = ego_feat
            merged[~ego_mask] = other_feat
        return merged

    def _recompute_map_tokens(self, pos_normalized, map_idx, map_env, scene_graph,
                               mult_samp=False, NS=None):
        """
        Re-crop and re-encode map tokens at given normalized positions.

        :param pos_normalized: (NA, 4) or (NA*NS, 4) normalized positions
        :param map_idx: (B,) map index per batch
        :param map_env: map environment for cropping
        :param scene_graph: scene graph (for .batch attribute)
        :param mult_samp: if True, pos is (NA*NS, 4) and mapixes need expansion
        :param NS: number of samples per agent (required if mult_samp=True)
        :return: map_tokens (NA, num_tokens, ch) or (NA*NS, num_tokens, ch)
        """
        pos_unnorm = self.normalizer.unnormalize(pos_normalized)
        if mult_samp and NS is not None:
            mapixes = map_idx[scene_graph.batch]  # (NA,)
            mapixes = mapixes.unsqueeze(1).expand(-1, NS).reshape(-1)  # (NA*NS,)
        else:
            mapixes = map_idx[scene_graph.batch]
        map_obs = map_env.get_map_crop_pos(pos_unnorm, mapixes).to(torch.float)
        map_early = self.map_conv_early(map_obs)
        map_tokens = map_early.flatten(2).permute(0, 2, 1)  # (N, num_tokens, ch)
        return map_tokens

    def _recompute_map_tokens_batched(self, all_pos_normalized, map_idx, map_env, scene_graph):
        """
        Batched map recrop: all timesteps at once → single CNN forward.

        :param all_pos_normalized: (NA, T, 4) normalized positions for all timesteps
        :param map_idx: (B,) map index per batch
        :param map_env: map environment for cropping
        :param scene_graph: scene graph (for .batch attribute)
        :return: map_tokens (T, NA, num_tokens, ch)
        """
        NA, T = all_pos_normalized.shape[0], all_pos_normalized.shape[1]

        # Flatten all positions: (NA*T, 4)
        pos_flat = all_pos_normalized.reshape(NA * T, 4)
        pos_flat_unnorm = self.normalizer.unnormalize(pos_flat)

        # Expand mapixes: (NA,) → (NA*T,)
        mapixes = map_idx[scene_graph.batch]  # (NA,)
        mapixes = mapixes.unsqueeze(1).expand(-1, T).reshape(NA * T)  # (NA*T,)

        # Single crop + CNN forward
        map_obs = map_env.get_map_crop_pos(pos_flat_unnorm, mapixes).to(torch.float)  # (NA*T, C, H, W)
        map_early = self.map_conv_early(map_obs)  # (NA*T, ch, H', W')
        map_tokens = map_early.flatten(2).permute(0, 2, 1)  # (NA*T, num_tokens, ch)

        # Pool raw raster per step for potential-weighted soft labels
        # map_obs: (NA*T, C, 256, 256) → (NA*T, C, 29, 29) → (T, NA, C, 29, 29)
        with torch.no_grad():
            pooled = nn.functional.adaptive_avg_pool2d(map_obs, self.map_token_spatial)
            self._map_raster_pooled_steps = pooled.reshape(
                NA, T, pooled.size(1), self.map_token_spatial, self.map_token_spatial
            ).permute(1, 0, 2, 3, 4)  # (T, NA, C, 29, 29)

        # Reshape: (NA, T, num_tokens, ch) → (T, NA, num_tokens, ch)
        map_tokens = map_tokens.reshape(NA, T, -1, map_tokens.shape[-1])
        map_tokens = map_tokens.permute(1, 0, 2, 3)  # (T, NA, num_tokens, ch)
        return map_tokens

    def rsample(self, mean, var):
        eps = torch.randn_like(mean)
        return mean + eps * torch.sqrt(var)

    def sim_traj(self, init_state, a, ddh, vehicle_len):
        """Bicycle model simulation. Everything is UNNORMALIZED."""
        cur_kinematics = kinematics2angle(init_state)
        sim_steps = a.size(-1)
        kin_seq = []
        for t in range(sim_steps):
            cur_kinematics = car_dynamics(cur_kinematics, a[:, :, t], ddh[:, :, t],
                                          self.dt, 0, 1, 2, 3,
                                          4, vehicle_len, self.bicycle_params['maxhdot'],
                                          self.bicycle_params['maxs'])
            kin_seq.append(kinematics2vec(cur_kinematics))
        return torch.stack(kin_seq, dim=2)

    # ============================================================
    # Phase control & freeze/unfreeze
    # ============================================================

    def set_phase(self, phase):
        """Set training phase: 1=pretrain, 2=finetune."""
        self.phase = phase

    def freeze_z_global(self):
        """Freeze z_global encoder for Phase 2."""
        frozen_modules = [
            self.latent_prior_net, self.latent_posterior_net,
            self.step_feature_extractor, self.temporal_gcn_encoder,
            self.prior_temporal_gru, self.positional_encoding, self.transformer_encoder,
            self.map_conv_early, self.map_conv_late, self.map_feature,
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        Logger.log('Frozen z_global encoder (prior, posterior, GCN, temporal, map encoders)')

    def unfreeze_z_global(self):
        """Unfreeze z_global encoder."""
        unfrozen_modules = [
            self.latent_prior_net, self.latent_posterior_net,
            self.step_feature_extractor, self.temporal_gcn_encoder,
            self.prior_temporal_gru, self.positional_encoding, self.transformer_encoder,
            self.map_conv_early, self.map_conv_late, self.map_feature,
        ]
        for module in unfrozen_modules:
            for param in module.parameters():
                param.requires_grad = True
        Logger.log('Unfrozen z_global encoder')

    def freeze_for_finetuning(self):
        """
        Freeze all EXCEPT ego-specific decoder components + intent codebook.

        FROZEN:
        - Encoder (prior, posterior, temporal GCN, map CNN)
        - Context encoder, map summary pooling
        - Interaction GCN, query_proj, step_pe
        - A2A K/V + sur Q/O, A2C K/V + sur Q/O, A2S K/V + sur Q/O
        - sur FFN (all layers), sur output head, ego pred head

        TRAINABLE:
        - A2A ego Q/O, A2C ego Q/O, A2S ego Q/O (all layers)
        - Ego FFN (all layers), ego output head
        - Intent codebook, intent CE head, sur pred head
        - Ego intent projection
        """
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Then unfreeze ego-specific components
        trainable_modules = [
            self.ego_output_head,
            self.intent_codebook,
            self.ego_intent_proj,
            self.intent_ce_head,
            self.sur_pred_head,
        ]

        # Unfreeze ego Q/O and ego FFN in each Transformer layer
        for layer in self.trans_layers:
            # A2A: ego Q/O
            trainable_modules.extend([
                layer.a2a_attn.ego_q_proj,
                layer.a2a_attn.ego_o_proj,
            ])
            # A2C: ego Q/O
            trainable_modules.extend([
                layer.a2c_ego_q_proj,
                layer.a2c_ego_o_proj,
            ])
            # A2S: ego Q/O
            trainable_modules.extend([
                layer.a2s_ego_q_proj,
                layer.a2s_ego_o_proj,
            ])
            # Ego FFN
            trainable_modules.extend([
                layer.ego_ffn_norm,
                layer.ego_ffn,
            ])

        # Sur intent components: FROZEN in Phase 2 (per design spec Section 5d)
        # sur_codebook, sur_intent_predictor, sur_intent_ce_head → freeze

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        num_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        Logger.log(f'Frozen for fine-tuning: {num_frozen} params frozen, {num_trainable} params trainable')

    # ============================================================
    # Analysis getters
    # ============================================================

    def get_z_local(self):
        """Get z_local from last forward pass.
        Training: tensor (FT, num_ego, intent_dim)
        Inference: list of FT tensors (num_ego, intent_dim)
        """
        if not hasattr(self, '_z_local_outputs'):
            return None
        if isinstance(self._z_local_outputs, torch.Tensor):
            if self._z_local_outputs.numel() == 0:
                return None
            return self._z_local_outputs
        if len(self._z_local_outputs) == 0:
            return None
        return self._z_local_outputs

    def get_z_local_stacked(self):
        """Get z_local stacked as (FT, num_ego, intent_dim)."""
        raw = self.get_z_local()
        if raw is None:
            return None
        if isinstance(raw, torch.Tensor):
            return raw  # Already stacked (training mode)
        if isinstance(raw, list) and len(raw) > 0:
            return torch.stack(raw, dim=0)
        return None

    def get_z_local_mean(self):
        """Backward-compatible alias."""
        return self.get_z_local_stacked()

    def get_z_local_var(self):
        """z_local is discrete — no variance."""
        return None

    def get_intent_weights(self):
        """Get intent selection weights.
        Training: tensor (FT, num_ego, K)
        Inference: list of FT tensors
        """
        if not hasattr(self, '_intent_weights_outputs'):
            return None
        if isinstance(self._intent_weights_outputs, torch.Tensor):
            if self._intent_weights_outputs.numel() == 0:
                return None
            return self._intent_weights_outputs
        if len(self._intent_weights_outputs) == 0:
            return None
        return self._intent_weights_outputs

    def get_ego_map_attn_weights(self):
        """Get ego map attention weights.
        TF: tensor (B*T_total, N_ego, num_tokens) from Layer 0
        AR: list of (1, N_ego, num_tokens) per step → cat to (FT, N_ego, num_tokens)
        """
        if not hasattr(self, '_ego_map_attn_weights_outputs'):
            return None
        if self._ego_map_attn_weights_outputs is None:
            return None
        if isinstance(self._ego_map_attn_weights_outputs, torch.Tensor):
            if self._ego_map_attn_weights_outputs.numel() == 0:
                return None
            return self._ego_map_attn_weights_outputs
        if isinstance(self._ego_map_attn_weights_outputs, list):
            if len(self._ego_map_attn_weights_outputs) == 0:
                return None
            return torch.cat(self._ego_map_attn_weights_outputs, dim=0)  # (FT, N_ego, num_tokens)
        return None

    def get_sur_map_attn_weights(self):
        """Get sur map attention weights.
        TF: tensor (B*T_total, N_sur, num_tokens) from Layer 0
        AR: list of (1, N_sur, num_tokens) per step → cat to (FT, N_sur, num_tokens)
        """
        if not hasattr(self, '_sur_map_attn_weights_outputs'):
            return None
        if self._sur_map_attn_weights_outputs is None:
            return None
        if isinstance(self._sur_map_attn_weights_outputs, torch.Tensor):
            if self._sur_map_attn_weights_outputs.numel() == 0:
                return None
            return self._sur_map_attn_weights_outputs
        if isinstance(self._sur_map_attn_weights_outputs, list):
            if len(self._sur_map_attn_weights_outputs) == 0:
                return None
            return torch.cat(self._sur_map_attn_weights_outputs, dim=0)  # (FT, N_sur, num_tokens)
        return None

    def get_map_raster_pooled(self):
        """Get pooled raw raster for potential-weighted soft labels.
        Initial crop: (NA, C, grid, grid).
        Per-step (map_recrop): (T, NA, C, grid, grid).
        Returns whichever is available, preferring per-step."""
        if hasattr(self, '_map_raster_pooled_steps') and self._map_raster_pooled_steps is not None:
            return self._map_raster_pooled_steps  # (T, NA, C, 29, 29)
        if hasattr(self, '_map_raster_pooled') and self._map_raster_pooled is not None:
            return self._map_raster_pooled  # (NA, C, 29, 29)
        return None

    def get_map_attn_weights(self):
        """Get ego map attention weights (detached, for analysis)."""
        w = self.get_ego_map_attn_weights()
        if w is None:
            return None
        if isinstance(w, torch.Tensor):
            return w.detach()
        if isinstance(w, list):
            return [x.detach() if isinstance(x, torch.Tensor) else x for x in w]
        return w

    def get_attn_weights(self):
        return self.get_map_attn_weights()

    def get_sur_pred_outputs(self):
        """Get predicted sur deltas.
        Training: tensor (FT, num_ego, sur_pred_dim)
        Inference: list of tensors
        """
        if not hasattr(self, '_sur_pred_outputs'):
            return None
        if isinstance(self._sur_pred_outputs, torch.Tensor):
            if self._sur_pred_outputs.numel() == 0:
                return None
            return self._sur_pred_outputs
        if len(self._sur_pred_outputs) == 0:
            return None
        return self._sur_pred_outputs

    def get_ego_pred_outputs(self):
        """Get predicted ego deltas.
        Training: tensor (FT, num_sur, sur_pred_dim)
        Inference: list of tensors
        """
        if not hasattr(self, '_ego_pred_outputs'):
            return None
        if isinstance(self._ego_pred_outputs, torch.Tensor):
            if self._ego_pred_outputs.numel() == 0:
                return None
            return self._ego_pred_outputs
        if len(self._ego_pred_outputs) == 0:
            return None
        return self._ego_pred_outputs

    def get_z_local_sparsity_loss(self, target_sparsity=0.1):
        """Backward-compatible sparsity loss (unused in redesign but kept for interface)."""
        z_local = self.get_z_local()
        if z_local is None:
            return torch.tensor(0.0)
        l1_loss = torch.mean(torch.abs(z_local))
        return l1_loss
