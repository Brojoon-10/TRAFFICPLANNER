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

from torch_geometric.data import Data


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

class DecoderHistoryAttention(nn.Module):
    """
    Cross-attention over accumulated GCN features in history buffer.
    Q = current GRU hidden, KV = GCN features from steps 0..t-1 with learnable PE.
    Ego and Sur each have separate instances (separate weights).
    """
    def __init__(self, d_model, nhead=4, max_len=12):
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.pe = nn.Embedding(max_len, d_model)  # learnable positional encoding

    def forward(self, query, history_buffer, t):
        """
        :param query: (N, D) — current GRU hidden state (last layer)
        :param history_buffer: (N, FT, D) — pre-allocated, GCN features accumulated
        :param t: current timestep (0-indexed)
        :return: (N, D) — history context
        """
        if t == 0:
            return torch.zeros_like(query)

        # GCN feature → projected space + positional encoding
        history = self.feat_proj(history_buffer[:, :t, :])  # (N, t, D)
        positions = self.pe(torch.arange(t, device=query.device))  # (t, D)
        history = history + positions.unsqueeze(0)  # broadcast (1, t, D)

        Q = query.unsqueeze(1)  # (N, 1, D)
        attn_out, _ = self.cross_attn(Q, history, history)
        return attn_out.squeeze(1)  # (N, D)

    def forward_batched(self, queries, history_buffer, FT):
        """
        Batched forward: all FT timesteps at once using causal mask.

        Each timestep t attends to history[0..t-1]. t=0 sees nothing → zero output.
        To avoid NaN from all-masked softmax at t=0, we only process t=1..FT-1
        through MHA and set t=0 output to zero.

        :param queries: (N, FT, D) — GRU hidden states for all timesteps
        :param history_buffer: (N, FT, D) — full GCN features for all timesteps
        :param FT: number of future timesteps
        :return: (N, FT, D) — history context for all timesteps
        """
        N, _, D = queries.size()
        device = queries.device

        # Output buffer — t=0 stays zero
        output = torch.zeros(N, FT, D, device=device)

        if FT <= 1:
            return output

        # Project all history + add PE
        history = self.feat_proj(history_buffer)  # (N, FT, D)
        positions = self.pe(torch.arange(FT, device=device))  # (FT, D)
        history = history + positions.unsqueeze(0)  # (N, FT, D)

        # Only process t=1..FT-1 (FT-1 queries)
        FT1 = FT - 1  # number of active queries
        Q = queries[:, 1:, :].reshape(N * FT1, 1, D)  # (N*FT1, 1, D)
        KV = history.unsqueeze(1).expand(N, FT1, FT, D).reshape(N * FT1, FT, D)  # (N*FT1, FT, D)

        # Causal mask for t=1..FT-1: timestep t can see history[0..t-1]
        # Mask shape: (FT1, FT) where row i corresponds to t=i+1
        causal_mask = torch.zeros(FT1, FT, device=device)
        for i in range(FT1):
            t = i + 1  # actual timestep
            causal_mask[i, t:] = float('-inf')  # block positions t..FT-1

        # Expand for batch: (N*FT1, FT) → (N*FT1*num_heads, 1, FT)
        # NOTE: must use expand (batch-major) not repeat (head-major) to match
        # MHA's internal ordering: (B, nhead, ...) → (B*nhead, ...)
        num_heads = self.cross_attn.num_heads
        t_indices = torch.arange(FT1, device=device).unsqueeze(0).expand(N, FT1).reshape(N * FT1)
        batch_mask = causal_mask[t_indices]  # (N*FT1, FT)
        batch_mask = batch_mask.unsqueeze(1).unsqueeze(1)  # (N*FT1, 1, 1, FT)
        batch_mask = batch_mask.expand(-1, num_heads, -1, -1).reshape(
            N * FT1 * num_heads, 1, FT)  # (N*FT1*num_heads, 1, FT)

        attn_out, _ = self.cross_attn(Q, KV, KV, attn_mask=batch_mask)
        attn_out = attn_out.squeeze(1).reshape(N, FT1, D)  # (N, FT1, D)

        output[:, 1:, :] = attn_out

        return output


class MapCrossAttention(nn.Module):
    """
    Cross-attention over CNN spatial features (map tokens).
    Q = current GRU hidden, KV = conv3 spatial tokens (H*W tokens).
    Ego and Sur each have separate instances.
    """
    def __init__(self, d_model, map_feat_dim, nhead=4):
        super().__init__()
        self.map_proj = nn.Linear(map_feat_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, map_tokens):
        """
        :param query: (N, D) — agent's current GRU hidden state
        :param map_tokens: (N, num_tokens, map_feat_dim) — conv3 spatial features
        :return: (attn_out, attn_weights)
            attn_out: (N, D)
            attn_weights: (N, 1, num_tokens)
        """
        map_kv = self.map_proj(map_tokens)  # (N, num_tokens, D)
        Q = query.unsqueeze(1)  # (N, 1, D)
        attn_out, attn_weights = self.cross_attn(Q, map_kv, map_kv)
        return attn_out.squeeze(1), attn_weights  # (N, D), (N, 1, num_tokens)


class IntentCodebook(nn.Module):
    """
    Discrete intent codebook for z_local (ego only).
    K learnable intent vectors, selected via Gumbel-Softmax.
    Phase 1: returns zero (inactive). Phase 2: active.
    """
    def __init__(self, num_intents=8, intent_dim=32, input_dim=73):
        super().__init__()
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.codebook = nn.Embedding(num_intents, intent_dim)
        self.intent_predictor = MLP([input_dim, 64, num_intents])

    def forward(self, situation, temperature=1.0):
        """
        :param situation: (N_ego, input_dim) — ego_state + predicted_sur_delta + map_ctx(detached)
        :param temperature: Gumbel-Softmax temperature
        :return: (z_local, intent_weights)
            z_local: (N_ego, intent_dim)
            intent_weights: (N_ego, K)
        """
        logits = self.intent_predictor(situation)  # (N_ego, K)

        if self.training:
            intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            intent_weights = F.one_hot(logits.argmax(-1), self.num_intents).float()

        z_local = intent_weights @ self.codebook.weight  # (N_ego, intent_dim)
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
                 latent_size=32,          # z_global size (was 64, now 32)
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
                 hist_attn_nhead=4,
                 map_attn_nhead=4,
                 sur_pred_dim=2,          # predicted sur delta dim (dx, dy)
                 map_recrop=False,        # re-crop map tokens every decode step
                 use_ego_intent=True,     # enable ego intent codebook
                 use_sur_intent=False,    # enable sur intent codebook
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
        print(f'Map tokens: {self.map_token_spatial}x{self.map_token_spatial} = {self.map_num_tokens} tokens, ch={self.map_token_ch}')

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
        self.positional_encoding = PositionalEncoding(gcn_hidden_dim, max_len=max(self.PT, self.FT))
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
        # Map re-crop flag
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
        # DECODER COMPONENTS (Redesign)
        # =============================================
        #

        # 1. Interaction GCN: one shared GCN for history buffer (replaces decoder_net + z_local_gcn)
        # Input: state(6) + lw(2) + sem(NC) = 10
        interaction_gcn_in = self.state_size + self.att_feat_size + self.NC
        self.interaction_gcn = IndividualSceneInteractionNet(
            interaction_gcn_in, self.NC, 4, 64, self.d_model,
        )

        # 2. History Attention (ego/sur separate)
        self.ego_history_attn = DecoderHistoryAttention(self.d_model, nhead=hist_attn_nhead, max_len=self.FT)
        self.sur_history_attn = DecoderHistoryAttention(self.d_model, nhead=hist_attn_nhead, max_len=self.FT)

        # 3. Map Cross-Attention (ego/sur separate)
        self.ego_map_attn = MapCrossAttention(self.d_model, self.map_token_ch, nhead=map_attn_nhead)
        self.sur_map_attn = MapCrossAttention(self.d_model, self.map_token_ch, nhead=map_attn_nhead)

        # 4. Intent Codebook
        intent_input_dim = self.state_size + self.sur_pred_dim + self.d_model  # 6 + 2 + 64 = 72
        self.intent_codebook = IntentCodebook(
            num_intents=num_intents,
            intent_dim=self.intent_dim,
            input_dim=intent_input_dim,
        )
        # Sur Intent Codebook (symmetric to ego)
        self.sur_intent_codebook = IntentCodebook(
            num_intents=num_intents,
            intent_dim=self.intent_dim,
            input_dim=intent_input_dim,  # same dim: state(6) + predicted_ego_delta(2) + sur_map_ctx(64)
        )

        # 5. Ego decoder GRU
        # Input: hist_ctx(D) + map_ctx(D) + z_global(z_size) + z_local(intent_dim) + lw(2) + sem(NC)
        ego_gru_in_size = self.d_model + self.d_model + self.z_size + self.intent_dim + self.att_feat_size + self.NC
        self.ego_decoder_gru = nn.GRU(
            ego_gru_in_size, self.d_model, 3, batch_first=True,
        )
        self.ego_output_head = nn.Linear(self.d_model, self.traj_out_size)

        # 6. Sur decoder GRU
        # Input: hist_ctx(D) + map_ctx(D) + z_global(z_size) + z_local_sur(intent_dim) + lw(2) + sem(NC)
        sur_gru_in_size = self.d_model + self.d_model + self.z_size + self.intent_dim + self.att_feat_size + self.NC
        self.sur_decoder_gru = nn.GRU(
            sur_gru_in_size, self.d_model, 3, batch_first=True,
        )
        self.sur_output_head = nn.Linear(self.d_model, self.traj_out_size)

        # 7. Auxiliary loss heads (all Linear — no MLP, force modules to encode directly)
        self.sur_pred_head = nn.Linear(self.d_model, self.sur_pred_dim)  # (B) ego→sur delta
        self.ego_pred_head = nn.Linear(self.d_model, self.sur_pred_dim)  # (C) sur→ego delta
        self.intent_ce_head = nn.Linear(self.intent_dim, self.num_intents)  # (A) ego intent CE head
        self.sur_intent_ce_head = nn.Linear(self.intent_dim, self.num_intents)  # sur intent CE head

        # 8. Ego warmup GRU for initializing ego_decoder_gru hidden state
        self.ego_warmup_gru = nn.GRU(
            gcn_hidden_dim, self.d_model, 3, batch_first=True,
        )

        # 9. Sur warmup GRU for initializing sur_decoder_gru hidden state
        self.sur_warmup_gru = nn.GRU(
            gcn_hidden_dim, self.d_model, 3, batch_first=True,
        )

        # Phase control
        self.phase = 1
        self.gumbel_temperature = 1.0

        # Intent control flags (per agent type)
        self.use_ego_intent = use_ego_intent
        self.use_sur_intent = use_sur_intent


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
                teacher_forcing=False):
        # Map encoding (encoder-level 64dim + decoder-level tokens)
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)

        # PRIOR
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)

        # POSTERIOR
        past_context = past_seq_out[:, -1, :]
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_context)

        # DECODER
        if use_post_mean:
            z_samp = post_mu
        else:
            z_samp = self.rsample(post_mu, post_var)
        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp, map_idx, map_env,
                                   map_tokens=map_tokens,
                                   teacher_forcing=teacher_forcing)

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

        for t in range(T):
            cur_state = traj_data[:, t, :]
            cur_vis = vis_data[:, t].unsqueeze(-1)
            cur_lw = scene_graph.lw
            cur_sem = scene_graph.sem

            step_in_feat = torch.cat([cur_state, cur_lw, cur_vis, cur_sem], dim=-1)
            gcn_node_in = self.step_feature_extractor(step_in_feat)

            g_in_data.x = gcn_node_in
            g_in_data.pos = cur_state[:, :4]

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
    # Decoder
    # ============================================================

    def decoder(self, scene_graph, map_feat, past_seq_out, z, map_idx, map_env,
                map_tokens=None,
                ext_future=None, nfuture=None,
                teacher_forcing=False,
                sur_gt_replay=False):
        """
        Decoder dispatcher.
        If map_tokens not provided, compute them (slower path for backward compat).
        """
        if map_tokens is None:
            # Fallback: compute map tokens (needed for decode_embedding with old embed format)
            scene_graph_pos_backup = scene_graph.pos.clone()
            scene_graph.pos = scene_graph.past[:, -1, :4]
            _, map_tokens = self.encode_map(scene_graph, map_idx, map_env, return_tokens=True)
            scene_graph.pos = scene_graph_pos_backup

        if teacher_forcing:
            return self.teacher_forcing_decoder(scene_graph, map_feat, past_seq_out, z,
                                                map_idx, map_env, map_tokens)
        else:
            return self.autoregressive_decoder(scene_graph, map_feat, past_seq_out, z,
                                               map_idx, map_env, map_tokens,
                                               ext_future=ext_future, nfuture=nfuture,
                                               sur_gt_replay=sur_gt_replay)

    def autoregressive_decoder(self, scene_graph, map_feat, past_seq_out, z,
                                map_idx, map_env, map_tokens,
                                ext_future=None, nfuture=None, sur_gt_replay=False):
        """
        Autoregressive decoder (Redesign).

        6-step loop per timestep:
        1. GCN → history buffer
        2. History Attention (ego + sur)
        3. Map Cross-Attention (ego + sur)
        4. Intent Selection (ego only, Phase 2)
        5. GRU + Output (ego + sur)
        6. Scene graph update + bicycle model
        """
        NA = map_feat.size(0)
        FT = self.FT if nfuture is None else nfuture
        B = map_idx.size(0)

        ego_inds = scene_graph.ptr[:-1]
        ego_mask = self._get_ego_mask(scene_graph)
        num_ego = int(ego_mask.sum())
        num_sur = NA - num_ego
        device = map_feat.device

        # Multi-sample check
        zsize = z.size()
        NS = None
        mult_samp = len(zsize) == 3
        if mult_samp:
            NS = zsize[1]

        # Initialize prev_state
        prev_state = scene_graph.past[:, -1, :] if self.output_bicycle else scene_graph.past[:, -1, :4]

        # Vehicle lengths for bicycle model
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)
        cur_lw = scene_graph.lw
        cur_sem = scene_graph.sem

        # Map tokens: initial tokens (used for mult_samp fallback)
        # For single-sample (training): re-cropped every step inside loop
        # For mult_samp (inference): use initial tokens (re-crop too expensive)
        ego_map_tokens = map_tokens[ego_mask]  # (num_ego, num_tokens, ch) or (num_ego, NS, ...)
        sur_map_tokens = map_tokens[~ego_mask]

        # z_global per agent type
        z_ego = z[ego_mask]  # (num_ego, z_size) or (num_ego, NS, z_size)
        z_sur = z[~ego_mask]

        # Pre-allocate history buffers
        ego_history_buffer = torch.zeros(num_ego, FT, self.d_model, device=device)
        sur_history_buffer = torch.zeros(num_sur, FT, self.d_model, device=device)

        # Initialize GRU hidden states via warmup
        ego_gru_hidden = self._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=True, gt_future=None, target_t=0)
        sur_gru_hidden = self._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=False, gt_future=None, target_t=0)

        # Handle multi-sample expansion
        if mult_samp:
            prev_state = prev_state.unsqueeze(1).expand(NA, NS, -1).reshape(NA * NS, -1)
            cur_veh_len = cur_veh_len.unsqueeze(1).expand(NA, NS, 1).reshape(NA * NS, 1)
            ego_gru_hidden = ego_gru_hidden.unsqueeze(2).expand(-1, -1, NS, -1).contiguous().reshape(3, num_ego * NS, self.d_model)
            sur_gru_hidden = sur_gru_hidden.unsqueeze(2).expand(-1, -1, NS, -1).contiguous().reshape(3, num_sur * NS, self.d_model)
            ego_map_tokens = ego_map_tokens.unsqueeze(1).expand(-1, NS, -1, -1).reshape(num_ego * NS, self.map_num_tokens, self.map_token_ch)
            sur_map_tokens = sur_map_tokens.unsqueeze(1).expand(-1, NS, -1, -1).reshape(num_sur * NS, self.map_num_tokens, self.map_token_ch)
            ego_history_buffer = ego_history_buffer.unsqueeze(1).expand(-1, NS, -1, -1).reshape(num_ego * NS, FT, self.d_model).contiguous()
            sur_history_buffer = sur_history_buffer.unsqueeze(1).expand(-1, NS, -1, -1).reshape(num_sur * NS, FT, self.d_model).contiguous()
            z_ego = z_ego.reshape(num_ego * NS, self.z_size)
            z_sur = z_sur.reshape(num_sur * NS, self.z_size)
            if ext_future is not None:
                ext_future = ext_future.unsqueeze(1).expand(NA, NS, -1, 4).reshape(NA * NS, -1, 4)
            scene_graph.pos = scene_graph.past[:, -1, :4].unsqueeze(1).expand(NA, NS, -1)

        # Initialize analysis storage
        self._z_local_outputs = []
        self._intent_weights_outputs = []
        self._z_local_sur_outputs = []
        self._sur_intent_weights_outputs = []
        self._ego_map_attn_weights_outputs = []
        self._sur_map_attn_weights_outputs = []
        self._sur_pred_outputs = []
        self._ego_pred_outputs = []

        traj_dim = 4  # global state dim (x, y, heading, speed or similar)
        if mult_samp:
            traj_out = torch.zeros(NA * NS, FT, traj_dim, device=device)
        else:
            traj_out = torch.zeros(NA, FT, traj_dim, device=device)

        for t in range(FT):
            # ============================================================
            # STEP 1: GCN → History Buffer
            # ============================================================
            cur_state_6d = self._get_6d_state(prev_state, mult_samp, NA, NS)

            if mult_samp:
                cur_state_for_gcn = cur_state_6d.reshape(NA, NS, -1)
                gcn_in = torch.cat([cur_state_for_gcn,
                                    cur_lw.unsqueeze(1).expand(-1, NS, -1),
                                    cur_sem.unsqueeze(1).expand(-1, NS, -1)], dim=-1)
                scene_graph.x = gcn_in
                scene_graph.pos = cur_state_for_gcn[..., :4]
            else:
                gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)
                scene_graph.x = gcn_in
                scene_graph.pos = cur_state_6d[:, :4]

            ego_gcn_feat, sur_gcn_feat = self.interaction_gcn(scene_graph, ego_mask)
            # ego_gcn_feat: (num_ego, D) or (num_ego, NS, D)
            # sur_gcn_feat: (num_sur, D) or (num_sur, NS, D)

            if mult_samp:
                ego_gcn_feat_flat = ego_gcn_feat.reshape(num_ego * NS, self.d_model)
                sur_gcn_feat_flat = sur_gcn_feat.reshape(num_sur * NS, self.d_model)
            else:
                ego_gcn_feat_flat = ego_gcn_feat
                sur_gcn_feat_flat = sur_gcn_feat

            # In-place history buffer update
            ego_history_buffer[:, t, :] = ego_gcn_feat_flat
            sur_history_buffer[:, t, :] = sur_gcn_feat_flat

            # ============================================================
            # STEP 2: History Attention
            # ============================================================
            ego_gru_h_last = ego_gru_hidden[-1]  # (num_ego[*NS], D)
            sur_gru_h_last = sur_gru_hidden[-1]

            ego_hist_ctx = self.ego_history_attn(ego_gru_h_last, ego_history_buffer, t)  # (num_ego[*NS], D)
            sur_hist_ctx = self.sur_history_attn(sur_gru_h_last, sur_history_buffer, t)

            # Auxiliary: predicted sur delta from ego history context
            predicted_sur_delta = self.sur_pred_head(ego_hist_ctx)  # (num_ego[*NS], sur_pred_dim)
            predicted_ego_delta = self.ego_pred_head(sur_hist_ctx)  # (num_sur[*NS], sur_pred_dim)

            self._sur_pred_outputs.append(predicted_sur_delta.detach())
            self._ego_pred_outputs.append(predicted_ego_delta.detach())

            # ============================================================
            # STEP 3: Map Cross-Attention
            # ============================================================
            if self.map_recrop and map_idx is not None and map_env is not None:
                if mult_samp:
                    cur_map_tokens = self._recompute_map_tokens(
                        cur_state_6d[:, :4], map_idx, map_env, scene_graph,
                        mult_samp=True, NS=NS)
                    ego_map_tokens = cur_map_tokens.reshape(NA, NS, -1, self.map_token_ch)[ego_mask].reshape(
                        num_ego * NS, self.map_num_tokens, self.map_token_ch)
                    sur_map_tokens = cur_map_tokens.reshape(NA, NS, -1, self.map_token_ch)[~ego_mask].reshape(
                        num_sur * NS, self.map_num_tokens, self.map_token_ch)
                else:
                    cur_map_tokens = self._recompute_map_tokens(
                        cur_state_6d[:, :4], map_idx, map_env, scene_graph)
                    ego_map_tokens = cur_map_tokens[ego_mask]
                    sur_map_tokens = cur_map_tokens[~ego_mask]

            ego_map_ctx, ego_map_attn_w = self.ego_map_attn(ego_gru_h_last, ego_map_tokens)
            sur_map_ctx, sur_map_attn_w = self.sur_map_attn(sur_gru_h_last, sur_map_tokens)

            self._ego_map_attn_weights_outputs.append(ego_map_attn_w)
            self._sur_map_attn_weights_outputs.append(sur_map_attn_w)

            # ============================================================
            # STEP 4: Intent Selection (ego only)
            # ============================================================
            if mult_samp:
                ego_state_for_intent = cur_state_6d.reshape(NA, NS, -1)[ego_mask].reshape(num_ego * NS, -1)
            else:
                ego_state_for_intent = cur_state_6d[ego_mask]

            intent_input = torch.cat([
                ego_state_for_intent,
                predicted_sur_delta,
                ego_map_ctx.detach(),  # detach: prevent intent CE from affecting map attn
            ], dim=-1)

            z_local, intent_weights = self.intent_codebook(
                intent_input, temperature=self.gumbel_temperature)

            if not self.use_ego_intent:
                z_local = torch.zeros_like(z_local)

            self._z_local_outputs.append(z_local.detach())
            self._intent_weights_outputs.append(intent_weights.detach())

            # Sur Intent Selection (symmetric to ego)
            if mult_samp:
                sur_state_for_intent = cur_state_6d.reshape(NA, NS, -1)[~ego_mask].reshape(num_sur * NS, -1)
            else:
                sur_state_for_intent = cur_state_6d[~ego_mask]

            sur_intent_input = torch.cat([
                sur_state_for_intent,
                predicted_ego_delta,
                sur_map_ctx.detach(),
            ], dim=-1)

            if self.use_sur_intent:
                z_local_sur, sur_intent_weights = self.sur_intent_codebook(
                    sur_intent_input, temperature=self.gumbel_temperature)
            else:
                z_local_sur = torch.zeros(sur_state_for_intent.size(0), self.intent_dim, device=device)
                sur_intent_weights = torch.zeros(sur_state_for_intent.size(0), self.num_intents, device=device)

            self._z_local_sur_outputs.append(z_local_sur.detach())
            self._sur_intent_weights_outputs.append(sur_intent_weights.detach())

            # ============================================================
            # STEP 5: GRU + Output
            # ============================================================
            if mult_samp:
                ego_lw_flat = cur_lw[ego_mask].unsqueeze(1).expand(-1, NS, -1).reshape(num_ego * NS, -1)
                ego_sem_flat = cur_sem[ego_mask].unsqueeze(1).expand(-1, NS, -1).reshape(num_ego * NS, -1)
                sur_lw_flat = cur_lw[~ego_mask].unsqueeze(1).expand(-1, NS, -1).reshape(num_sur * NS, -1)
                sur_sem_flat = cur_sem[~ego_mask].unsqueeze(1).expand(-1, NS, -1).reshape(num_sur * NS, -1)
            else:
                ego_lw_flat = cur_lw[ego_mask]
                ego_sem_flat = cur_sem[ego_mask]
                sur_lw_flat = cur_lw[~ego_mask]
                sur_sem_flat = cur_sem[~ego_mask]

            # Ego GRU
            ego_gru_in = torch.cat([
                ego_hist_ctx, ego_map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
            ], dim=-1).unsqueeze(1)  # (num_ego[*NS], 1, ego_gru_in_size)

            ego_gru_out, ego_gru_hidden = self.ego_decoder_gru(ego_gru_in, ego_gru_hidden)
            ego_traj_out = self.ego_output_head(ego_gru_out[:, 0])  # (num_ego[*NS], traj_out_size)

            # Sur GRU
            sur_gru_in = torch.cat([
                sur_hist_ctx, sur_map_ctx, z_sur, z_local_sur, sur_lw_flat, sur_sem_flat
            ], dim=-1).unsqueeze(1)

            sur_gru_out, sur_gru_hidden = self.sur_decoder_gru(sur_gru_in, sur_gru_hidden)
            sur_traj_out = self.sur_output_head(sur_gru_out[:, 0])

            # ============================================================
            # STEP 6: Merge + Bicycle Model + State Update
            # ============================================================
            if mult_samp:
                ego_traj_out = ego_traj_out.reshape(num_ego, NS, -1)
                sur_traj_out = sur_traj_out.reshape(num_sur, NS, -1)

            decoder_out = self._merge_ego_other_feat(ego_traj_out, sur_traj_out, ego_mask)

            if mult_samp:
                decoder_out = decoder_out.reshape(NA * NS, -1)

            cur_state_global, cur_state_local, cur_bike_state = self._apply_dynamics(
                decoder_out, prev_state, cur_veh_len, NA, NS, mult_samp)

            # Handle external future
            if ext_future is not None:
                cur_state_global = cur_state_global.clone()
                if mult_samp:
                    ext_ego_inds = ego_inds.unsqueeze(1).expand(B, NS).reshape(B * NS)
                    cur_state_global[ext_ego_inds] = ext_future[:, t]
                else:
                    cur_state_global[ego_inds] = ext_future[:, t]
                cur_state_local = cur_state_local.clone()
                if mult_samp:
                    cur_state_local[ext_ego_inds] = transform2frame(
                        prev_state[ext_ego_inds][:, :4], cur_state_global[ext_ego_inds].unsqueeze(1))[:, 0, :]
                else:
                    cur_state_local[ego_inds] = transform2frame(
                        prev_state[ego_inds][:, :4], cur_state_global[ego_inds].unsqueeze(1))[:, 0, :]

            # Handle sur GT replay
            if sur_gt_replay and hasattr(scene_graph, 'future_gt'):
                cur_state_global, cur_state_local, cur_bike_state = self._apply_sur_gt_replay(
                    scene_graph, cur_state_global, cur_state_local, cur_bike_state,
                    prev_state, ego_mask, t, mult_samp, NA, NS, num_sur)

            traj_out[:, t, :] = cur_state_global

            # Update prev_state
            if self.output_bicycle and cur_bike_state is not None:
                prev_state = cur_bike_state
            else:
                prev_state = cur_state_global

            # Update scene_graph.pos for next step GCN
            if t < FT - 1:
                if mult_samp:
                    scene_graph.pos = cur_state_global.detach().reshape(NA, NS, -1)
                else:
                    scene_graph.pos = cur_state_global.detach()

        if mult_samp:
            traj_out = traj_out.reshape(NA, NS, FT, traj_dim)
        return traj_out

    def teacher_forcing_decoder(self, scene_graph, map_feat, past_seq_out, z,
                                 map_idx, map_env, map_tokens):
        """
        GT-cached Teacher Forcing decoder (Redesign v2).

        Both ego and sur use GT-based pre-computation with independent 1-step prediction.
        No segment structure, no AR dependency during training.

        Loop 1: Sur prediction (ego=GT, sur=GT-cached 1-step)
        Loop 2: Ego prediction (sur=GT, ego=GT-cached 1-step)

        Returns: (NA, FT, 4) flat tensor (same format as AR decoder)
        """
        NA = map_feat.size(0)
        FT = self.FT
        B = map_idx.size(0)

        ego_mask = self._get_ego_mask(scene_graph)
        num_ego = int(ego_mask.sum())
        num_sur = NA - num_ego
        device = map_feat.device

        gt_future = scene_graph.future_gt[:, :, :]

        # Initialize analysis storage
        self._z_local_outputs = []
        self._intent_weights_outputs = []
        self._z_local_sur_outputs = []
        self._sur_intent_weights_outputs = []
        self._ego_map_attn_weights_outputs = []
        self._sur_map_attn_weights_outputs = []
        self._sur_pred_outputs = []
        self._ego_pred_outputs = []

        # Map tokens
        ego_map_tokens = map_tokens[ego_mask]
        sur_map_tokens = map_tokens[~ego_mask]

        # z_global
        z_ego = z[ego_mask]
        z_sur = z[~ego_mask]

        cur_lw = scene_graph.lw
        cur_sem = scene_graph.sem
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:, 0].unsqueeze(1)

        # ====================================================================
        # Loop 1: Sur prediction (ego=GT, sur=GT-cached 1-step)
        # ====================================================================
        sur_traj_all = self._sur_loop_decoder(
            scene_graph, z_sur, ego_mask, gt_future,
            sur_map_tokens, cur_lw, cur_sem, cur_veh_len,
            map_idx, map_env)

        # ====================================================================
        # Loop 2: Ego prediction (sur=GT, ego=GT-cached 1-step)
        # ====================================================================
        ego_traj_all = self._ego_loop_decoder(
            scene_graph, z_ego, ego_mask, gt_future,
            ego_map_tokens, cur_lw, cur_sem, cur_veh_len,
            map_idx, map_env)

        # ====================================================================
        # Merge into flat (NA, FT, 4) tensor
        # ====================================================================
        traj_out = torch.zeros(NA, FT, 4, device=device)
        traj_out[ego_mask] = ego_traj_all
        traj_out[~ego_mask] = sur_traj_all

        return traj_out

    def _sur_loop_decoder(self, scene_graph, z_sur, ego_mask, gt_future,
                              sur_map_tokens, cur_lw, cur_sem, cur_veh_len,
                              map_idx=None, map_env=None):
        """Sur loop: ego=GT, sur=GT-cached independent 1-step prediction (BATCHED).

        All 3 steps are parallelized:
        Step A: Batch.from_data_list → 1 GCN call for all FT timesteps
        Step B: Batched input prep + sequential GRU for hidden snapshots (no_grad)
        Step C: (num_sur*FT) batched 1-step prediction
        """
        NA = ego_mask.size(0)
        FT = self.FT
        num_ego = int(ego_mask.sum())
        num_sur = NA - num_ego
        device = z_sur.device

        # ================================================================
        # Step A: Batched GCN — 12 graphs → 1 Batch call
        # ================================================================
        # Prepare GT states for all FT: t=0 uses past[:,-1,:], t>0 uses gt_future[:,t-1,:]
        gt_state_t0 = scene_graph.past[:, -1, :]  # (NA, S)
        all_gt_states = torch.cat([gt_state_t0.unsqueeze(1), gt_future[:, :FT-1, :]], dim=1)  # (NA, FT, S)

        # Ensure 6d states
        S = all_gt_states.size(-1)
        if S < 6:
            pad = torch.zeros(NA, FT, 6 - S, device=device)
            all_gt_states_6d = torch.cat([all_gt_states, pad], dim=-1)
        else:
            all_gt_states_6d = all_gt_states

        # Build manually batched graph: FT copies with edge_index offset
        batched_graph = self._build_temporal_graph(
            scene_graph, all_gt_states_6d, cur_lw, cur_sem, FT)
        batched_ego_mask = ego_mask.repeat(FT)
        _, sur_gcn_all = self.interaction_gcn(batched_graph, batched_ego_mask)
        # sur_gcn_all: (num_sur*FT, D) → reshape to (FT, num_sur, D) → (num_sur, FT, D)
        gt_gcn_cache = sur_gcn_all.reshape(FT, num_sur, self.d_model).permute(1, 0, 2).contiguous()

        # Pre-compute GT map tokens if re-crop enabled
        gt_map_tokens_cache = None
        if self.map_recrop and map_idx is not None and map_env is not None:
            gt_map_tokens_cache = []
            for t in range(FT):
                gt_pos = all_gt_states_6d[:, t, :4]
                cur_map_tokens = self._recompute_map_tokens(
                    gt_pos, map_idx, map_env, scene_graph)
                gt_map_tokens_cache.append(cur_map_tokens[~ego_mask])

        # ================================================================
        # Step B: Warmup (with grad) + GT forward pass for hidden snapshots (no_grad)
        # GRU hidden accumulation is sequential (h[t] depends on h[t-1]).
        # But GRU inputs at each step are pre-computable from GT.
        # ================================================================
        sur_gru_hidden_init = self._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=False, gt_future=None, target_t=0)

        with torch.no_grad():
            sur_gru_hidden = sur_gru_hidden_init.detach().clone()
            gt_hidden_snapshots = [sur_gru_hidden.clone()]

            # gt_gcn_cache is the history buffer itself
            gt_history_buffer = gt_gcn_cache  # (num_sur, FT, D)

            for t in range(FT):
                sur_gru_h_last = sur_gru_hidden[-1]
                sur_hist_ctx = self.sur_history_attn(sur_gru_h_last, gt_history_buffer, t)

                predicted_ego_delta = self.ego_pred_head(sur_hist_ctx)

                cur_sur_map_tokens = gt_map_tokens_cache[t] if gt_map_tokens_cache is not None else sur_map_tokens
                sur_map_ctx, _ = self.sur_map_attn(sur_gru_h_last, cur_sur_map_tokens)

                # Sur intent
                sur_state_for_intent = all_gt_states[:, t, :][~ego_mask]
                if sur_state_for_intent.size(-1) < 6:
                    sur_state_for_intent = torch.cat([sur_state_for_intent,
                        torch.zeros(num_sur, 6 - sur_state_for_intent.size(-1), device=device)], dim=-1)
                sur_intent_input = torch.cat([sur_state_for_intent, predicted_ego_delta, sur_map_ctx.detach()], dim=-1)
                if self.use_sur_intent:
                    z_local_sur, _ = self.sur_intent_codebook(sur_intent_input, self.gumbel_temperature)
                else:
                    z_local_sur = torch.zeros(num_sur, self.intent_dim, device=device)

                sur_lw_flat = cur_lw[~ego_mask]
                sur_sem_flat = cur_sem[~ego_mask]
                sur_gru_in = torch.cat([
                    sur_hist_ctx, sur_map_ctx, z_sur, z_local_sur, sur_lw_flat, sur_sem_flat
                ], dim=-1).unsqueeze(1)

                _, sur_gru_hidden = self.sur_decoder_gru(sur_gru_in, sur_gru_hidden)
                gt_hidden_snapshots.append(sur_gru_hidden.clone())

        # Replace snapshot[0] with the grad-carrying warmup result
        gt_hidden_snapshots[0] = sur_gru_hidden_init

        # ================================================================
        # Step C: Batched independent 1-step prediction (num_sur*FT batch)
        # ================================================================

        # Stack hidden snapshots: (FT+1) × (3, num_sur, D) → use [0..FT-1]
        # gt_hidden_snapshots[t] is the hidden state BEFORE step t
        all_hidden = torch.stack(gt_hidden_snapshots[:FT], dim=0)  # (FT, 3, num_sur, D)

        # ---- ALL flat tensors below use TIME-MAJOR layout ----
        # time-major: [t0a0, t0a1, ..., t0aN, t1a0, ...] = (FT, N, D).reshape(N*FT, D)
        # This matches all_hidden_gru = (3, FT, N, D).reshape(3, N*FT, D)

        # GT prev_state for bicycle model: (num_sur, FT, S) → time-major
        if self.output_bicycle:
            sur_prev_states = all_gt_states[:, :FT, :][~ego_mask]  # (num_sur, FT, S)
        else:
            sur_prev_states = all_gt_states_6d[:, :FT, :4][~ego_mask]  # (num_sur, FT, 4)
        sur_prev_flat = sur_prev_states.permute(1, 0, 2).reshape(num_sur * FT, -1)  # time-major

        # History Attention (batched with causal mask)
        # forward_batched expects (N, FT, D) and returns (N, FT, D) — agent-major internally
        queries = all_hidden[:, -1, :, :]  # (FT, num_sur, D)
        queries = queries.permute(1, 0, 2)  # (num_sur, FT, D) for forward_batched
        sur_hist_ctx_all = self.sur_history_attn.forward_batched(queries, gt_gcn_cache, FT)  # (num_sur, FT, D)
        # Convert to time-major for flat operations
        sur_hist_ctx_flat = sur_hist_ctx_all.permute(1, 0, 2).reshape(num_sur * FT, self.d_model)  # time-major

        # Auxiliary: ego prediction from sur history context
        predicted_ego_delta_flat = self.ego_pred_head(sur_hist_ctx_flat)  # (num_sur*FT, 2) time-major
        # Store per-timestep for loss (need agent-major → reshape as time-major then permute)
        predicted_ego_delta_per_t = predicted_ego_delta_flat.reshape(FT, num_sur, -1).permute(1, 0, 2)  # (num_sur, FT, ...)
        for t in range(FT):
            self._ego_pred_outputs.append(predicted_ego_delta_per_t[:, t, :].detach())

        # Map Attention (batched) — already time-major
        if gt_map_tokens_cache is not None:
            # Stack map tokens: (FT, num_sur, tokens, ch) → (num_sur*FT, tokens, ch) time-major
            sur_map_tokens_stacked = torch.stack(gt_map_tokens_cache, dim=0)  # (FT, num_sur, T, ch)
            sur_map_tokens_flat = sur_map_tokens_stacked.reshape(num_sur * FT, -1, self.map_token_ch)
        else:
            # Repeat same tokens for each timestep — time-major
            sur_map_tokens_flat = sur_map_tokens.unsqueeze(0).expand(FT, -1, -1, -1).reshape(
                num_sur * FT, -1, self.map_token_ch)

        # GRU hidden last layer for Map Attention queries — time-major
        # all_hidden: (FT, 3, num_sur, D), last GRU layer = [:, -1, :, :]
        gru_h_last_per_t = all_hidden[:, -1, :, :]  # (FT, num_sur, D)
        gru_h_last_flat = gru_h_last_per_t.reshape(num_sur * FT, self.d_model)  # time-major (no permute!)

        sur_map_ctx_flat, sur_map_attn_w_flat = self.sur_map_attn(gru_h_last_flat, sur_map_tokens_flat)
        # Store per-timestep
        sur_map_attn_w_per_t = sur_map_attn_w_flat.reshape(FT, num_sur, 1, -1).permute(1, 0, 2, 3)  # (num_sur, FT, 1, ...)
        for t in range(FT):
            self._sur_map_attn_weights_outputs.append(sur_map_attn_w_per_t[:, t, :, :])

        # Sur Intent (batched) — time-major
        sur_states_for_intent = all_gt_states[:, :FT, :][~ego_mask]  # (num_sur, FT, S)
        if sur_states_for_intent.size(-1) < 6:
            pad = torch.zeros(num_sur, FT, 6 - sur_states_for_intent.size(-1), device=device)
            sur_states_for_intent = torch.cat([sur_states_for_intent, pad], dim=-1)
        sur_states_intent_flat = sur_states_for_intent.permute(1, 0, 2).reshape(num_sur * FT, -1)  # time-major
        sur_intent_input_flat = torch.cat([
            sur_states_intent_flat,
            predicted_ego_delta_flat,
            sur_map_ctx_flat.detach()
        ], dim=-1)

        if self.use_sur_intent:
            z_local_sur_flat, sur_intent_weights_flat = self.sur_intent_codebook(
                sur_intent_input_flat, self.gumbel_temperature)
        else:
            z_local_sur_flat = torch.zeros(num_sur * FT, self.intent_dim, device=device)
            sur_intent_weights_flat = torch.zeros(num_sur * FT, self.num_intents, device=device)

        # Store per-timestep (time-major → agent-major for indexing)
        z_local_sur_per_t = z_local_sur_flat.reshape(FT, num_sur, -1).permute(1, 0, 2)  # (num_sur, FT, ...)
        sur_intent_w_per_t = sur_intent_weights_flat.reshape(FT, num_sur, -1).permute(1, 0, 2)
        for t in range(FT):
            self._z_local_sur_outputs.append(z_local_sur_per_t[:, t, :].detach())
            self._sur_intent_weights_outputs.append(sur_intent_w_per_t[:, t, :].detach())

        # Sur GRU 1-step (batched): all time-major (num_sur*FT, ...)
        sur_lw_flat = cur_lw[~ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)  # time-major
        sur_sem_flat = cur_sem[~ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)  # time-major
        z_sur_flat = z_sur.unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)  # time-major

        sur_gru_in_flat = torch.cat([
            sur_hist_ctx_flat, sur_map_ctx_flat, z_sur_flat, z_local_sur_flat,
            sur_lw_flat, sur_sem_flat
        ], dim=-1).unsqueeze(1)  # (num_sur*FT, 1, in_dim) time-major

        # Hidden for GRU: (3, FT, num_sur, D) → (3, num_sur*FT, D) time-major
        all_hidden_contiguous = all_hidden.permute(1, 0, 2, 3).contiguous()  # (3, FT, num_sur, D)
        all_hidden_gru = all_hidden_contiguous.reshape(3, num_sur * FT, self.d_model).contiguous()  # time-major

        sur_gru_out_flat, _ = self.sur_decoder_gru(sur_gru_in_flat, all_hidden_gru)
        sur_traj_step_flat = self.sur_output_head(sur_gru_out_flat[:, 0])  # (num_sur*FT, 2) time-major

        # Bicycle model (batched) — time-major
        sur_veh_len_flat = cur_veh_len[~ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_sur * FT, -1)
        sur_state_global_flat, _ = self._apply_dynamics_single(
            sur_traj_step_flat, sur_prev_flat, sur_veh_len_flat)

        # Reshape: time-major (num_sur*FT, 4) → (FT, num_sur, 4) → (num_sur, FT, 4)
        sur_traj_all = sur_state_global_flat.reshape(FT, num_sur, 4).permute(1, 0, 2).contiguous()

        return sur_traj_all  # (num_sur, FT, 4)

    def _ego_loop_decoder(self, scene_graph, z_ego, ego_mask, gt_future,
                              ego_map_tokens, cur_lw, cur_sem, cur_veh_len,
                              map_idx=None, map_env=None):
        """Ego loop: sur=GT fixed, ego=GT-cached independent 1-step prediction (BATCHED).

        All 3 steps are parallelized:
        Step A: Batch.from_data_list → 1 GCN call for all FT timesteps
        Step B: Batched input prep + sequential GRU for hidden snapshots (no_grad)
        Step C: (num_ego*FT) batched 1-step prediction
        """
        NA = ego_mask.size(0)
        FT = self.FT
        num_ego = int(ego_mask.sum())
        device = z_ego.device

        # ================================================================
        # Step A: Batched GCN — 12 graphs → 1 Batch call
        # ================================================================
        gt_state_t0 = scene_graph.past[:, -1, :]
        all_gt_states = torch.cat([gt_state_t0.unsqueeze(1), gt_future[:, :FT-1, :]], dim=1)  # (NA, FT, S)

        S = all_gt_states.size(-1)
        if S < 6:
            pad = torch.zeros(NA, FT, 6 - S, device=device)
            all_gt_states_6d = torch.cat([all_gt_states, pad], dim=-1)
        else:
            all_gt_states_6d = all_gt_states

        batched_graph = self._build_temporal_graph(
            scene_graph, all_gt_states_6d, cur_lw, cur_sem, FT)
        batched_ego_mask = ego_mask.repeat(FT)
        ego_gcn_all, _ = self.interaction_gcn(batched_graph, batched_ego_mask)
        gt_gcn_cache = ego_gcn_all.reshape(FT, num_ego, self.d_model).permute(1, 0, 2).contiguous()

        # Pre-compute GT map tokens if re-crop enabled
        gt_map_tokens_cache = None
        if self.map_recrop and map_idx is not None and map_env is not None:
            gt_map_tokens_cache = []
            for t in range(FT):
                gt_pos = all_gt_states_6d[:, t, :4]
                cur_map_tokens = self._recompute_map_tokens(
                    gt_pos, map_idx, map_env, scene_graph)
                gt_map_tokens_cache.append(cur_map_tokens[ego_mask])

        # ================================================================
        # Step B: Warmup (with grad) + GT forward pass for hidden snapshots (no_grad)
        # ================================================================
        ego_gru_hidden_init = self._warmup_gru_hidden(
            scene_graph, ego_mask, is_ego=True, gt_future=gt_future, target_t=0)

        with torch.no_grad():
            ego_gru_hidden = ego_gru_hidden_init.detach().clone()
            gt_hidden_snapshots = [ego_gru_hidden.clone()]
            gt_history_buffer = gt_gcn_cache  # (num_ego, FT, D)

            for t in range(FT):
                ego_gru_h_last = ego_gru_hidden[-1]
                ego_hist_ctx = self.ego_history_attn(ego_gru_h_last, gt_history_buffer, t)

                predicted_sur_delta = self.sur_pred_head(ego_hist_ctx)

                cur_ego_map_tokens = gt_map_tokens_cache[t] if gt_map_tokens_cache is not None else ego_map_tokens
                ego_map_ctx, _ = self.ego_map_attn(ego_gru_h_last, cur_ego_map_tokens)

                ego_state_for_intent = all_gt_states[:, t, :][ego_mask]
                if ego_state_for_intent.size(-1) < 6:
                    ego_state_for_intent = torch.cat([ego_state_for_intent,
                        torch.zeros(num_ego, 6 - ego_state_for_intent.size(-1), device=device)], dim=-1)

                intent_input = torch.cat([ego_state_for_intent, predicted_sur_delta, ego_map_ctx.detach()], dim=-1)
                z_local, _ = self.intent_codebook(intent_input, self.gumbel_temperature)
                if not self.use_ego_intent:
                    z_local = torch.zeros_like(z_local)

                ego_lw_flat = cur_lw[ego_mask]
                ego_sem_flat = cur_sem[ego_mask]
                ego_gru_in = torch.cat([
                    ego_hist_ctx, ego_map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
                ], dim=-1).unsqueeze(1)

                _, ego_gru_hidden = self.ego_decoder_gru(ego_gru_in, ego_gru_hidden)
                gt_hidden_snapshots.append(ego_gru_hidden.clone())

        gt_hidden_snapshots[0] = ego_gru_hidden_init

        # ================================================================
        # Step C: Batched independent 1-step prediction (num_ego*FT batch)
        # ================================================================

        # Stack hidden snapshots: use [0..FT-1] as initial hidden for each timestep
        all_hidden = torch.stack(gt_hidden_snapshots[:FT], dim=0)  # (FT, 3, num_ego, D)

        # ---- ALL flat tensors below use TIME-MAJOR layout ----
        # time-major: [t0a0, t0a1, ..., t0aN, t1a0, ...] = (FT, N, D).reshape(N*FT, D)
        # This matches all_hidden_gru = (3, FT, N, D).reshape(3, N*FT, D)

        # GT prev_state for bicycle model — time-major
        if self.output_bicycle:
            ego_prev_states = all_gt_states[:, :FT, :][ego_mask]  # (num_ego, FT, S)
        else:
            ego_prev_states = all_gt_states_6d[:, :FT, :4][ego_mask]
        ego_prev_flat = ego_prev_states.permute(1, 0, 2).reshape(num_ego * FT, -1)  # time-major

        # History Attention (batched with causal mask)
        # forward_batched expects (N, FT, D) and returns (N, FT, D) — agent-major internally
        queries = all_hidden[:, -1, :, :]  # (FT, num_ego, D)
        queries = queries.permute(1, 0, 2)  # (num_ego, FT, D) for forward_batched
        ego_hist_ctx_all = self.ego_history_attn.forward_batched(queries, gt_gcn_cache, FT)
        # Convert to time-major for flat operations
        ego_hist_ctx_flat = ego_hist_ctx_all.permute(1, 0, 2).reshape(num_ego * FT, self.d_model)  # time-major

        # Auxiliary: sur prediction
        predicted_sur_delta_flat = self.sur_pred_head(ego_hist_ctx_flat)  # time-major
        predicted_sur_delta_per_t = predicted_sur_delta_flat.reshape(FT, num_ego, -1).permute(1, 0, 2)  # (num_ego, FT, ...)
        for t in range(FT):
            self._sur_pred_outputs.append(predicted_sur_delta_per_t[:, t, :].detach())

        # Map Attention (batched) — already time-major
        if gt_map_tokens_cache is not None:
            ego_map_tokens_stacked = torch.stack(gt_map_tokens_cache, dim=0)  # (FT, num_ego, T, ch)
            ego_map_tokens_flat = ego_map_tokens_stacked.reshape(num_ego * FT, -1, self.map_token_ch)
        else:
            ego_map_tokens_flat = ego_map_tokens.unsqueeze(0).expand(FT, -1, -1, -1).reshape(
                num_ego * FT, -1, self.map_token_ch)

        # GRU hidden last layer — time-major (no permute!)
        gru_h_last_per_t = all_hidden[:, -1, :, :]  # (FT, num_ego, D)
        gru_h_last_flat = gru_h_last_per_t.reshape(num_ego * FT, self.d_model)  # time-major

        ego_map_ctx_flat, ego_map_attn_w_flat = self.ego_map_attn(gru_h_last_flat, ego_map_tokens_flat)
        ego_map_attn_w_per_t = ego_map_attn_w_flat.reshape(FT, num_ego, 1, -1).permute(1, 0, 2, 3)  # (num_ego, FT, 1, ...)
        for t in range(FT):
            self._ego_map_attn_weights_outputs.append(ego_map_attn_w_per_t[:, t, :, :])

        # Intent (batched) — time-major
        ego_states_for_intent = all_gt_states[:, :FT, :][ego_mask]  # (num_ego, FT, S)
        if ego_states_for_intent.size(-1) < 6:
            pad = torch.zeros(num_ego, FT, 6 - ego_states_for_intent.size(-1), device=device)
            ego_states_for_intent = torch.cat([ego_states_for_intent, pad], dim=-1)
        ego_states_intent_flat = ego_states_for_intent.permute(1, 0, 2).reshape(num_ego * FT, -1)  # time-major

        intent_input_flat = torch.cat([
            ego_states_intent_flat,
            predicted_sur_delta_flat,
            ego_map_ctx_flat.detach()
        ], dim=-1)

        z_local_flat, intent_weights_flat = self.intent_codebook(intent_input_flat, self.gumbel_temperature)
        if not self.use_ego_intent:
            z_local_flat = torch.zeros_like(z_local_flat)

        z_local_per_t = z_local_flat.reshape(FT, num_ego, -1).permute(1, 0, 2)  # (num_ego, FT, ...)
        intent_w_per_t = intent_weights_flat.reshape(FT, num_ego, -1).permute(1, 0, 2)
        for t in range(FT):
            self._z_local_outputs.append(z_local_per_t[:, t, :].detach())
            self._intent_weights_outputs.append(intent_w_per_t[:, t, :].detach())

        # Ego GRU 1-step (batched) — all time-major
        ego_lw_flat = cur_lw[ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_ego * FT, -1)  # time-major
        ego_sem_flat = cur_sem[ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_ego * FT, -1)  # time-major
        z_ego_flat = z_ego.unsqueeze(0).expand(FT, -1, -1).reshape(num_ego * FT, -1)  # time-major

        ego_gru_in_flat = torch.cat([
            ego_hist_ctx_flat, ego_map_ctx_flat, z_ego_flat, z_local_flat,
            ego_lw_flat, ego_sem_flat
        ], dim=-1).unsqueeze(1)  # (num_ego*FT, 1, in_dim) time-major

        # Hidden for GRU: (3, FT, num_ego, D) → (3, num_ego*FT, D) time-major
        all_hidden_contiguous = all_hidden.permute(1, 0, 2, 3).contiguous()  # (3, FT, num_ego, D)
        all_hidden_gru = all_hidden_contiguous.reshape(3, num_ego * FT, self.d_model).contiguous()  # time-major

        ego_gru_out_flat, _ = self.ego_decoder_gru(ego_gru_in_flat, all_hidden_gru)
        ego_traj_out_flat = self.ego_output_head(ego_gru_out_flat[:, 0])

        # Bicycle model (batched) — time-major
        ego_veh_len_flat = cur_veh_len[ego_mask].unsqueeze(0).expand(FT, -1, -1).reshape(num_ego * FT, -1)
        ego_state_global_flat, _ = self._apply_dynamics_single(
            ego_traj_out_flat, ego_prev_flat, ego_veh_len_flat)

        # Reshape: time-major (num_ego*FT, 4) → (FT, num_ego, 4) → (num_ego, FT, 4)
        ego_traj_all = ego_state_global_flat.reshape(FT, num_ego, 4).permute(1, 0, 2).contiguous()

        return ego_traj_all  # (num_ego, FT, 4)

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
    # Temporal GCN batching helper
    # ============================================================

    def _build_temporal_graph(self, scene_graph, all_states_6d, cur_lw, cur_sem, FT):
        """
        Manually build a batched graph for FT timesteps.
        Each timestep reuses the same edge topology but with different node features/positions.

        Unlike Batch.from_data_list, this preserves the original sub-graph structure
        (ptr/batch from DataLoader) by correctly repeating and offsetting them.

        :param scene_graph: original batched graph from DataLoader
        :param all_states_6d: (NA, FT, 6) GT states for all timesteps
        :param cur_lw: (NA, 2) vehicle length/width
        :param cur_sem: (NA, NC) semantic class
        :param FT: number of future timesteps
        :return: batched_graph (Data-like), batched_ego_mask
        """
        NA = all_states_6d.size(0)
        device = all_states_6d.device

        # x: (NA*FT, feat_dim), pos: (NA*FT, 4), sem: (NA*FT, NC)
        # Layout: time-major — [t0_agent0, t0_agent1, ..., t0_agentNA-1, t1_agent0, ...]
        # This matches edge_index offset: t0 nodes = 0..NA-1, t1 nodes = NA..2*NA-1
        gcn_in_list = []
        for t in range(FT):
            gcn_in_t = torch.cat([all_states_6d[:, t, :], cur_lw, cur_sem], dim=-1)
            gcn_in_list.append(gcn_in_t)
        x_cat = torch.cat(gcn_in_list, dim=0)  # (NA*FT, feat_dim) — time-major
        # pos & sem must also be time-major: permute (NA,FT,...) → (FT,NA,...) then reshape
        pos_cat = all_states_6d[:, :, :4].permute(1, 0, 2).reshape(NA * FT, 4)  # (NA*FT, 4)
        sem_cat = cur_sem.unsqueeze(0).expand(FT, -1, -1).reshape(NA * FT, -1)  # already time-major

        # edge_index: repeat with +NA*t offset per timestep
        orig_edge_index = scene_graph.edge_index  # (2, E)
        E = orig_edge_index.size(1)
        edge_list = []
        for t in range(FT):
            edge_list.append(orig_edge_index + NA * t)
        edge_cat = torch.cat(edge_list, dim=1)  # (2, E*FT)

        # batch: repeat with offset per timestep
        orig_batch = scene_graph.batch  # (NA,)
        B = orig_batch.max().item() + 1  # number of scenes in original batch
        batch_list = []
        for t in range(FT):
            batch_list.append(orig_batch + B * t)
        batch_cat = torch.cat(batch_list, dim=0)  # (NA*FT,)

        # ptr: repeat with offset per timestep
        orig_ptr = scene_graph.ptr  # (B+1,)
        ptr_list = []
        for t in range(FT):
            ptr_list.append(orig_ptr[:-1] + NA * t)
        ptr_list.append(torch.tensor([NA * FT], device=device))
        ptr_cat = torch.cat(ptr_list, dim=0)  # (B*FT + 1,)

        # Build Data object (not Batch — just a plain Data with batch/ptr)
        batched_graph = Data(x=x_cat, edge_index=edge_cat, pos=pos_cat, sem=sem_cat)
        batched_graph.batch = batch_cat
        batched_graph.ptr = ptr_cat

        return batched_graph

    # ============================================================
    # Warmup & utility methods
    # ============================================================

    def _warmup_gru_hidden(self, scene_graph, ego_mask, is_ego, gt_future=None, target_t=0):
        """
        Warm-up GRU hidden state using past (+ GT future if TF mode).
        Uses interaction_gcn (same as decode loop) for consistency.

        :param is_ego: True for ego warmup, False for sur warmup
        :return: gru_hidden (3, N_agent, D)
        """
        NA = ego_mask.size(0)
        device = scene_graph.past.device
        cur_lw = scene_graph.lw
        cur_sem = scene_graph.sem

        gcn_features = []

        for pt in range(self.PT):
            t_window = target_t - self.PT + pt

            if t_window < 0:
                cur_state = scene_graph.past[:, self.PT + t_window, :]
            else:
                cur_state = gt_future[:, t_window, :]

            cur_state_6d = cur_state if cur_state.size(-1) >= 6 else \
                torch.cat([cur_state, torch.zeros(NA, 6 - cur_state.size(-1), device=device)], dim=-1)
            gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)
            scene_graph.x = gcn_in
            scene_graph.pos = cur_state_6d[:, :4]

            ego_feat_t, sur_feat_t = self.interaction_gcn(scene_graph, ego_mask)
            if is_ego:
                gcn_features.append(ego_feat_t)
            else:
                gcn_features.append(sur_feat_t)

        sequence = torch.stack(gcn_features, dim=1)  # (N_agent, PT, D)

        if is_ego:
            _, gru_hidden = self.ego_warmup_gru(sequence)
        else:
            _, gru_hidden = self.sur_warmup_gru(sequence)

        return gru_hidden

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
        :param mult_samp: if True, pos is (NA*NS, 4), expand batch accordingly
        :param NS: number of samples (required if mult_samp=True)
        :return: map_tokens (NA, num_tokens, ch) or (NA*NS, num_tokens, ch)
        """
        pos_unnorm = self.normalizer.unnormalize(pos_normalized)
        if mult_samp and NS is not None:
            # scene_graph.batch is (NA,), expand to (NA*NS,)
            mapixes = map_idx[scene_graph.batch]  # (NA,)
            mapixes = mapixes.unsqueeze(1).expand(-1, NS).reshape(-1)  # (NA*NS,)
        else:
            mapixes = map_idx[scene_graph.batch]
        map_obs = map_env.get_map_crop_pos(pos_unnorm, mapixes).to(torch.float)
        map_early = self.map_conv_early(map_obs)
        map_tokens = map_early.flatten(2).permute(0, 2, 1)
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
        Freeze all EXCEPT ego decoder components + intent codebook.

        FROZEN: encoder, z_global, sur decoder, interaction GCN, map CNN, sur history/map attn
        TRAINABLE: ego history attn, ego map attn, ego GRU, ego output head,
                   intent codebook, intent CE head, sur pred head,
                   ego warmup GRU
        """
        frozen_modules = [
            # Encoder
            self.latent_prior_net, self.latent_posterior_net,
            self.step_feature_extractor, self.temporal_gcn_encoder,
            self.prior_temporal_gru, self.positional_encoding, self.transformer_encoder,
            # Map
            self.map_conv_early, self.map_conv_late, self.map_feature,
            # Interaction GCN
            self.interaction_gcn,
            # Sur decoder
            self.sur_history_attn, self.sur_map_attn,
            self.sur_decoder_gru, self.sur_output_head,
            self.sur_warmup_gru,
            # Sur intent codebook (freeze with sur decoder)
            self.sur_intent_codebook, self.sur_intent_ce_head,
            # Ego pred head (sur→ego, Phase 1 only)
            self.ego_pred_head,
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        num_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        Logger.log(f'Frozen for fine-tuning: {num_frozen} params frozen, {num_trainable} params trainable')

    # ============================================================
    # Analysis getters
    # ============================================================

    def get_z_local(self):
        """Get z_local from last forward pass. Flat list of FT tensors (num_ego, intent_dim)."""
        if not hasattr(self, '_z_local_outputs') or len(self._z_local_outputs) == 0:
            return None
        return self._z_local_outputs

    def get_z_local_stacked(self):
        """Get z_local stacked as (FT, num_ego, intent_dim). AR mode only."""
        raw = self.get_z_local()
        if raw is None:
            return None
        return torch.stack(raw, dim=0)

    def get_z_local_mean(self):
        """Backward-compatible alias."""
        return self.get_z_local_stacked()

    def get_z_local_var(self):
        """z_local is discrete — no variance."""
        return None

    def get_intent_weights(self):
        """Get intent selection weights. Flat list of FT tensors (num_ego, K)."""
        if not hasattr(self, '_intent_weights_outputs') or len(self._intent_weights_outputs) == 0:
            return None
        return self._intent_weights_outputs

    def get_ego_map_attn_weights(self):
        """Get ego map attention weights (with gradient). List of (num_ego, 1, num_tokens)."""
        if not hasattr(self, '_ego_map_attn_weights_outputs') or len(self._ego_map_attn_weights_outputs) == 0:
            return None
        return self._ego_map_attn_weights_outputs

    def get_sur_map_attn_weights(self):
        """Get sur map attention weights (with gradient). List of (num_sur, 1, num_tokens)."""
        if not hasattr(self, '_sur_map_attn_weights_outputs') or len(self._sur_map_attn_weights_outputs) == 0:
            return None
        return self._sur_map_attn_weights_outputs

    def get_map_attn_weights(self):
        """Get ego map attention weights (detached, for analysis)."""
        w = self.get_ego_map_attn_weights()
        if w is None:
            return None
        return [x.detach() for x in w]

    def get_attn_weights(self):
        return self.get_map_attn_weights()

    def get_sur_pred_outputs(self):
        """Get predicted sur deltas. List of (num_ego, sur_pred_dim) per timestep."""
        if not hasattr(self, '_sur_pred_outputs') or len(self._sur_pred_outputs) == 0:
            return None
        return self._sur_pred_outputs

    def get_ego_pred_outputs(self):
        """Get predicted ego deltas. List of (num_sur, sur_pred_dim) per timestep."""
        if not hasattr(self, '_ego_pred_outputs') or len(self._ego_pred_outputs) == 0:
            return None
        return self._ego_pred_outputs

    def get_sur_z_local(self):
        """Get sur z_local outputs. Flat list of FT tensors (num_sur, intent_dim)."""
        if not hasattr(self, '_z_local_sur_outputs') or len(self._z_local_sur_outputs) == 0:
            return None
        return self._z_local_sur_outputs

    def get_sur_intent_weights(self):
        """Get sur intent selection weights. Flat list of FT tensors (num_sur, K)."""
        if not hasattr(self, '_sur_intent_weights_outputs') or len(self._sur_intent_weights_outputs) == 0:
            return None
        return self._sur_intent_weights_outputs

    def get_z_local_sparsity_loss(self, target_sparsity=0.1):
        """Backward-compatible sparsity loss (unused in redesign but kept for interface)."""
        z_local = self.get_z_local()
        if z_local is None:
            return torch.tensor(0.0)
        l1_loss = torch.mean(torch.abs(z_local))
        return l1_loss
