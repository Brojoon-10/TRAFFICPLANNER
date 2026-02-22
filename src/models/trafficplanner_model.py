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

    def forward(self, situation, temperature=1.0, phase=1):
        """
        :param situation: (N_ego, input_dim) — ego_state + predicted_sur_delta + map_ctx(detached)
        :param temperature: Gumbel-Softmax temperature
        :param phase: 1=zero output (inactive), 2=active
        :return: (z_local, intent_weights)
            z_local: (N_ego, intent_dim)
            intent_weights: (N_ego, K)
        """
        N = situation.size(0)
        device = situation.device

        if phase == 1:
            z_local = torch.zeros(N, self.intent_dim, device=device)
            intent_weights = torch.zeros(N, self.num_intents, device=device)
            return z_local, intent_weights

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
                 tf_max_annealing_epoch=1600,
                 tf_init_segment_len=1,
                 # New redesign params
                 num_intents=8,           # K for intent codebook
                 hist_attn_nhead=4,
                 map_attn_nhead=4,
                 sur_pred_dim=2,          # predicted sur delta dim (dx, dy)
                 map_recrop=False,        # re-crop map tokens every decode step
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
        # Teacher Forcing parameters
        #
        self.tf_max_annealing_epoch = tf_max_annealing_epoch
        self.tf_init_segment_len = tf_init_segment_len
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

        # 4. Intent Codebook (ego only)
        intent_input_dim = self.state_size + self.sur_pred_dim + self.d_model  # 6 + 2 + 64 = 72
        self.intent_codebook = IntentCodebook(
            num_intents=num_intents,
            intent_dim=self.intent_dim,
            input_dim=intent_input_dim,
        )

        # 5. Ego decoder GRU
        # Input: hist_ctx(D) + map_ctx(D) + z_global(z_size) + z_local(intent_dim) + lw(2) + sem(NC)
        ego_gru_in_size = self.d_model + self.d_model + self.z_size + self.intent_dim + self.att_feat_size + self.NC
        self.ego_decoder_gru = nn.GRU(
            ego_gru_in_size, self.d_model, 3, batch_first=True,
        )
        self.ego_output_head = nn.Linear(self.d_model, self.traj_out_size)

        # 6. Sur decoder GRU
        # Input: hist_ctx(D) + map_ctx(D) + z_global(z_size) + lw(2) + sem(NC)
        sur_gru_in_size = self.d_model + self.d_model + self.z_size + self.att_feat_size + self.NC
        self.sur_decoder_gru = nn.GRU(
            sur_gru_in_size, self.d_model, 3, batch_first=True,
        )
        self.sur_output_head = nn.Linear(self.d_model, self.traj_out_size)

        # 7. Auxiliary loss heads (all Linear — no MLP, force modules to encode directly)
        self.sur_pred_head = nn.Linear(self.d_model, self.sur_pred_dim)  # (B) ego→sur delta
        self.ego_pred_head = nn.Linear(self.d_model, self.sur_pred_dim)  # (C) sur→ego delta
        self.intent_ce_head = nn.Linear(self.intent_dim, self.num_intents)  # (A) intent → num_intents classes

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

        # Ablation flag
        self.use_z_local = True


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

    def get_tf_segment_len(self, current_epoch):
        """Cosine annealing for TF segment length: tf_init_segment_len → FT."""
        if current_epoch >= self.tf_max_annealing_epoch:
            return self.FT
        progress = current_epoch / self.tf_max_annealing_epoch
        cos_out = 1 - math.cos(math.pi / 2 * progress)
        segment_len = self.tf_init_segment_len + (self.FT - self.tf_init_segment_len) * cos_out
        return int(segment_len)

    # ============================================================
    # Forward / Reconstruct / Sample — public API (unchanged interface)
    # ============================================================

    def forward(self, scene_graph, map_idx, map_env,
                use_post_mean=False,
                future_sample=False,
                teacher_forcing=False,
                tf_segment_len=None,
                current_epoch=0):
        # Compute segment length using cosine annealing if not explicitly provided
        if tf_segment_len is None and teacher_forcing:
            tf_segment_len = self.get_tf_segment_len(current_epoch)
        elif tf_segment_len is None:
            tf_segment_len = self.FT

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
                                   teacher_forcing=teacher_forcing,
                                   tf_segment_len=tf_segment_len)

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
                teacher_forcing=False, tf_segment_len=3,
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
                                                map_idx, map_env, map_tokens, tf_segment_len)
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
            if self.map_recrop and not mult_samp:
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
                intent_input, temperature=self.gumbel_temperature, phase=self.phase)

            if not self.use_z_local:
                z_local = torch.zeros_like(z_local)

            self._z_local_outputs.append(z_local.detach())
            self._intent_weights_outputs.append(intent_weights.detach())

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
                sur_hist_ctx, sur_map_ctx, z_sur, sur_lw_flat, sur_sem_flat
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
                                 map_idx, map_env, map_tokens, tf_segment_len=3):
        """
        Dual-loop Teacher Forcing decoder (Redesign).

        Loop 1: Sur (ego=GT, sur=autoregressive with new decoder)
        Loop 2: Ego (sur=GT, ego=TF segmented with new decoder)

        Returns: list of FT segments for loss computation
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
        # Loop 1: Sur prediction (ego=GT, sur=autoregressive)
        # ====================================================================
        sur_traj_all = self._sur_loop_decoder(
            scene_graph, z_sur, ego_mask, gt_future,
            sur_map_tokens, cur_lw, cur_sem, cur_veh_len,
            map_idx, map_env)

        # ====================================================================
        # Loop 2: Ego prediction (sur=GT, ego=TF segmented)
        # ====================================================================
        ego_segments = self._ego_loop_decoder(
            scene_graph, z_ego, ego_mask, gt_future,
            ego_map_tokens, cur_lw, cur_sem, cur_veh_len, tf_segment_len,
            map_idx, map_env)

        # ====================================================================
        # Merge
        # ====================================================================
        all_segment_preds = []
        for seg_idx, ego_seg in enumerate(ego_segments):
            segment_preds = []
            for step, ego_pred in enumerate(ego_seg):
                actual_t = seg_idx + step
                if actual_t >= FT:
                    break
                full_pred = torch.zeros(NA, 4, device=device)
                full_pred[ego_mask] = ego_pred
                full_pred[~ego_mask] = sur_traj_all[:, actual_t, :]
                segment_preds.append(full_pred)
            all_segment_preds.append(segment_preds)

        return all_segment_preds

    def _sur_loop_decoder(self, scene_graph, z_sur, ego_mask, gt_future,
                              sur_map_tokens, cur_lw, cur_sem, cur_veh_len,
                              map_idx=None, map_env=None):
        """Sur loop: ego=GT fixed, sur=autoregressive with new symmetric decoder."""
        NA = ego_mask.size(0)
        FT = self.FT
        num_ego = int(ego_mask.sum())
        num_sur = NA - num_ego
        device = z_sur.device

        # Pre-allocate
        sur_history_buffer = torch.zeros(num_sur, FT, self.d_model, device=device)
        sur_traj_all = torch.zeros(num_sur, FT, 4, device=device)

        # Init GRU hidden via warmup
        sur_gru_hidden = self._warmup_gru_hidden(scene_graph, ego_mask, is_ego=False, gt_future=None, target_t=0)

        # Init prev_state (full NA, needed for GCN)
        sur_prev_state = scene_graph.past[:, -1, :] if self.output_bicycle else scene_graph.past[:, -1, :4]

        for t in range(FT):
            # Build combined position: ego=GT, sur=predicted/initial
            combined_state = sur_prev_state.clone()
            if t == 0:
                ego_gt_state = scene_graph.past[:, -1, :][ego_mask]
            else:
                ego_gt_state = gt_future[:, t - 1, :][ego_mask]
            combined_state[ego_mask] = ego_gt_state

            # GCN on combined state
            cur_state_6d = combined_state if combined_state.size(-1) >= 6 else \
                torch.cat([combined_state, torch.zeros(NA, 6 - combined_state.size(-1), device=device)], dim=-1)
            gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)
            scene_graph.x = gcn_in
            scene_graph.pos = cur_state_6d[:, :4]
            _, sur_gcn_feat = self.interaction_gcn(scene_graph, ego_mask)

            # History buffer
            sur_history_buffer[:, t, :] = sur_gcn_feat

            # History Attention
            sur_gru_h_last = sur_gru_hidden[-1]
            sur_hist_ctx = self.sur_history_attn(sur_gru_h_last, sur_history_buffer, t)

            # Auxiliary: ego prediction from sur history context
            predicted_ego_delta = self.ego_pred_head(sur_hist_ctx)
            self._ego_pred_outputs.append(predicted_ego_delta.detach())

            # Re-crop map tokens at current positions (if enabled)
            if self.map_recrop and map_idx is not None and map_env is not None:
                cur_map_tokens = self._recompute_map_tokens(
                    cur_state_6d[:, :4], map_idx, map_env, scene_graph)
                sur_map_tokens = cur_map_tokens[~ego_mask]

            # Map Attention
            sur_map_ctx, sur_map_attn_w = self.sur_map_attn(sur_gru_h_last, sur_map_tokens)
            self._sur_map_attn_weights_outputs.append(sur_map_attn_w)

            # Sur GRU
            sur_lw_flat = cur_lw[~ego_mask]
            sur_sem_flat = cur_sem[~ego_mask]
            sur_gru_in = torch.cat([sur_hist_ctx, sur_map_ctx, z_sur, sur_lw_flat, sur_sem_flat], dim=-1).unsqueeze(1)
            sur_gru_out, sur_gru_hidden = self.sur_decoder_gru(sur_gru_in, sur_gru_hidden)
            sur_traj_step = self.sur_output_head(sur_gru_out[:, 0])

            # Bicycle model for sur
            sur_prev = sur_prev_state[~ego_mask]
            sur_state_global, sur_bike_state = self._apply_dynamics_single(
                sur_traj_step, sur_prev, cur_veh_len[~ego_mask])

            sur_traj_all[:, t, :] = sur_state_global

            # Update states
            if t < FT - 1:
                new_prev = sur_prev_state.clone()
                if self.output_bicycle and sur_bike_state is not None:
                    new_prev[~ego_mask] = sur_bike_state
                else:
                    # Pad to 6d for next step
                    if sur_state_global.size(-1) < 6:
                        new_prev[~ego_mask, :4] = sur_state_global
                    else:
                        new_prev[~ego_mask] = sur_state_global
                ego_gt_next = gt_future[:, t, :][ego_mask]
                new_prev[ego_mask] = ego_gt_next
                sur_prev_state = new_prev

        return sur_traj_all  # (num_sur, FT, 4)

    def _ego_loop_decoder(self, scene_graph, z_ego, ego_mask, gt_future,
                              ego_map_tokens, cur_lw, cur_sem, cur_veh_len, tf_segment_len,
                              map_idx=None, map_env=None):
        """Ego loop: sur=GT fixed, ego=TF segmented (sliding window).

        Sliding window: segment_len=3, FT=12 → segments at target_t=0,1,...,9
        Each segment: step 0 uses GT GCN cache, steps 1+ use predicted position.

        구조:
        Step A: GT GCN 사전계산 (future FT)
        Step B: GT forward pass → 매 시점 decoder GRU hidden 스냅샷 저장 (no_grad)
        Step C: Sliding window — segment 경계에서 GT 상태 복원, 내에서만 예측 누적

        Segment 경계 리셋: ego position, history buffer, GRU hidden 모두 GT 복원
        Segment 내부: 예측 기반 position/history/hidden 누적
        """
        NA = ego_mask.size(0)
        FT = self.FT
        num_ego = int(ego_mask.sum())
        device = z_ego.device

        # ================================================================
        # Step A: Pre-compute GT GCN features for future FT timesteps
        # ================================================================
        gt_gcn_cache = torch.zeros(num_ego, FT, self.d_model, device=device)
        for t in range(FT):
            if t == 0:
                gt_state = scene_graph.past[:, -1, :]
            else:
                gt_state = gt_future[:, t - 1, :]

            cur_state_6d = gt_state if gt_state.size(-1) >= 6 else \
                torch.cat([gt_state, torch.zeros(NA, 6 - gt_state.size(-1), device=device)], dim=-1)
            gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)
            scene_graph.x = gcn_in
            scene_graph.pos = cur_state_6d[:, :4]
            ego_gcn_feat, _ = self.interaction_gcn(scene_graph, ego_mask)
            gt_gcn_cache[:, t, :] = ego_gcn_feat

        # Also pre-compute GT map tokens if re-crop enabled
        gt_map_tokens_cache = None
        if self.map_recrop and map_idx is not None and map_env is not None:
            gt_map_tokens_cache = []
            for t in range(FT):
                if t == 0:
                    gt_pos = scene_graph.past[:, -1, :4]
                else:
                    gt_pos = gt_future[:, t - 1, :4]
                cur_map_tokens = self._recompute_map_tokens(
                    gt_pos, map_idx, map_env, scene_graph)
                gt_map_tokens_cache.append(cur_map_tokens[ego_mask])

        # ================================================================
        # Step B: GT forward pass → hidden 스냅샷 저장 (no_grad)
        # warmup 1회 후, 매 스텝 GT 기반 full pipeline으로 decoder GRU hidden 누적
        # 각 시점의 hidden을 저장해두고, segment 시작 시 복원
        # loss에 직접 기여하지 않으므로 gradient 불필요
        # ================================================================
        with torch.no_grad():
            ego_gru_hidden = self._warmup_gru_hidden(
                scene_graph, ego_mask, is_ego=True, gt_future=gt_future, target_t=0)

            gt_history_buffer = torch.zeros(num_ego, FT, self.d_model, device=device)
            gt_hidden_snapshots = [ego_gru_hidden.clone()]

            for t in range(FT):
                gt_history_buffer[:, t, :] = gt_gcn_cache[:, t, :]

                ego_gru_h_last = ego_gru_hidden[-1]
                ego_hist_ctx = self.ego_history_attn(ego_gru_h_last, gt_history_buffer, t)

                predicted_sur_delta = self.sur_pred_head(ego_hist_ctx)

                cur_ego_map_tokens = gt_map_tokens_cache[t] if gt_map_tokens_cache is not None else ego_map_tokens
                ego_map_ctx, _ = self.ego_map_attn(ego_gru_h_last, cur_ego_map_tokens)

                if t == 0:
                    ego_state_for_intent = scene_graph.past[:, -1, :][ego_mask]
                else:
                    ego_state_for_intent = gt_future[:, t - 1, :][ego_mask]
                if ego_state_for_intent.size(-1) < 6:
                    ego_state_for_intent = torch.cat([ego_state_for_intent,
                        torch.zeros(num_ego, 6 - ego_state_for_intent.size(-1), device=device)], dim=-1)

                intent_input = torch.cat([ego_state_for_intent, predicted_sur_delta, ego_map_ctx.detach()], dim=-1)
                z_local, _ = self.intent_codebook(intent_input, self.gumbel_temperature, self.phase)
                if not self.use_z_local:
                    z_local = torch.zeros_like(z_local)

                ego_lw_flat = cur_lw[ego_mask]
                ego_sem_flat = cur_sem[ego_mask]
                ego_gru_in = torch.cat([
                    ego_hist_ctx, ego_map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
                ], dim=-1).unsqueeze(1)

                _, ego_gru_hidden = self.ego_decoder_gru(ego_gru_in, ego_gru_hidden)
                gt_hidden_snapshots.append(ego_gru_hidden.clone())

        # ================================================================
        # Step C: Sliding window segments
        # segment 시작 시 GT hidden/history/position 복원
        # segment 내에서만 예측 기반 누적
        # ================================================================
        ego_segments = []
        ego_history_buffer = torch.zeros(num_ego, FT, self.d_model, device=device)

        for target_t in range(FT):
            if target_t + tf_segment_len > FT:
                break

            # TF: segment 경계에서 모든 상태를 GT 기반으로 리셋
            # 1) ego position → GT
            if target_t == 0:
                ego_prev_state = scene_graph.past[:, -1, :][ego_mask] if self.output_bicycle else scene_graph.past[:, -1, :4][ego_mask]
            else:
                ego_prev_state = gt_future[:, target_t - 1, :][ego_mask] if self.output_bicycle else gt_future[:, target_t - 1, :4][ego_mask]

            # 2) history buffer → GT GCN 캐시로 교정 (0~target_t)
            ego_history_buffer[:, :target_t + 1, :] = gt_gcn_cache[:, :target_t + 1, :]

            # 3) GRU hidden → GT 스냅샷 복원
            ego_gru_hidden = gt_hidden_snapshots[target_t].clone()

            segment_preds = []
            seg_sur_pred = []
            seg_map_attn = []
            seg_z_local = []
            seg_intent_w = []

            for step in range(tf_segment_len):
                actual_t = target_t + step
                if actual_t >= FT:
                    break

                if step == 0:
                    # Step 0: use cached GT GCN feature
                    ego_gcn_feat = gt_gcn_cache[:, actual_t, :]
                    cur_ego_map_tokens = gt_map_tokens_cache[actual_t] if gt_map_tokens_cache is not None else ego_map_tokens
                else:
                    # Step 1+: ego at predicted position, sur at GT → new GCN
                    sur_gt_state = gt_future[:, actual_t - 1, :][~ego_mask]
                    combined_state = torch.zeros(NA, ego_prev_state.size(-1), device=device)
                    combined_state[ego_mask] = ego_prev_state
                    combined_state[~ego_mask] = sur_gt_state
                    cur_state_6d = combined_state if combined_state.size(-1) >= 6 else \
                        torch.cat([combined_state, torch.zeros(NA, 6 - combined_state.size(-1), device=device)], dim=-1)
                    gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)
                    scene_graph.x = gcn_in
                    scene_graph.pos = cur_state_6d[:, :4]
                    ego_gcn_feat, _ = self.interaction_gcn(scene_graph, ego_mask)

                    # Re-crop map tokens at predicted position (if enabled)
                    if self.map_recrop and map_idx is not None and map_env is not None:
                        cur_map_tokens = self._recompute_map_tokens(
                            cur_state_6d[:, :4], map_idx, map_env, scene_graph)
                        cur_ego_map_tokens = cur_map_tokens[ego_mask]
                    else:
                        cur_ego_map_tokens = ego_map_tokens

                # History buffer
                ego_history_buffer[:, actual_t, :] = ego_gcn_feat

                # History Attention
                ego_gru_h_last = ego_gru_hidden[-1]
                ego_hist_ctx = self.ego_history_attn(ego_gru_h_last, ego_history_buffer, actual_t)

                # Auxiliary: sur prediction
                predicted_sur_delta = self.sur_pred_head(ego_hist_ctx)
                seg_sur_pred.append(predicted_sur_delta.detach())

                # Map Attention
                ego_map_ctx, ego_map_attn_w = self.ego_map_attn(ego_gru_h_last, cur_ego_map_tokens)
                seg_map_attn.append(ego_map_attn_w)

                # Intent
                if step == 0:
                    if actual_t == 0:
                        ego_state_for_intent = scene_graph.past[:, -1, :][ego_mask]
                    else:
                        ego_state_for_intent = gt_future[:, actual_t - 1, :][ego_mask]
                else:
                    ego_state_for_intent = ego_prev_state

                if ego_state_for_intent.size(-1) < 6:
                    ego_state_for_intent = torch.cat([ego_state_for_intent,
                        torch.zeros(num_ego, 6 - ego_state_for_intent.size(-1), device=device)], dim=-1)

                intent_input = torch.cat([ego_state_for_intent, predicted_sur_delta, ego_map_ctx.detach()], dim=-1)
                z_local, intent_weights = self.intent_codebook(intent_input, self.gumbel_temperature, self.phase)

                if not self.use_z_local:
                    z_local = torch.zeros_like(z_local)

                seg_z_local.append(z_local.detach())
                seg_intent_w.append(intent_weights.detach())

                # Ego GRU
                ego_lw_flat = cur_lw[ego_mask]
                ego_sem_flat = cur_sem[ego_mask]
                ego_gru_in = torch.cat([
                    ego_hist_ctx, ego_map_ctx, z_ego, z_local, ego_lw_flat, ego_sem_flat
                ], dim=-1).unsqueeze(1)

                ego_gru_out, ego_gru_hidden = self.ego_decoder_gru(ego_gru_in, ego_gru_hidden)
                ego_traj_out = self.ego_output_head(ego_gru_out[:, 0])

                # Bicycle model
                ego_state_global, ego_bike_state = self._apply_dynamics_single(
                    ego_traj_out, ego_prev_state, cur_veh_len[ego_mask])

                segment_preds.append(ego_state_global)

                # Update ego state for next step within segment
                if step < tf_segment_len - 1 and actual_t < FT - 1:
                    if self.output_bicycle and ego_bike_state is not None:
                        ego_prev_state = ego_bike_state
                    else:
                        ego_prev_state = ego_state_global

            ego_segments.append(segment_preds)
            self._sur_pred_outputs.append(seg_sur_pred)
            self._ego_map_attn_weights_outputs.append(seg_map_attn)
            self._z_local_outputs.append(seg_z_local)
            self._intent_weights_outputs.append(seg_intent_w)

        return ego_segments

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

    def _recompute_map_tokens(self, pos_normalized, map_idx, map_env, scene_graph):
        """
        Re-crop and re-encode map tokens at given normalized positions.

        :param pos_normalized: (NA, 4) normalized positions (x, y, hx, hy)
        :param map_idx: (B,) map index per batch
        :param map_env: map environment for cropping
        :param scene_graph: scene graph (for .batch attribute)
        :return: map_tokens (NA, num_tokens, map_token_ch)
        """
        pos_unnorm = self.normalizer.unnormalize(pos_normalized)
        mapixes = map_idx[scene_graph.batch]
        map_obs = map_env.get_map_crop_pos(pos_unnorm, mapixes).to(torch.float)  # (NA, C, H, W)
        map_early = self.map_conv_early(map_obs)  # (NA, ch, H', W')
        map_tokens = map_early.flatten(2).permute(0, 2, 1)  # (NA, num_tokens, ch)
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
        """Get z_local from last forward pass.
        AR mode: list of FT tensors (num_ego, intent_dim)
        TF mode: list of segments, each segment is list of tensors
        """
        if not hasattr(self, '_z_local_outputs') or len(self._z_local_outputs) == 0:
            return None
        return self._z_local_outputs

    def get_z_local_stacked(self):
        """Get z_local stacked as (FT, num_ego, intent_dim). AR mode only."""
        raw = self.get_z_local()
        if raw is None:
            return None
        if isinstance(raw[0], list):
            return None  # TF mode — cannot stack
        return torch.stack(raw, dim=0)

    def get_z_local_mean(self):
        """Backward-compatible alias."""
        return self.get_z_local_stacked()

    def get_z_local_var(self):
        """z_local is discrete — no variance."""
        return None

    def get_intent_weights(self):
        """Get intent selection weights.
        AR mode: list of FT tensors (num_ego, K)
        TF mode: list of segments, each segment is list of tensors
        """
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
        # Handle both flat list (AR) and segmented list (TF)
        if len(w) > 0 and isinstance(w[0], list):
            return [[x.detach() for x in seg] for seg in w]
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

    def get_z_local_sparsity_loss(self, target_sparsity=0.1):
        """Backward-compatible sparsity loss (unused in redesign but kept for interface)."""
        z_local = self.get_z_local()
        if z_local is None:
            return torch.tensor(0.0)
        l1_loss = torch.mean(torch.abs(z_local))
        return l1_loss
