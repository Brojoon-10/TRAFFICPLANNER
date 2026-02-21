# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
import os, argparse, time

import gc
import tqdm
import torch
import torch.optim as optim

import itertools
import math
import numpy as np

import torch
from torch import nn
from torch.distributions import Normal
# 새로 추가된 모듈
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

from models.interaction_net import SceneInteractionNet
from models.common import MLP, car_dynamics

from datasets.utils import normalize_scene_graph
from utils.transforms import transform2frame, kinematics2angle, kinematics2vec
from utils.torch import calc_conv_out
from utils.logger import throw_err, Logger

# TRAJ_ENCODER_CHOICES = ['mlp', 'gru'] # 더 이상 사용되지 않음

class PositionalEncoding(nn.Module):
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
        x: (Batch, Seq, Feature) 또는 (Seq, Batch, Feature) - 여기서는 (Batch, Seq, Feature) [batch_first=True] 기준
        """
        # x의 shape에 맞춰 pe를 가져와 더함
        if x.dim() == 3: # (NA, T, D)
            pe_to_add = self.pe[:x.size(1)].permute(1, 0, 2) # (T, 1, D) -> (1, T, D)
            x = x + pe_to_add
        else: # (T, NA, D)
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FITTrafficModel(nn.Module):
    def __init__(self, npast, nfuture, map_obs_size_pix, nclasses,
                 map_feat_size=64,
                 gcn_hidden_dim=64, # past_feat_size 대신 GCN 및 Transformer의 hidden dim으로 사용
                 latent_size=32,
                 output_bicycle=False,
                 dt = 0.5,
                 # traj_encoder='mlp', # 더 이상 사용되지 않음
                 # GNN-Transformer 인코더 파라미터
                 step_feat_dim=128,
                 gcn_message_dim=128,
                 transformer_nhead=8,
                 transformer_nlayer=3,
                 # Attention-GRU 디코더 파라미터
                 decoder_gru_layers=3,
                 decoder_attn_heads=8,
                 # 기존 Map Conv 파라미터
                 conv_channel_in=4,
                 conv_kernel_list=[7, 5, 5, 3, 3, 3],
                 conv_stride_list=[2, 2, 2, 2, 2, 2],
                 conv_filter_list=[16, 32, 64, 64, 128, 128]
                 ):
        '''
        :param gcn_hidden_dim: GCN의 출력 차원이자 Transformer의 d_model, GRU의 hidden_size
        '''
        super(FITTrafficModel, self).__init__()
        self.normalizer = self.att_normalizer = None
        self.PT = npast
        self.FT = nfuture
        self.dt = dt
        self.NC = nclasses
        self.output_bicycle = output_bicycle
        self.z_size = latent_size

        if self.output_bicycle:
            self.bicycle_params = None
            Logger.log('Using bicycle model as output parameterization of model...')
        
        self.state_size = 6 #(x,y,hx,hy,s,hdot)
        self.att_feat_size = 2 #(l,w)
        self.gcn_hidden_dim = gcn_hidden_dim # Transformer d_model 겸 GRU hidden_size

        #
        # 1. Map encoding (기존과 동일)
        #
        self.mapH = map_obs_size_pix
        self.mapW = map_obs_size_pix
        self.map_obs_size_pix = map_obs_size_pix
        # ... (Map Conv 레이어 정의, 기존 코드와 동일) ...
        conv_layer_list = []
        final_conv_out = map_obs_size_pix
        assert len(conv_kernel_list) == len(conv_stride_list)
        assert len(conv_kernel_list) == len(conv_filter_list)
        conv_filter_list = [conv_channel_in] + conv_filter_list
        for lidx in range(len(conv_kernel_list)):
            cur_conv = nn.Conv2d(conv_filter_list[lidx],
                                 conv_filter_list[lidx+1],
                                 kernel_size=conv_kernel_list[lidx],
                                 stride=conv_stride_list[lidx],
                                 padding=0)
            cur_gn = nn.GroupNorm(1, conv_filter_list[lidx+1])
            conv_layer_list.extend([cur_conv, cur_gn, nn.ReLU()])
            final_conv_out = calc_conv_out(final_conv_out, conv_kernel_list[lidx], conv_stride_list[lidx])
        self.map_conv = nn.Sequential(*conv_layer_list)
        self.map_feat_in_size = conv_filter_list[-1] * final_conv_out * final_conv_out
        self.map_feat_out_size = map_feat_size
        self.map_feature = nn.Linear(self.map_feat_in_size, self.map_feat_out_size)

        #
        # 2. GNN-Transformer Encoder 모듈 (제안하신 구조)
        #
        
        # 2-1. 각 스텝별 State -> Feature Vector로 변환하는 MLP
        # (state + lw + vis + sem)
        step_input_size = self.state_size + self.att_feat_size + 1 + self.NC
        self.step_feature_extractor = MLP([step_input_size, step_feat_dim, step_feat_dim])
        
        # 2-2. 스텝별 상호작용을 추출하는 GCN
        # (SceneInteractionNet을 재사용)
        self.temporal_gcn_encoder = SceneInteractionNet(step_feat_dim, # agent input feat size
                                                         self.NC, # semantic feat size
                                                         4, # edge feat size
                                                         gcn_message_dim, # interaction node size
                                                         self.gcn_hidden_dim, # out feat size
                                                         )
        
        # 2-3. Positional Encoding
        self.positional_encoding = PositionalEncoding(self.gcn_hidden_dim, max_len=max(self.PT, self.FT))
        
        # 2-4. Transformer Encoder (Self-Attention)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.gcn_hidden_dim,
            nhead=transformer_nhead,
            batch_first=True # (NA, T, D) 입력을 받음
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_nlayer)
        
        # 2-5. Latent Variable (z)을 추출하는 MLP
        # Prior: Past context + Map + Sem
        self.latent_prior_net = MLP([self.gcn_hidden_dim + self.map_feat_out_size + self.NC,
                                     128,
                                     self.z_size * 2])
        # Posterior: Past context + Future context + Map + Sem
        self.latent_posterior_net = MLP([self.gcn_hidden_dim * 2 + self.map_feat_out_size + self.NC,
                                         128,
                                         self.z_size * 2])

        #
        # 3. Attention-GRU Decoder 모듈 (제안하신 구조)
        #
        
        # 3-1. Latent 'z'를 Decoder 차원으로 투영
        self.z_projection = nn.Linear(self.z_size, self.gcn_hidden_dim)

        # 3-2. 디코더 GRU (기존 decoder_memory 재사용, hidden_size만 맞춤)
        self.num_memory_layers = decoder_gru_layers
        self.decoder_memory = nn.GRU(4, # input size (x,y,hx,hy)
                                    self.gcn_hidden_dim, # hidden size
                                    self.num_memory_layers,
                                    batch_first=True,
                                    )
        
        # 3-3. Decoder Cross-Attention (Past sequence attention)
        self.decoder_attention = MultiheadAttention(
            embed_dim=self.gcn_hidden_dim,
            num_heads=decoder_attn_heads,
            batch_first=True # (Batch, Seq, Feature)
        )

        ###########################################################################
        # HJ_ADDED: Agent Cross-Attention for Future Interaction
        ###########################################################################
        # This module enables agents to see other agents' current positions during
        # future trajectory decoding (autoregressive step).
        # - Query: Current agent's GRU hidden state
        # - Key/Value: Other agents' current states (position + heading)
        # This allows reactive behavior: agents can avoid collisions by attending
        # to where other agents currently are at each decoding timestep.
        ###########################################################################
        self.agent_state_encoder = MLP([4, 64, self.gcn_hidden_dim])  # (x,y,hx,hy) -> D
        self.agent_cross_attention = MultiheadAttention(
            embed_dim=self.gcn_hidden_dim,
            num_heads=decoder_attn_heads,
            batch_first=True
        )
        ###########################################################################
        
        # 3-4. 최종 궤적 (a, hdot) 예측 MLP
        if self.output_bicycle:
            self.traj_out_size = 2 # (a,hdot)
        else:
            self.traj_out_size = 4 # (x,y,hx,hy)

        ###########################################################################
        # HJ_MODIFIED: Added agent_attn (gcn_hidden_dim) to decoder MLP input
        ###########################################################################
        # Input: GRU output + Past Attention + Agent Attention + Z context + Map + LW + Sem
        decoder_mlp_in_size = (self.gcn_hidden_dim + # GRU output
                               self.gcn_hidden_dim + # Past sequence attention (map/temporal context)
                               self.gcn_hidden_dim + # Agent cross-attention (other agents' positions)  # HJ_ADDED
                               self.gcn_hidden_dim + # Z projection
                               self.map_feat_out_size +
                               self.att_feat_size +
                               self.NC)
        ###########################################################################
        self.decoder_output_mlp = MLP([decoder_mlp_in_size, 128, self.traj_out_size])

    def forward(self, scene_graph, map_idx, map_env,
                use_post_mean=False,
                future_sample=False):
        
        # 1. 맵 특징 추출 (기존과 동일)
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # 2. PRIOR (Past 인코딩)
        # (past_seq_out은 디코더의 Key/Value로 사용됨)
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)

        # 3. POSTERIOR (Past + Future 인코딩)
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_seq_out)

        # 4. DECODER
        if use_post_mean:
            z_samp = post_mu
        else:
            z_samp = self.rsample(post_mu, post_var)
            
        # 디코더의 Key/Value로 past_seq_out (Transformer Encoder 출력)을 전달
        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp, map_idx, map_env) 

        net_out = {
            'prior_out' : (prior_mu, prior_var),
            'posterior_out' : (post_mu, post_var),
            'future_pred' : future_pred
        }

        if future_sample:
            prior_samp = self.rsample(prior_mu, prior_var)
            future_samp = self.decoder(scene_graph, map_feat, past_seq_out, prior_samp, map_idx, map_env)
            net_out['future_samp'] = future_samp
        
        return net_out

    def prior(self, scene_graph, map_feat):
        """
        Past 궤적을 GNN-Transformer로 인코딩하여 Prior (z)와
        Decoder에서 사용할 Key/Value (past_seq_out)를 생성합니다.
        """
        # 1. Past 궤적을 시퀀셜 GNN-Transformer로 인코딩
        # past_seq_out: [NA, PT, D], past_context: [NA, D]
        past_seq_out, past_context = self._run_temporal_gnn_transformer(
            scene_graph,
            scene_graph.past,
            scene_graph.past_vis,
            self.PT
        )
        
        # 2. Prior Latent 계산
        prior_in_feat = torch.cat([past_context, map_feat, scene_graph.sem], dim=-1)
        prior_z = self.latent_prior_net(prior_in_feat)
        
        mean, logvar = prior_z[:, :self.z_size], prior_z[:, self.z_size:]
        var = torch.exp(logvar)
        
        return mean, var, past_seq_out # past_seq_out을 디코더로 전달

    def encoder(self, scene_graph, map_feat, past_seq_out):
        """
        Future 궤적을 GNN-Transformer로 인코딩하고,
        Past 컨텍스트와 결합하여 Posterior (z)를 생성합니다.
        (past_seq_out은 prior에서 계산한 것을 재사용)
        """
        # 1. Past 컨텍스트 (이미 계산됨)
        past_context = past_seq_out[:, -1, :] # 간단히 마지막 스텝 사용 (혹은 mean pool)
        
        # 2. Future 궤적 인코딩
        # future_seq_out: [NA, FT, D], future_context: [NA, D]
        future_seq_out, future_context = self._run_temporal_gnn_transformer(
            scene_graph,
            scene_graph.future,
            scene_graph.future_vis,
            self.FT
        )
        
        # 3. Posterior Latent 계산
        posterior_in_feat = torch.cat([past_context, future_context, map_feat, scene_graph.sem], dim=-1)
        posterior_z = self.latent_posterior_net(posterior_in_feat)
        
        mean, logvar = posterior_z[:, :self.z_size], posterior_z[:, self.z_size:]
        var = torch.exp(logvar)
        
        return mean, var

    def _run_temporal_gnn_transformer(self, scene_graph, traj_data, vis_data, T):
        """
        (제안하신 핵심 로직)
        궤적 데이터(past 또는 future)를 받아,
        매 스텝 GCN을 실행하고, 그 결과를 시퀀스로 묶어
        Transformer Encoder를 통과시키는 헬퍼 함수.
        
        :param traj_data: (NA, T, 6)
        :param vis_data: (NA, T)
        :param T: Time steps (PT or FT)
        :return: (sequence_output, context_vector)
                 - sequence_output: (NA, T, D) - Transformer의 전체 시퀀스 출력
                 - context_vector: (NA, D) - Transformer의 요약된 컨텍스트 (예: 마지막 스텝)
        """
        NA = traj_data.size(0)
        
        # scene_graph의 edge_index 등은 재사용
        g_in_data = scene_graph
        
        all_gcn_features = []
        
        # 1. Temporal GCN (매 스텝 GCN 실행)
        for t in range(T):
            # t시점의 state, vis, lw, sem
            cur_state = traj_data[:, t, :]
            cur_vis = vis_data[:, t].unsqueeze(-1)
            cur_lw = scene_graph.lw
            cur_sem = scene_graph.sem
            
            # (state + lw + vis + sem)
            step_in_feat = torch.cat([cur_state, cur_lw, cur_vis, cur_sem], dim=-1)
            # MLP로 GCN 입력 특징 생성
            gcn_node_in = self.step_feature_extractor(step_in_feat)
            
            # GCN에 입력하기 위해 scene_graph 객체 업데이트
            g_in_data.x = gcn_node_in
            g_in_data.pos = cur_state[:, :4] # t시점의 위치
            
            # t시점의 GCN 실행
            gcn_feat_t = self.temporal_gcn_encoder(g_in_data) # (NA, gcn_hidden_dim)
            all_gcn_features.append(gcn_feat_t)
            
        # (T, NA, D) -> (NA, T, D)
        sequence_features = torch.stack(all_gcn_features, dim=1) 
        
        # 2. Positional Encoding
        sequence_features = self.positional_encoding(sequence_features)
        
        # 3. Transformer Encoder (Self-Attention)
        # batch_first=True로 설정했으므로 (NA, T, D) 그대로 입력
        transformer_out = self.transformer_encoder(sequence_features) # (NA, T, D)
        
        # 4. Context Vector 추출 (예: 마지막 스텝의 출력)
        context_vector = transformer_out[:, -1, :] # (NA, D)
        
        # 전체 시퀀스(Attention K/V용)와 컨텍스트(z 예측용) 반환
        return transformer_out, context_vector


    def decoder(self, scene_graph, map_feat, past_seq_out, z, map_idx, map_env,
                ext_future=None,
                ext_future_mask=None,
                nfuture=None):
        """
        새로운 Attention + GRU 기반 디코더

        :param past_seq_out: (NA, PT, D) - Prior에서 계산된 Transformer Encoder 출력 (Cross-Attention의 K, V)
        :param z: (NA, z_size) 또는 (NA, NS, z_size)
        :param ext_future: (B, FT, 4) for ego-only (legacy) or (NA, FT, 4) for all agents
        :param ext_future_mask: (NA,) bool tensor - which agents to inject. If None, uses ego_inds (legacy behavior)
        """
        return self.autoregressive_decoder(scene_graph, map_feat, past_seq_out, z, map_idx, map_env,
                                            ext_future=ext_future,
                                            ext_future_mask=ext_future_mask,
                                            nfuture=nfuture)

    def autoregressive_decoder(self, scene_graph, map_feat, past_seq_out, z, map_idx, map_env,
                                ext_future=None,
                                ext_future_mask=None,
                                nfuture=None):
        NA = map_feat.size(0)
        FT = self.FT if nfuture is None else nfuture

        prev_state = scene_graph.past[:, -1, :] if self.output_bicycle else scene_graph.past[:, -1, :4]
        traj_out = []
        cur_map_feat = map_feat
        cur_sem = scene_graph.sem
        cur_lw = scene_graph.lw
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:,0].unsqueeze(1)
        ego_inds = scene_graph.ptr[:-1]
        scene_graph.pos = scene_graph.past[:, -1, :4]

        # ext_future_mask: (NA,) bool - True means inject, False means predict
        # If ext_future_mask is None, use legacy behavior (ego_inds only)

        zsize = z.size()
        NS = None
        if len(zsize) == 3: # Multi-sample (NA, NS, D)
            NS = zsize[1]
            bsize = NA * NS # 배치 크기를 NS배
            
            # scene_graph의 위치 및 특징들 확장
            scene_graph.pos = scene_graph.pos.unsqueeze(1).expand(NA, NS, scene_graph.pos.size(1)).reshape(bsize, -1)
            cur_map_feat = cur_map_feat.unsqueeze(1).expand(NA, NS, map_feat.size(1)).reshape(bsize, -1)
            cur_sem = cur_sem.unsqueeze(1).expand(NA, NS, scene_graph.sem.size(1)).reshape(bsize, -1)
            cur_lw = cur_lw.unsqueeze(1).expand(NA, NS, scene_graph.lw.size(1)).reshape(bsize, -1)
            cur_veh_len = cur_veh_len.unsqueeze(1).expand(NA, NS, 1).reshape(bsize, 1)
            prev_state = prev_state.unsqueeze(1).expand(NA, NS, prev_state.size(1)).reshape(bsize, -1)
            
            if ext_future is not None:
                ext_future = ext_future.unsqueeze(1).expand(NA, NS, -1, 4).reshape(bsize, -1, 4)
                if ext_future_mask is not None:
                    # (NA,) -> (NA, NS) -> (NA*NS,)
                    ext_future_mask = ext_future_mask.unsqueeze(1).expand(NA, NS).reshape(bsize)
                ego_inds = ego_inds.unsqueeze(1).expand(NA, NS).reshape(bsize)
                
            # Z context (z_proj) 확장
            z_proj = self.z_projection(z).reshape(bsize, -1) # (NA*NS, D)
            
            # Attention Key/Value (past_seq_out) 확장
            # (NA, PT, D) -> (NA, NS, PT, D) -> (NA*NS, PT, D)
            encoder_k_v = past_seq_out.unsqueeze(1).expand(NA, NS, self.PT, -1).reshape(bsize, self.PT, -1)

            # GRU Hidden State (past_context) 확장
            # (NA, D) -> (NA, NS, D) -> (NA*NS, D)
            past_context = past_seq_out[:, -1, :].unsqueeze(1).expand(NA, NS, -1).reshape(bsize, -1)

        else: # Single-sample
            bsize = NA
            z_proj = self.z_projection(z) # (NA, D)
            encoder_k_v = past_seq_out # (NA, PT, D)
            past_context = past_seq_out[:, -1, :] # (NA, D)

        mult_samp = (NS is not None)
            
        # GRU hidden state 초기화 (Past의 마지막 Context 사용)
        cur_mem_state = past_context.unsqueeze(0).expand(self.num_memory_layers, bsize, self.gcn_hidden_dim).contiguous()

        for t in range(FT):
            # [제거] scene_graph.x = ... (GCN을 더 이상 사용 안 함)
            # [제거] decoder_out = self.decoder_net(scene_graph)

            # 1. GRU (Query 생성)
            # (이전 스텝의 local state를 GRU 입력으로 사용)
            if t == 0:
                # 첫 스텝은 0 벡터 또는 마지막 과거 state의 local transform 사용
                # 여기서는 간단히 0 벡터 사용
                cur_state_local = torch.zeros_like(prev_state[:, :4])
            # (cur_state_local은 이전 루프에서 계산됨)
            
            gru_input = cur_state_local.unsqueeze(1) # (B, 1, 4)
            gru_out, cur_mem_state = self.decoder_memory(gru_input, cur_mem_state)
            gru_out = gru_out.squeeze(1) # (B, D) - 이것이 Attention Query
            
            # 2. Cross-Attention (Past sequence)
            # Q: (B, 1, D), K/V: (B, PT, D) (batch_first=True)
            query = gru_out.unsqueeze(1)
            attn_out, past_attn_weights = self.decoder_attention(query=query, key=encoder_k_v, value=encoder_k_v)
            attn_out = attn_out.squeeze(1) # (B, D) - Context Vector

            ###########################################################################
            # HJ_ADDED: Agent Cross-Attention for Future Interaction
            ###########################################################################
            # Compute attention over other agents' current positions.
            # This enables each agent to "see" where other agents are at the current
            # decoding timestep, allowing for reactive collision avoidance behavior.
            #
            # For each agent i:
            #   Query: agent i's GRU hidden state
            #   Key/Value: all agents' current global positions (including self)
            #
            # The attention mechanism will learn to focus on nearby/relevant agents.
            ###########################################################################

            # Get all agents' current global positions for attention
            # At t=0, use last past position; at t>0, use previous step's output
            if t == 0:
                all_agents_pos = scene_graph.past[:, -1, :4].clone()  # (NA, 4)
                # HJ_FIX: At t=0, also apply ext_future injection for Agent Attention
                if ext_future is not None and ext_future_mask is not None:
                    inject_inds = ext_future_mask.nonzero(as_tuple=True)[0]
                    if inject_inds.numel() > 0:
                        all_agents_pos[inject_inds] = ext_future[inject_inds, 0]
                if mult_samp:
                    all_agents_pos = all_agents_pos.unsqueeze(1).expand(NA, NS, 4).reshape(bsize, 4)
            else:
                # Use the previous timestep's global position (already computed)
                all_agents_pos = prev_state[:, :4]  # (bsize, 4)

            # Encode all agents' positions: (bsize, 4) -> (bsize, D)
            all_agents_feat = self.agent_state_encoder(all_agents_pos)  # (bsize, D)

            # Reshape for cross-attention: need (bsize, NA, D) where NA is number of agents
            # Each agent attends to ALL agents (including itself)
            # For single sample: (NA, D) -> each agent sees all NA agents
            # For multi-sample: (NA*NS, D) -> each (agent, sample) sees all NA agents in that sample

            if mult_samp:
                # (NA*NS, D) -> (NA, NS, D) -> for each sample, gather all agents
                all_agents_feat_reshaped = all_agents_feat.reshape(NA, NS, -1)  # (NA, NS, D)
                # Each agent in each sample should see all NA agents in that same sample
                # K/V: (NS, NA, D) -> (NA*NS, NA, D) by repeating for each agent
                agent_kv = all_agents_feat_reshaped.permute(1, 0, 2)  # (NS, NA, D)
                agent_kv = agent_kv.unsqueeze(1).expand(NS, NA, NA, -1)  # (NS, NA, NA, D)
                agent_kv = agent_kv.reshape(bsize, NA, -1)  # (NA*NS, NA, D)
            else:
                # (NA, D) -> (NA, NA, D): each agent sees all NA agents
                agent_kv = all_agents_feat.unsqueeze(0).expand(NA, NA, -1)  # (NA, NA, D)

            # Agent Cross-Attention
            # Q: (bsize, 1, D), K/V: (bsize, NA, D)
            agent_query = gru_out.unsqueeze(1)  # (bsize, 1, D)
            agent_attn_out, agent_attn_weights = self.agent_cross_attention(
                query=agent_query, key=agent_kv, value=agent_kv
            )
            agent_attn_out = agent_attn_out.squeeze(1)  # (bsize, D)

            # DEBUG: Print attention weight distribution
            if t == 0 and hasattr(self, '_debug_attention') and self._debug_attention:
                print(f"\n=== ATTENTION WEIGHT DEBUG (t={t}) ===")
                # Past Cross-Attention: past_attn_weights (bsize, 1, PT)
                past_w = past_attn_weights.squeeze(1)  # (bsize, PT)
                print(f"Past Attn shape: {past_w.shape}")
                print(f"Past Attn (agent 0): {past_w[0].detach().cpu().numpy()}")

                # Agent Cross-Attention: agent_attn_weights (bsize, 1, NA)
                agent_w = agent_attn_weights.squeeze(1)  # (bsize, NA)
                print(f"Agent Attn shape: {agent_w.shape}")
                for agent_idx in range(min(5, bsize)):
                    self_idx = agent_idx % NA
                    self_attn = agent_w[agent_idx, self_idx].item()  # attention to self
                    other_attn = agent_w[agent_idx].sum().item() - self_attn
                    print(f"  Agent {agent_idx}: self={self_attn:.4f}, others_sum={other_attn:.4f}")
                print(f"Agent Attn weights (first 5 agents):\n{agent_w[:5].detach().cpu().numpy()}")
                print("=" * 50)
            ###########################################################################

            # 3. Output MLP
            ###########################################################################
            # HJ_MODIFIED: Added agent_attn_out to MLP input
            ###########################################################################
            # Input: GRU out + Past Attn + Agent Attn + Z proj + Map + LW + Sem
            mlp_in = torch.cat([gru_out, attn_out, agent_attn_out, z_proj, cur_map_feat, cur_lw, cur_sem], dim=-1)
            ###########################################################################
            decoder_out = self.decoder_output_mlp(mlp_in) # (B, 2)
            
            # --- (이하 Bicycle 모델 적용은 기존과 거의 동일) ---
            
            cur_state_local_kin = None # local (x,y,hx,hy)
            cur_state_global = None # global (x,y,hx,hy)
            cur_bike_state = None # global (x,y,hx,hy,s,hdot)

            if self.output_bicycle:
                dynamics_out = decoder_out.view(bsize, 1, 1, 2) # (B, 1, 1, 2)
                # unnormalize
                a_out = dynamics_out[:,:,:,0]*self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
                ddh_out = dynamics_out[:,:,:,1]*self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
                # simulate forward
                init_state = self.normalizer.unnormalize(prev_state)
                cur_bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, cur_veh_len)[:,0,0]
                cur_bike_state = self.normalizer.normalize(cur_bike_state)

                cur_state_global = cur_bike_state[:, :4]
                # 다음 스텝 GRU 입력을 위한 local frame 계산
                cur_state_local_kin = transform2frame(prev_state[:,:4], cur_state_global.unsqueeze(1))[:,0]
            else:
                # (x,y,hx,hy) 직접 예측
                heading_mag = torch.norm(decoder_out[:, 2:], dim=-1, keepdim=True)
                cur_state_local_kin = torch.cat([decoder_out[:, :2], decoder_out[:, 2:] / heading_mag], dim=-1)
                cur_state_global = transform2frame(prev_state,
                                                   cur_state_local_kin.unsqueeze(1),
                                                   inverse=True)[:, 0, :]
            
            # 다음 스텝 GRU 입력 업데이트
            cur_state_local = cur_state_local_kin

            # save for output
            traj_out.append(cur_state_global)

            if ext_future is not None:
                # (외부 입력이 있으면 GRU 입력 및 다음 state 덮어쓰기)
                cur_state_global_clone = cur_state_global.clone()

                if ext_future_mask is not None:
                    # New behavior: ext_future_mask (NA,) or (NA*NS,) bool tensor
                    # ext_future: (NA, FT, 4) or (NA*NS, FT, 4)
                    # True인 agent만 주입
                    inject_inds = ext_future_mask.nonzero(as_tuple=True)[0]
                    if inject_inds.numel() > 0:
                        cur_state_global_clone[inject_inds] = ext_future[inject_inds, t]
                        cur_state_local[inject_inds] = transform2frame(
                            prev_state[inject_inds][:, :4],
                            cur_state_global_clone[inject_inds].unsqueeze(1))[:, 0, :]
                else:
                    # Legacy behavior: ego_inds only (for backward compatibility)
                    cur_state_global_clone[ego_inds] = ext_future[:, t]
                    cur_state_local[ego_inds] = transform2frame(prev_state[ego_inds][:, :4],
                                                                cur_state_global_clone[ego_inds].unsqueeze(1))[:, 0, :]

                # global (다음 map crop용)
                cur_state_global = cur_state_global_clone
            
            # update prev state
            if self.output_bicycle:
                prev_state = cur_bike_state.clone()
                # HJ_FIX: For injected agents, overwrite prev_state with ext_future values
                # so that Agent Attention can see the injected positions in next timestep
                if ext_future is not None and ext_future_mask is not None:
                    inject_inds = ext_future_mask.nonzero(as_tuple=True)[0]
                    if inject_inds.numel() > 0:
                        # ext_future: (x,y,hx,hy) 4-dim, cur_bike_state: 6-dim (x,y,hx,hy,s,hdot)
                        # Overwrite position/heading only, keep s/hdot from original
                        prev_state[inject_inds, :4] = ext_future[inject_inds, t]
            else:
                prev_state = cur_state_global

            if t < FT - 1:
                # [제거] GRU 메모리 업데이트 (이미 루프 시작 시 수행됨)
                
                # crop and encode map around new position
                scene_graph.pos = cur_state_global.detach() if not mult_samp else cur_state_global.detach().reshape(NA, NS, -1)
                cur_map_feat = self.encode_map(scene_graph, map_idx, map_env)
                
                if mult_samp:
                    # cur_map_feat이 (NA, NS, D)로 나오므로 (NA*NS, D)로 변경
                    cur_map_feat = cur_map_feat.reshape(bsize, -1)

                # update positions (다음 스텝에는 사용되지 않지만 혹시 모를 참조를 위해)
                scene_graph.pos = cur_state_global if not mult_samp else cur_state_global.reshape(NA, NS, -1)

        # return all outputs in global frame
        traj_out = torch.stack(traj_out, dim=1)
        if mult_samp:
            traj_out = traj_out.reshape(NA, NS, FT, -1)
        return traj_out

    
    # [기존] set_normalizer, get_normalizer 등
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

    # [기존] reconstruct, sample, sample_batched, embed, decode_embedding
    # 참고: 이 함수들은 내부적으로 forward, prior, decoder를 호출하므로
    #       forward/prior/decoder가 수정되었기 때문에 자동으로 새 로직을 따릅니다.
    #       (단, embed/decode_embedding은 'past_seq_out'을 전달하도록 수정 필요)
    
    def reconstruct(self, scene_graph, map_idx, map_env):
        # ... (기존 코드와 동일, 내부적으로 forward/decoder 호출) ...
        # (이 함수는 새 로직을 위해 약간 수정이 필요할 수 있습니다)
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) 
        
        # prior에서 past_seq_out을 받아와야 함
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        post_mu, post_var = self.encoder(scene_graph, map_feat, past_seq_out)
        
        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, post_mu, map_idx, map_env)

        net_out = {
            'posterior_out' : (post_mu, post_var),
            'future_pred' : future_pred
        }
        return net_out

    def sample(self, scene_graph, map_idx, map_env, num_samples,
                include_mean=False,
                nfuture=None):
        # ... (기존 로직과 거의 동일, prior/decoder 호출 부분만 확인) ...
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) 

        # PRIOR (past_seq_out을 받아옴)
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        prior_distrib = Normal(prior_mu, torch.sqrt(prior_var))
        # ... (net_out 초기화) ...
        net_out = { 'prior_out': (prior_mu, prior_var), 'z_samp': [], 'z_logprob': [], 'z_mdist': [], 'future_pred': [] }

        for sidx in range(num_samples):
            if include_mean and sidx == (num_samples-1):
                z_samp = prior_mu
            else:
                z_samp = self.rsample(prior_mu, prior_var)

            z_logprob = prior_distrib.log_prob(z_samp).sum(dim=-1)
            z_mdist = torch.norm((z_samp - prior_mu) / torch.sqrt(prior_var), dim=-1)
            
            # Decoder (past_seq_out 전달)
            future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp, map_idx, map_env,
                                        nfuture=nfuture) 
            net_out['z_samp'].append(z_samp)
            net_out['z_logprob'].append(z_logprob)
            net_out['z_mdist'].append(z_mdist)
            net_out['future_pred'].append(future_pred)
        
        # ... (stack) ...
        net_out['z_samp'] = torch.stack(net_out['z_samp'], dim=1)
        net_out['z_logprob'] = torch.stack(net_out['z_logprob'], dim=1)
        net_out['z_mdist'] = torch.stack(net_out['z_mdist'], dim=1)
        net_out['future_pred'] = torch.stack(net_out['future_pred'], dim=1)
        return net_out

    def sample_batched(self, scene_graph, map_idx, map_env, num_samples,
                        include_mean=False,
                        nfuture=None):
        # ... (기존 로직과 거의 동일, prior/decoder 호출 부분만 확인) ...
        NA = scene_graph.past.size(0)
        NS = num_samples

        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) 

        # PRIOR (past_seq_out 받아옴)
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)
        
        # ... (z sampling 로직) ...
        samp_mu = prior_mu.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        samp_var = prior_var.view(1, NA, self.z_size).expand(NS, NA, self.z_size)
        prior_distrib = Normal(samp_mu, torch.sqrt(samp_var))
        z_samp = self.rsample(samp_mu, samp_var)
        if include_mean:
            z_samp[-1, :, :] = prior_mu
        
        # Decoder (past_seq_out 전달)
        # z_samp: (NS, NA, Z) -> (NA, NS, Z)
        # past_seq_out: (NA, PT, D)
        future_pred = self.decoder(scene_graph, map_feat, past_seq_out, z_samp.transpose(0, 1), map_idx, map_env,
                                    nfuture=nfuture) # (NA, NS, FT, 4)
        
        net_out = {
            'prior_out' : (prior_mu, prior_var),
            'z_samp' : z_samp.view(NS, NA, self.z_size).transpose(0, 1),
            'future_pred' : future_pred
        }
        # ... (logprob, mdist 계산 및 반환) ...
        return net_out

    # [기존] encode_map
    def encode_map(self, scene_graph, map_idx, map_env):
        # (이 함수는 기존 코드와 100% 동일합니다)
        NA = scene_graph.pos.size(0)
        NS = None if len(scene_graph.pos.size()) != 3 else scene_graph.pos.size(1)
        # first must unnormalize to get true world space state
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=True)
        # get local crops based on .pos
        map_obs = map_env.get_map_crop(scene_graph, map_idx).to(torch.float) # NA x C x mapH x mapW
        # encode
        map_feat = self.map_conv(map_obs)
        bsize = NA if NS is None else NA*NS
        map_feat = self.map_feature(map_feat.view(bsize, self.map_feat_in_size))

        if NS is not None:
            map_feat = map_feat.reshape(NA, NS, -1)

        # re-normalize scene graph
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=False)
        return map_feat

    # [제거] encode_past (-> _run_temporal_gnn_transformer 에 통합됨)
    # [제거] encode_future (-> _run_temporal_gnn_transformer 에 통합됨)

    # [기존] rsample
    def rsample(self, mean, var):
        eps = torch.randn_like(mean)
        z = mean + eps*torch.sqrt(var)
        return z

    # [기존] sim_traj
    def sim_traj(self, init_state, a, ddh, vehicle_len):
        cur_kinematics = kinematics2angle(init_state)
        sim_steps = a.size(-1)
        kin_seq = []
        for t in range(sim_steps):
            cur_kinematics = car_dynamics(cur_kinematics, a[:,:,t], ddh[:,:,t],
                                        self.dt, 0, 1, 2, 3,
                                        4, vehicle_len, self.bicycle_params['maxhdot'],
                                        self.bicycle_params['maxs'])
            kin_seq.append(kinematics2vec(cur_kinematics))

        traj_out = torch.stack(kin_seq, dim=2)
        return traj_out

#--------------------------------------HJ ADDED---------------------------------------

    def embed(self, scene_graph, map_idx, map_env):
        '''
        Given past and optionally future, embed the given trajectories
        using the prior and (optionally) posterior.

        returns dict with prior_out and (optionally) posterior_out as well as
        other required values to decode (map feat etc..)
        '''
        # extract map feature for each agent based on last frame of past
        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat

        # PRIOR (returns 3 values: mu, var, past_seq_out)
        prior_mu, prior_var, past_seq_out = self.prior(scene_graph, map_feat)

        embed_out = {
            'prior_out' : (prior_mu, prior_var),
            'map_feat' : map_feat,
            'past_seq_out' : past_seq_out
        }

        if 'future' in scene_graph:
            # POSTERIOR
            post_mu, post_var = self.encoder(scene_graph, map_feat, past_seq_out)
            embed_out['posterior_out'] = (post_mu, post_var)

        return embed_out

    def decode_embedding(self, z, embed_out, scene_graph, map_idx, map_env,
                            ext_future=None,
                            ext_future_mask=None,
                            nfuture=None):
        '''
        Given inputs/outputs of embed function, decodes to predicted trajectory in world space.

        :param ext_future: (B, FT, 4) for ego-only (legacy) or (NA, FT, 4) for all agents
        :param ext_future_mask: (NA,) bool tensor - which agents to inject. If None, uses ego_inds (legacy)
        '''
        future_pred = self.decoder(scene_graph, embed_out['map_feat'], embed_out['past_seq_out'],
                                        z, map_idx, map_env, ext_future=ext_future,
                                        ext_future_mask=ext_future_mask,
                                        nfuture=nfuture) # (NA, FT, 4)
        return {'future_pred' : future_pred}

#--------------------------------------HJ ADDED---------------------------------------
