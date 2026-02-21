# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Modified from interaction_net.py for TrafficPlannerModel
# while keeping message passing shared for interaction preservation.

import numpy as np

import torch
from torch import nn

from torch_geometric.nn import MessagePassing

from models.common import MLP

from utils.transforms import transform2frame


class IndividualSceneInteractionNet(nn.Module):
    """
    Modified SceneInteractionNet with separate output MLPs for ego and other agents.

    Used for trajectory decoding with past_feat (GRU compressed).

    Architecture:
        1. Input MLP (shared, like original STRIVE):
           - mlp_in: all agents -> msg_node_channels

        2. Message Passing (shared):
           - All agents exchange messages through the same conv layers
           - Interaction information is preserved

        3. Output MLP (individualized):
           - mlp_out_ego: msg_node_channels -> out_channels (ego-specific)
           - mlp_out_other: msg_node_channels -> out_channels (other-specific)

    By sharing input MLP and message passing, agents learn consistent interaction
    representations. Output MLP separation allows ego and other agents to have
    different learned output features for their respective decoders.
    """
    def __init__(self, in_node_channels,
                       in_sem_channels,
                       in_edge_channels,
                       msg_node_channels,
                       out_channels,
                       gru_update=False,
                       gru_single_step=False,
                       k=1,
                       nonlinearity=nn.ReLU):
        super(IndividualSceneInteractionNet, self).__init__()

        self.msg_node_channels = msg_node_channels
        self.out_channels = out_channels
        self.gru_update = gru_update
        self.gru_single_step = gru_single_step

        # Shared input MLP for all agents (like original STRIVE)
        self.mlp_in = MLP([in_node_channels, 128, 128, msg_node_channels],
                          nonlinearity=nonlinearity)

        # Shared message passing layers (interaction preserved)
        interaction_layers = []
        for ki in range(k):
            graph_conv = AgentInteractionConv(msg_node_channels,
                                              in_sem_channels,
                                              in_edge_channels,
                                              msg_node_channels,
                                              hidden_size=128,
                                              gru_update=self.gru_update,
                                              gru_single_step=self.gru_single_step,
                                              nonlinearity=nonlinearity,
                                              aggr='max')
            interaction_layers.append(graph_conv)
        self.msg = nn.ModuleList(interaction_layers)

        # Separate output MLPs for ego and other agents
        self.mlp_out_ego = MLP([msg_node_channels, 128, 128, out_channels],
                               nonlinearity=nonlinearity)
        self.mlp_out_other = MLP([msg_node_channels, 128, 128, out_channels],
                                 nonlinearity=nonlinearity)

    def forward(self, scene_graph, ego_mask, h=None, return_out=True):
        """
        :param scene_graph: graph data with x, edge_index, pos, sem
                            x: (NA, D) or (NA, NS, D) for multiple samples
                            pos: (NA, 4) or (NA, NS, 4) for multiple samples
        :param ego_mask: (NA,) boolean tensor, True for ego agents
        :param h: (NA x k x msg_node_channels) hidden state for GRU update
        :param return_out: if True, apply output MLP

        :return ego_feat: (num_ego, out_channels) or (num_ego, NS, out_channels) for mult_samp
        :return other_feat: (num_other, out_channels) or (num_other, NS, out_channels) for mult_samp
        :return h_out: (optional) hidden state if using GRU update
        """
        device = scene_graph.x.device

        # Check if we have multiple samples (3D input)
        mult_samp = len(scene_graph.x.size()) == 3
        if mult_samp:
            NA, NS, D = scene_graph.x.size()
        else:
            NA = scene_graph.x.size(0)
            NS = None

        # 1. Shared input MLP for all agents (like original STRIVE)
        # mlp_in handles both 2D and 3D inputs (broadcasts over last dim)
        x = self.mlp_in(scene_graph.x)

        # 2. Shared message passing (interaction preserved)
        # AgentInteractionConv handles 3D input internally (reshapes to NA, NS*D)
        h_out = []
        for k_idx, layer in enumerate(self.msg):
            cur_h = None if h is None else h[:, k_idx]
            x = layer(x, scene_graph.edge_index, scene_graph.pos, scene_graph.sem, h=cur_h)
            if self.gru_update and h is not None:
                h_out.append(x)

        # 3. Separate output MLP for ego and other
        # After message passing, x is (NA, D) or (NA, NS, D)
        if return_out:
            if ego_mask.any():
                ego_feat = self.mlp_out_ego(x[ego_mask])
            else:
                if mult_samp:
                    ego_feat = torch.zeros(0, NS, self.out_channels, device=device)
                else:
                    ego_feat = torch.zeros(0, self.out_channels, device=device)

            if (~ego_mask).any():
                other_feat = self.mlp_out_other(x[~ego_mask])
            else:
                if mult_samp:
                    other_feat = torch.zeros(0, NS, self.out_channels, device=device)
                else:
                    other_feat = torch.zeros(0, self.out_channels, device=device)
        else:
            if ego_mask.any():
                ego_feat = x[ego_mask]
            else:
                if mult_samp:
                    ego_feat = torch.zeros(0, NS, self.msg_node_channels, device=device)
                else:
                    ego_feat = torch.zeros(0, self.msg_node_channels, device=device)

            if (~ego_mask).any():
                other_feat = x[~ego_mask]
            else:
                if mult_samp:
                    other_feat = torch.zeros(0, NS, self.msg_node_channels, device=device)
                else:
                    other_feat = torch.zeros(0, self.msg_node_channels, device=device)

        if self.gru_update and h is not None:
            h_out_tensor = torch.stack(h_out, dim=1)
            return ego_feat, other_feat, h_out_tensor
        else:
            return ego_feat, other_feat


class AgentInteractionConv(MessagePassing):
    """
    Graph convolution for agent interaction.
    Copied from interaction_net.py without modification.
    """
    def __init__(self, in_node_channels,
                       in_sem_channels,
                       in_edge_channels,
                       out_channels,
                       hidden_size=128,
                       gru_update=False,
                       gru_single_step=False,
                       nonlinearity=nn.ReLU,
                       aggr='max'):
        super(AgentInteractionConv, self).__init__(aggr=aggr,
                                                   flow='source_to_target')
        self.gru_update = gru_update
        self.gru_single_step = gru_single_step
        # source to target constructs messages to node i for each edge in (j,i)
        edge_mlp_input_size = 2*(in_node_channels + in_sem_channels) + in_edge_channels
        if self.gru_update and not self.gru_single_step:
            edge_mlp_input_size += 2*in_node_channels
        self.edge_mlp = MLP([edge_mlp_input_size,
                            hidden_size,
                            hidden_size,
                            out_channels],
                            nonlinearity=nonlinearity)
        # node update function
        if self.gru_update:
            self.update_mlp = MLP([in_node_channels + out_channels + in_sem_channels,
                                    hidden_size,
                                    hidden_size,
                                    out_channels],
                                    nonlinearity=nonlinearity)
            self.update_func = nn.GRUCell(out_channels,
                                          in_node_channels)
        else:
            self.update_mlp = MLP([in_node_channels + out_channels + in_sem_channels,
                                    hidden_size,
                                    out_channels],
                                    nonlinearity=nonlinearity)
        self.out_channels = out_channels

    def forward(self, x, edge_index, pos, sem, h=None):
        """
        :param x: (N x in_node_channels) or (N x NS x in_node_channels) INPUTS to each node
        :param edge_index: (2 x num_edges)
        :param pos: (N x in_edge_channels) or (N x NS x in_edge_channels) (x,y,hx,hy)
        :param sem: (N x in_sem_channels) one-hot vector representing semantic class
        :param h: [OPTIONAL for GRU update] (N x in_node_channels) hidden state
        """
        if len(x.size()) == 3:
            NA, self.NS, D = x.size()
            x = x.reshape(NA, self.NS*D)
            pos = pos.reshape(NA, -1)
        else:
            self.NS = None
        return self.propagate(edge_index, x=x, pos=pos, sem=sem, h=h)

    def message(self, x_i, x_j, pos_i, pos_j, sem_i, sem_j, h_i, h_j):
        """
        :param x_i: (num_edges, in_node_channels)
        :param x_j: (num_edges, in_node_channels) for neighbors
        :param pos_i: (num_edges, in_edge_channels)
        :param pos_j: (num_edges, in_edge_channels) for neighbors
        :param sem_i: (num_edges, in_sem_channels)
        :param sem_j: (num_edges, in_sem_channels) for neighbors
        :param h_i: [OPTIONAL] (num_edges, in_node_channels)
        :param h_j: [OPTIONAL] (num_edges, in_node_channels) for neighbors
        """
        if x_i.size(0) == 0:
            if self.NS is not None:
                return torch.zeros((0, self.out_channels*self.NS)).to(x_i.device)
            else:
                return torch.zeros((0, self.out_channels)).to(x_i.device)
        if self.NS is not None:
            NE, NS = x_i.size(0), self.NS
            pos_i = pos_i.reshape(NE*NS, -1)
            pos_j = pos_j.reshape(NE*NS, -1)
        # need agent j in the frame of agent i
        rel_trans = transform2frame(pos_i, pos_j.unsqueeze(1))[:,0,:]
        rel_trans = torch.where(torch.isnan(rel_trans), torch.zeros_like(rel_trans), rel_trans)

        if self.NS is not None:
            x_i = x_i.reshape(NE, self.NS, -1)
            x_j = x_j.reshape(NE, self.NS, -1)
            sem_i = sem_i.unsqueeze(1).expand(NE, NS, sem_i.size(1))
            sem_j = sem_j.unsqueeze(1).expand(NE, NS, sem_j.size(1))
            rel_trans = rel_trans.reshape(NE, NS, -1)
            if self.gru_update and h_i is not None and h_j is not None:
                h_i = h_i.unsqueeze(1).expand(NE, NS, h_i.size(1))
                h_j = h_j.unsqueeze(1).expand(NE, NS, h_j.size(1))

        msg_in = torch.cat([x_i, x_j, sem_i, sem_j, rel_trans], dim=-1)
        if self.gru_update and h_i is not None and h_j is not None:
            msg_in = torch.cat([msg_in, h_i, h_j], dim=-1)

        if self.NS is not None:
            edge_out = self.edge_mlp(msg_in)
            return edge_out.reshape(NE, NS*edge_out.size(-1))
        else:
            return self.edge_mlp(msg_in)

    def update(self, aggr_out, x, sem, h):
        """
        :param aggr_out: (N x out_channels) output of the aggregation step
        :param x: (N x in_node_channels) or (N x num_samples x in_node_channels)
        :param sem: (N x in_sem_channels)
        :param h: [OPTIONAL] (N x in_node_channels) current hidden state
        """
        if self.NS is not None:
            NA, NS = x.size(0), self.NS
            x = x.reshape(NA, NS, -1)
            aggr_out = aggr_out.reshape(NA, NS, -1)
            sem = sem.unsqueeze(1).expand(-1, NS, -1)

        update_in = torch.cat([x, aggr_out, sem], dim=-1)
        if self.gru_update and h is not None:
            if self.NS is not None:
                update_in = update_in.reshape(NA*NS, -1)
                h = h.reshape(NA*NS, -1)
                update_res = self.update_func(update_in, h)
                return update_res.reshape(NA, NS, -1)
            else:
                return self.update_func(update_in, h)
        elif self.gru_update and self.gru_single_step:
            if self.NS is not None:
                update_in = update_in.reshape(NA*NS, -1)
                update_prepr = self.update_mlp(update_in)
                update_res = self.update_func(update_prepr, x.reshape(NA*NS, -1))
                return update_res.reshape(NA, NS, -1)
            else:
                prepr_out = self.update_mlp(update_in)
                return self.update_func(prepr_out, x)
        else:
            return self.update_mlp(update_in)
