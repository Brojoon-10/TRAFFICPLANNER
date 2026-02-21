# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fine-tuning Dataset for TrafficPlannerModel ego reactive training.

Reads JSON files from adv_sol_success/ and creates a FLAT dataset:
  - Each scene produces 2 samples (normal + adv_sol)
  - Normal: ego=fut_init, sur=fut_init (standard driving)
  - Adv_sol: ego=fut_sol, sur=fut_adv (ego avoidance + sur attack)

Returns (scene_graph, map_idx) per sample - same interface as FITDataset.
450 scenes → 900 samples, shuffled together in training.
"""

import os, itertools
import json
import glob
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as Graph

import sys
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
from datasets.utils import MeanStdNormalizer, NUSC_NORM_STATS


class FinetuneTrafficPlannerDataset(Dataset):
    def __init__(self, scenario_path,
                 map_env,
                 split='train',
                 categories=['car', 'truck'],
                 npast=4,
                 nfuture=12,
                 dt=0.5,
                 data_types='both'):
        """
        Args:
            scenario_path: path to adv_sol_success/ directory containing scene_*.json
            map_env: FITMapEnv instance
            split: 'train' or 'val' (80/20 split)
            data_types: 'both' (normal+adv_sol), 'normal', or 'adv_sol'
        """
        assert data_types in ['both', 'normal', 'adv_sol']
        super(FinetuneTrafficPlannerDataset, self).__init__()
        assert split in ['train', 'val']
        self.split = split
        self.data_types = data_types
        self.map_env = map_env
        self.npast = npast
        self.nfuture = nfuture
        self.seq_len = npast + nfuture
        self.dt = dt
        self.categories = categories

        # Category encoding
        iden = torch.eye(len(self.categories), dtype=torch.int)
        self.cat2vec = {self.categories[i]: iden[i] for i in range(len(self.categories))}

        # Load and compile scenarios into flat list
        self.scene_data, self.seq_map = self._compile_scenarios(scenario_path)

        type_str = {'both': 'normal+adv_sol', 'normal': 'normal only', 'adv_sol': 'adv_sol only'}[data_types]
        samples_per_scene = 2 if data_types == 'both' else 1
        num_scenes = len(self.seq_map) // samples_per_scene
        print('FinetuneTP [%s] - Num scenes: %d, Num samples: %d (%s)' %
              (split, num_scenes, len(self.seq_map), type_str))

        # Build normalizers (same as FITDataset)
        ninfo = NUSC_NORM_STATS[tuple(sorted(self.categories))]
        norm_mean = [ninfo['lscale'][0], ninfo['lscale'][0], ninfo['h'][0], ninfo['h'][0], ninfo['s'][0], ninfo['hdot'][0]]
        norm_std = [ninfo['lscale'][1], ninfo['lscale'][1], ninfo['h'][1], ninfo['h'][1], ninfo['s'][1], ninfo['hdot'][1]]
        self.normalizer = MeanStdNormalizer(torch.Tensor(norm_mean), torch.Tensor(norm_std))

        att_norm_mean = [ninfo['l'][0], ninfo['w'][0]]
        att_norm_std = [ninfo['l'][1], ninfo['w'][1]]
        self.veh_att_normalizer = MeanStdNormalizer(torch.Tensor(att_norm_mean), torch.Tensor(att_norm_std))

    def get_state_normalizer(self):
        return self.normalizer

    def get_att_normalizer(self):
        return self.veh_att_normalizer

    def _compile_scenarios(self, scenario_path):
        """Load JSON scenes and build flat dataset. Each scene → 2 entries."""
        scene_files = sorted(glob.glob(os.path.join(scenario_path, 'scene_*.json')))
        n = len(scene_files)
        if n == 0:
            raise ValueError(f'No scene files found in {scenario_path}')

        # 80/20 train/val split
        n_train = int(n * 0.8)
        if self.split == 'train':
            scene_files = scene_files[:n_train]
        else:
            scene_files = scene_files[n_train:]

        scene_data = {}  # sname -> {'normal': {...}, 'adv_sol': {...}}
        seq_map = []     # flat list of (sname, 'normal') or (sname, 'adv_sol')

        for fpath in scene_files:
            with open(fpath, 'r') as f:
                data = json.load(f)

            sname = os.path.basename(fpath)[:-5]  # Remove .json
            N = data['N']

            agents_normal = {}
            agents_adv_sol = {}

            for agent_idx in range(N):
                agent_name = 'ego' if agent_idx == 0 else f'sur_{agent_idx}'

                past_traj = np.array(data['past'][agent_idx])       # [T_past, 6]
                fut_init = np.array(data['fut_init'][agent_idx])    # [T_fut, 4]
                fut_adv = np.array(data['fut_adv'][agent_idx])      # [T_fut, 4]
                fut_sol = np.array(data['fut_sol'][agent_idx])      # [T_fut, 4]
                lw = np.array(data['lw'][agent_idx])                # [2]

                T_fut = fut_init.shape[0]

                def expand_to_6d(fut_4d):
                    fut_6d = np.zeros((T_fut, 6))
                    fut_6d[:, :4] = fut_4d
                    return fut_6d

                # Normal: all agents use fut_init
                full_traj_normal = np.concatenate([past_traj, expand_to_6d(fut_init)], axis=0)

                # Adv+Sol: ego=fut_sol (avoidance), sur=fut_adv (attack)
                if agent_idx == 0:
                    full_traj_adv_sol = np.concatenate([past_traj, expand_to_6d(fut_sol)], axis=0)
                else:
                    full_traj_adv_sol = np.concatenate([past_traj, expand_to_6d(fut_adv)], axis=0)

                T_total = full_traj_normal.shape[0]
                is_vis = np.ones(T_total, dtype=np.float32)

                agents_normal[agent_name] = {
                    'traj': full_traj_normal, 'lw': lw, 'is_vis': is_vis,
                }
                agents_adv_sol[agent_name] = {
                    'traj': full_traj_adv_sol, 'lw': lw, 'is_vis': is_vis,
                }

            scene_data[sname] = {
                'normal': agents_normal,
                'adv_sol': agents_adv_sol,
            }

            # Add entries based on data_types selection
            if self.data_types in ['both', 'normal']:
                seq_map.append((sname, 'normal'))
            if self.data_types in ['both', 'adv_sol']:
                seq_map.append((sname, 'adv_sol'))

        return scene_data, seq_map

    def __len__(self):
        return len(self.seq_map)

    def __getitem__(self, idx):
        """
        Returns (scene_graph, map_idx) - same interface as FITDataset.
        """
        sname, data_type = self.seq_map[idx]
        sidx = 0
        midx = self.npast
        eidx = self.seq_len

        agents = self.scene_data[sname][data_type]
        scene_graph = self._build_scene_graph(agents, sidx, midx, eidx)
        map_idx = 0

        return scene_graph, map_idx

    def _build_scene_graph(self, agents, sidx, midx, eidx):
        """Build PyG scene graph."""
        # Ego always at node 0
        ego_data = agents['ego']
        past = [ego_data['traj'][sidx:midx, :]]
        future = [ego_data['traj'][midx:eidx, :]]
        sem = [self.cat2vec['car']]
        lw = [ego_data['lw']]
        past_vis = [ego_data['is_vis'][sidx:midx]]
        fut_vis = [ego_data['is_vis'][midx:eidx]]

        for agent_name in agents:
            if agent_name == 'ego':
                continue
            agent_data = agents[agent_name]
            if np.isnan(agent_data['traj'][midx - 1]).any():
                continue

            past.append(agent_data['traj'][sidx:midx, :])
            future.append(agent_data['traj'][midx:eidx, :])
            sem.append(self.cat2vec['car'])
            lw.append(agent_data['lw'])
            past_vis.append(agent_data['is_vis'][sidx:midx])
            fut_vis.append(agent_data['is_vis'][midx:eidx])

        past = torch.Tensor(np.stack(past, axis=0))
        future = torch.Tensor(np.stack(future, axis=0))
        sem = torch.Tensor(np.stack(sem, axis=0))
        lw = torch.Tensor(np.stack(lw, axis=0))
        past_vis = torch.Tensor(np.stack(past_vis, axis=0))
        fut_vis = torch.Tensor(np.stack(fut_vis, axis=0))

        # Normalize
        past_gt = self.normalizer.normalize(past)
        past = self.normalizer.normalize(past)
        future_gt = self.normalizer.normalize(future)
        future = self.normalizer.normalize(future)
        lw = self.veh_att_normalizer.normalize(lw)

        # Build fully-connected edge index (no self-loops)
        NA = past.size(0)
        if NA > 1:
            edge_index_list = [(i, j) for i in range(NA) for j in range(NA) if i != j]
            edge_index = torch.Tensor(edge_index_list).T.to(torch.long).contiguous()
        else:
            edge_index = torch.Tensor([[], []]).long()

        graph = Graph(
            x=torch.empty((NA,)),
            pos=torch.empty((NA,)),
            edge_index=edge_index,
            past=past,
            past_gt=past_gt,
            future=future,
            future_gt=future_gt,
            sem=sem,
            lw=lw,
            past_vis=past_vis,
            future_vis=fut_vis,
        )

        return graph
