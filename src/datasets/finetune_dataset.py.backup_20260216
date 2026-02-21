# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Fine-tuning Dataset for Agent Cross-Attention Training

Based on FITDataset, but reads JSON files from adv_gen output instead of Excel.
Returns BOTH Normal and Adv scene graphs for contrastive learning.

JSON Structure:
- past: [N, T_past, 6] - Common past trajectory (x, y, hcos, hsin, s, hdot)
- fut_init: [N, T_fut, 4] - Normal GT future (Ego GT, Sur GT) - pre-optimization
- fut_adv: [N, T_fut, 4] - Adversarial future (Ego unchanged, Sur optimized)
- lw: [N, 2] - Vehicle length/width
- sem: [N, 2] - Semantic one-hot (car, truck)
- z_prior: {mean: [N, latent_dim], var: [N, latent_dim]} - Prior z
"""

import os, itertools

import time
from tqdm import tqdm
import numpy as np
from itertools import chain
import json
import glob
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as Graph
from torch_geometric.data import DataLoader as GraphDataLoader

from pyquaternion import Quaternion

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
from datasets.map_env import NUSC_MAP_SIZES
import datasets.nuscenes_utils as nutils
from datasets.utils import MeanStdNormalizer, normalize_scene_graph, read_adv_scenes, NUSC_NORM_STATS, NUSC_VAL_SPLIT_200, NUSC_VAL_SPLIT_400


class FinetuneDataset(Dataset):
    #----------------------Original----------------------
    def __init__(self, data_path,
                 map_env,
                 split='train',
                 categories=['car', 'truck'],

                 npast=4,
                 nfuture=12,
                 dt = 0.5,

                 nusc=None,
                 noise_std=0.0,
                 flip_singapore=True,
                 seq_interval=1,
                 randomize_val=False,
                 val_size=200,
                 require_full_past=False,
                 use_challenge_splits=False,
                 reduce_cats=False,
                 scenario_path=None  # JSON scenario directory path (added for finetune)
                ):
    #----------------------Original----------------------
        '''
         - data_path : root directory of nuscenes version to be used
         - map_env : map environment to base map indices on
         - split : train, val, or test. By defaults, splits by maps.
         - categories : which types of agents to return from
                        ['car', 'truck', 'bus', 'motorcycle', 'trailer', 'cyclist', 'pedestrian']
         - npast : the number of input (past) steps
         - nfuture : the number of output (future) steps
         - nusc : pre-loaded NuScenes object to use rather than loading in again.
         - noise_std: standard dev of gaussian noise to add to state vector.
         - flip_singapore: if true, flips singapore trajectories about the
         - seq_interval: number of steps between sequences in the dataset. default is 1, i.e. each
                        subsequence sequence return from the dataset will be shifted one timestep later
                        than the previous within the same scene.
         - randomize_val: if true, uses a random subset of train split for val, rather than the nusc predict val scenes.
         - val_size: size of the validation split (default: 200 which is the nuscenes prediction challenge). Does not apply to mini version.
         - scenario_path: path to JSON scenario directory for fine-tuning
         - require_full_past: if true, past history must be fully available to return an agent seq.
         - use_challenge_splits: only loads nuScenes prediction challenge data
         - reduce_cats: maps the agent category to be one of 'car', 'truck', or 'pedestrian'
        '''
        super(FinetuneDataset, self).__init__()
        assert split in ['train', 'val', 'test', 'adv']
        self.data_path = data_path
        self.split = split
        self.map_env = map_env
        self.map_list = self.map_env.map_list

        self.dt = dt
        # self.dt = 0.5 # 2 Hz
        # self.dt = 0.1 # 10 Hz

        self.noise_std = noise_std
        self.npast = npast
        self.nfuture = nfuture
        self.seq_len = npast + nfuture
        self.seq_interval = seq_interval
        self.randomize_val = randomize_val
        self.val_size = val_size

        # JSON scenario path for fine-tuning
        if scenario_path is None:
            self.scenario_path = "/home/hj/RACE_STRIVE/data/fine_tuning/agent_attention_training_v1"
        else:
            self.scenario_path = scenario_path

        self.require_full_past = require_full_past
        if self.require_full_past:
            print('require_full_past activated...no agents will have nan past data')
        self.use_challenge_splits = use_challenge_splits
        if self.use_challenge_splits:
            print('Using official nuscenes pred challenge splits...')

        # high-level categories for the model that uses this data
        all_cats = ['car', 'truck', 'bus', 'motorcycle', 'trailer', 'cyclist', 'pedestrian', 'emergency', 'construction']
        all_cat2key = {
            'car' : ['vehicle.car'],
            'truck' : ['vehicle.truck'],
            'bus' : ['vehicle.bus'],
            'motorcycle' : ['vehicle.motorcycle'],
            'trailer' : ['vehicle.trailer'],
            'cyclist' : ['vehicle.bicycle'],
            'pedestrian' : ['human.pedestrian'],
            'emergency' : ['vehicle.emergency'],
            'construction' : ['vehicle.construction']
        }
        self.categories = categories
        self.key2cat = {}
        for cat in self.categories:
            if cat not in all_cats:
                print('Unrecognized category %s!' % (cat))
                exit()
            for k in all_cat2key[cat]:
                self.key2cat[k] = cat

        if reduce_cats:
            reduce_map = {
                'vehicle.car' : 'car',
                'vehicle.truck' : 'truck',
                'vehicle.bus' : 'truck',
                'vehicle.motorcycle' : 'motorcycle',
                'vehicle.trailer' : 'truck',
                'vehicle.bicycle' : 'cyclist',
                'human.pedestrian' : 'pedestrian',
                'vehicle.emergency' : 'car',
                'vehicle.construction' : 'truck'
            }
            self.key2cat = {k : reduce_map[k] for k in self.key2cat.keys()}
            self.categories = sorted(list(set([v for k, v in self.key2cat.items()])))

        iden = torch.eye(len(self.categories), dtype=torch.int)
        self.cat2vec = {self.categories[cat_idx] : iden[cat_idx] for cat_idx in range(len(self.categories))}
        self.vec2cat = {tuple(iden[cat_idx].tolist()) : self.categories[cat_idx]  for cat_idx in range(len(self.categories))}

        # tally number of frames in each class
        self.data_normal = {}
        self.data_adv = {}
        self.ext_future_data = {}  # Sur's fut_adv in 4D for model injection
        self.z_prior_data = {}
        self.seq_map = []
        self.scene2map = {}
        self.data_normal, self.data_adv, self.ext_future_data, self.z_prior_data, self.seq_map = self.compile_scenarios(self.scenario_path)

        self.data_len = len(self.seq_map)

        print('Num scenes: %d' % (len(self.data_normal)))
        print('Num subseq: %d' % (self.data_len))

        # build normalization info objects
        # state normalizer. states of (x, y, hx, hy, s, hdot)
        ninfo = NUSC_NORM_STATS[tuple(sorted(self.categories))]
        norm_mean = [ninfo['lscale'][0], ninfo['lscale'][0], ninfo['h'][0], ninfo['h'][0], ninfo['s'][0], ninfo['hdot'][0]]
        norm_std = [ninfo['lscale'][1], ninfo['lscale'][1], ninfo['h'][1], ninfo['h'][1], ninfo['s'][1], ninfo['hdot'][1]]
        self.normalizer = MeanStdNormalizer(torch.Tensor(norm_mean),
                                           torch.Tensor(norm_std))
        # vehicle attribute normalizer of (l, w)
        att_norm_mean = [ninfo['l'][0], ninfo['w'][0]]
        att_norm_std = [ninfo['l'][1], ninfo['w'][1]]
        self.veh_att_normalizer = MeanStdNormalizer(torch.Tensor(att_norm_mean),
                                                  torch.Tensor(att_norm_std))
        self.norm_info = ninfo

    def get_state_normalizer(self):
        return self.normalizer

    def get_att_normalizer(self):
        return self.veh_att_normalizer

    def compile_scenarios(self, scenario_path):
        """Load and compile JSON scenarios."""
        json_scenes = self.read_json_scenes(scenario_path)

        data_normal = {}
        data_adv = {}
        ext_future_data = {}  # Sur's fut_adv in 4D for model ext_future injection
        z_prior_data = {}
        seq_map = []

        for scene in json_scenes:
            sname = scene['name']

            # Normal data (using fut_init - pre-optimization)
            data_normal[sname] = {}
            # Adv data (using fut_adv - Ego unchanged, Sur optimized)
            data_adv[sname] = {}
            # ext_future: all agents' fut_adv in 4D (for model injection)
            ext_future_data[sname] = []

            N = scene['N']
            for agent_idx in range(N):
                agent_name = 'ego' if agent_idx == 0 else f'sur_{agent_idx}'

                # Past trajectory: [T_past, 6] - (x, y, hcos, hsin, s, hdot)
                past_traj = np.array(scene['past'][agent_idx])  # [T_past, 6]

                # Normal future: fut_init [T_fut, 4] -> need to expand to 6D
                fut_init = np.array(scene['fut_init'][agent_idx])  # [T_fut, 4]

                # Adv future: fut_adv [T_fut, 4] - Ego unchanged, Sur optimized
                fut_adv = np.array(scene['fut_adv'][agent_idx])  # [T_fut, 4]

                # Vehicle attributes
                lw = np.array(scene['lw'][agent_idx])  # [2]

                # Expand future to 6D (add dummy speed=0, hdot=0) for scene_graph compatibility
                T_fut = fut_init.shape[0]
                fut_init_6d = np.zeros((T_fut, 6))
                fut_init_6d[:, :4] = fut_init

                fut_adv_6d = np.zeros((T_fut, 6))
                fut_adv_6d[:, :4] = fut_adv

                # Combine past + future for full trajectory
                full_traj_normal = np.concatenate([past_traj, fut_init_6d], axis=0)  # [T_past+T_fut, 6]

                # For Adv pair:
                #   - Ego uses fut_init (GT) - we want Ego to learn to avoid while following original path
                #   - Sur uses fut_adv (optimized attack trajectory)
                if agent_idx == 0:  # Ego
                    full_traj_adv = np.concatenate([past_traj, fut_init_6d], axis=0)  # Ego keeps GT
                else:  # Sur
                    full_traj_adv = np.concatenate([past_traj, fut_adv_6d], axis=0)  # Sur uses attack traj

                # Visibility (all visible)
                T_total = full_traj_normal.shape[0]
                is_vis = np.ones(T_total, dtype=np.float32)

                # Store in same format as FITDataset
                data_normal[sname][agent_name] = {
                    'traj': full_traj_normal,
                    'lw': lw,
                    'is_vis': is_vis,
                    'k': 'ego' if agent_idx == 0 else 'sur'
                }

                data_adv[sname][agent_name] = {
                    'traj': full_traj_adv,
                    'lw': lw,
                    'is_vis': is_vis,
                    'k': 'ego' if agent_idx == 0 else 'sur'
                }

                # Store ext_future (4D) for all agents - used for model injection
                ext_future_data[sname].append(fut_adv)  # [T_fut, 4]

            # Stack ext_future: [N, T_fut, 4]
            ext_future_data[sname] = np.stack(ext_future_data[sname], axis=0)

            # Store z_prior
            z_prior_data[sname] = {
                'mean': np.array(scene['z_prior']['mean']),
                'var': np.array(scene['z_prior']['var'])
            }

            # Add to seq_map (one entry per scene, starting at idx 0)
            seq_map.append((sname, 0))

        return data_normal, data_adv, ext_future_data, z_prior_data, seq_map

    def read_json_scenes(self, scenario_path):
        """Read JSON scene files and return list of scene dicts."""
        scene_full_list = sorted(glob.glob(os.path.join(scenario_path, 'scene_*.json')))
        n = len(scene_full_list)

        if n == 0:
            raise ValueError(f"No scene files found in {scenario_path}")

        # Split: 80% train, 20% val (no test)
        n_train = int(n * 0.8)

        if self.split == "train":
            scene_separated_list = scene_full_list[:n_train]
        elif self.split == "val":
            scene_separated_list = scene_full_list[n_train:]
        elif self.split == "adv":
            scene_separated_list = scene_full_list
        else:
            raise ValueError("split must be 'train', 'val', or 'adv'")

        scene_list = []
        for scene_fpath in scene_separated_list:
            scene_name = os.path.basename(scene_fpath)[:-5]  # Remove .json
            print('Loading %s...' % (scene_name))

            with open(scene_fpath, 'r') as f:
                data = json.load(f)

            if data is None:
                print('Failed to load! Skipping')
                continue

            cur_scene = {
                'name': scene_name,
                'dt': self.dt,
                'N': data['N'],
                'past': data['past'],
                'fut_init': data['fut_init'],
                'fut_adv': data['fut_adv'],
                'lw': data['lw'],
                'sem': data['sem'],
                'z_prior': data['z_prior']
            }

            scene_list.append(cur_scene)

        return scene_list

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
        Returns:
            normal_graph: Scene graph for normal scenario
            adv_graph: Scene graph for adversarial scenario
            map_idx: Map index
            ext_future: Sur's attack trajectory [N, T_fut, 4] for model injection
            z_prior_mean: Prior z mean [N, latent_dim]
            z_prior_var: Prior z variance [N, latent_dim]
        """
        idx_info = self.seq_map[idx]
        scene_name, sidx = idx_info
        eidx = sidx + self.seq_len
        midx = sidx + self.npast
        _, map_idx = ('boston_seaport', 0)

        # Build normal scene graph
        normal_graph = self._build_scene_graph(self.data_normal, scene_name, sidx, midx, eidx)

        # Build adv scene graph
        adv_graph = self._build_scene_graph(self.data_adv, scene_name, sidx, midx, eidx, is_adv=True)

        # ext_future: Sur's attack trajectory [N, T_fut, 4] for model injection
        # Must normalize to match model's internal space
        ext_future = torch.Tensor(self.ext_future_data[scene_name])  # [N, T_fut, 4]
        ext_future = self.normalizer.normalize(ext_future)

        # z_prior tensors
        z_prior = self.z_prior_data[scene_name]
        z_prior_mean = torch.Tensor(z_prior['mean'])
        z_prior_var = torch.Tensor(z_prior['var'])

        return normal_graph, adv_graph, map_idx, ext_future, z_prior_mean, z_prior_var

    def _build_scene_graph(self, data_dict, scene_name, sidx, midx, eidx, is_adv=False):
        """Build scene graph from data dict (same logic as FITDataset.__getitem__)."""
        # Always put ego at node 0
        ego_data = data_dict[scene_name]['ego']
        past = [ego_data['traj'][sidx:midx, :]]
        future = [ego_data['traj'][midx:eidx, :]]
        sem = [self.cat2vec['car']]  # one-hot vec
        lw = [ego_data['lw']]
        past_vis = [ego_data['is_vis'][sidx:midx]]
        fut_vis = [ego_data['is_vis'][midx:eidx]]

        for agent in data_dict[scene_name]:
            if agent == 'ego':
                continue
            agent_data = data_dict[scene_name][agent]
            if np.isnan(agent_data['traj'][midx-1]).astype(np.int32).sum() > 0:
                continue
            if self.require_full_past and np.isnan(agent_data['traj'][:midx]).sum() > 0:
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

        # Build fully-connected scene graph
        NA = past.size(0)
        edge_index = None
        if NA > 1:
            node_list = range(NA)
            edge_index = list(itertools.product(node_list, node_list))
            edge_index_list = [(i, j) for i, j in edge_index if i != j]
            edge_index = torch.Tensor(edge_index_list).T.to(torch.long).contiguous()
        else:
            edge_index = torch.Tensor([[], []]).long()

        graph_prop_dict = {
            'x': torch.empty((NA,)),
            'pos': torch.empty((NA,)),
            'edge_index': edge_index,
            'past': past,
            'past_gt': past_gt,
            'future': future,
            'future_gt': future_gt,
            'sem': sem,
            'lw': lw,
            'past_vis': past_vis,
            'future_vis': fut_vis,
            'scene_name': scene_name,
            'is_adv': torch.tensor([is_adv])
        }
        scene_graph = Graph(**graph_prop_dict)

        return scene_graph


def finetune_collate_fn(batch):
    """
    Custom collate function for FinetuneDataset.

    Args:
        batch: List of (normal_graph, adv_graph, map_idx, ext_future, z_prior_mean, z_prior_var)

    Returns:
        Dict with batched normal/adv graphs, ext_future, and z_prior data
    """
    from torch_geometric.data import Batch

    normal_graphs = [item[0] for item in batch]
    adv_graphs = [item[1] for item in batch]
    map_idxs = torch.tensor([item[2] for item in batch])
    ext_futures = [item[3] for item in batch]  # List of [N, T_fut, 4]
    z_prior_means = [item[4] for item in batch]
    z_prior_vars = [item[5] for item in batch]

    # Batch graphs
    normal_batch = Batch.from_data_list(normal_graphs)
    adv_batch = Batch.from_data_list(adv_graphs)

    # Concatenate ext_futures along agent dimension (matching batched graph)
    # Each item is [N_i, T_fut, 4], concatenate to [sum(N_i), T_fut, 4]
    ext_future_batch = torch.cat(ext_futures, dim=0)

    return {
        'normal_graph': normal_batch,
        'adv_graph': adv_batch,
        'map_idx': map_idxs,
        'ext_future': ext_future_batch,  # [total_agents, T_fut, 4]
        'z_prior_mean': z_prior_means,
        'z_prior_var': z_prior_vars
    }
