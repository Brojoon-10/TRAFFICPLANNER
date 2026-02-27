# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

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

# This function is based on https://github.com/Khrylx/AgentFormer/blob/main/data/process_nuscenes.py#L20
# Copyright 2021 Carnegie Mellon University
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



class FITDataset(Dataset):
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
                 reduce_cats=False
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
         - scenario_path: (optional) path to additional adversarial scenarios to load in as part of the dataset. These will simply be appended to the end of the split. If this is given, data_path is allowed to be None.
         - require_full_past: if true, past history must be fully available to return an agent seq.
         - use_challenge_splits: only loads nuScenes prediction challenge data
         - reduce_cats: maps the agent category to be one of 'car', 'truck', or 'pedestrian'
        '''
        super(FITDataset, self).__init__()
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

        # self.scenario_path = "/home/hj/RACE_STRIVE/data/race_scenarios/adv_input_600_for_comparison"  # For Adv Gen
        # self.scenario_path = "/home/hj/RACE_STRIVE/data/race_scenarios/collision_test"  # For just temp test
        # self.scenario_path = "/home/hj/RACE_STRIVE/data/race_scenarios/adv_input_05"  # For Train

        # self.scenario_path = os.path.join(cur_file_path, '..', '..', 'data', 'race_scenarios', 'trafficplanner_normal_small_datasets')
        # self.scenario_path = os.path.join(cur_file_path, '..', '..', 'data', 'race_scenarios', 'trafficplanner_tf')
        # self.scenario_path = os.path.join(cur_file_path, '..', '..', 'data', 'race_scenarios', 'various_driving_data_20260224')
        self.scenario_path = os.path.join(cur_file_path, '..', '..', 'data', 'race_scenarios', 'various_500_sample')


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
        self.data = {}
        self.seq_map = []
        self.scene2map = {}
        self.data, self.seq_map = self.compile_scenarios(self.scenario_path)

        self.data_len = len(self.seq_map)

        print('Num scenes: %d' % (len(self.data)))
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
        # print('ninfo')
        # print(ninfo)
        # f_norm_mean = [0.5, 0.5]
        # f_norm_std = [0.5, 0.5]
        # self.fault_normalizer = MeanStdNormalizer(torch.Tensor(f_norm_mean),
        #                                           torch.Tensor(f_norm_std))
    
    def get_state_normalizer(self):
        return self.normalizer

    def get_att_normalizer(self):
        return self.veh_att_normalizer

    def compile_scenarios(self, scenario_path):
        fit_scenes = self.read_fit_scenes(scenario_path)

        scene2data = {}

        for scene in fit_scenes:
            sname = scene['name']

            if sname not in scene2data:
                scene2data[sname] = {}
                scene2data[sname]['ego'] = {'traj': [], 'w': 1.73,
                                            'l': 4.084, 'k': 'ego'}
                scene2data[sname]['sur'] = {'traj': [], 'w': 1.73,
                                            'l': 4.084, 'k': 'sur'}
                # print(torch.stack([scene['ego_quat'][3],scene['ego_quat'][0],scene['ego_quat'][1],scene['ego_quat'][2]],dim=0))
                for i in range(len(scene['ego_quat'][0])):
                    rot = Quaternion(scene['ego_quat'][3][i],scene['ego_quat'][0][i],scene['ego_quat'][1][i],scene['ego_quat'][2][i]).rotation_matrix
                    rot = np.arctan2(rot[1, 0], rot[0, 0])
                    scene2data[sname]['ego']['traj'].append({
                        'x': scene['ego_pos'][0][i],
                        'y': scene['ego_pos'][1][i],
                        'h' : rot,
                        'hcos': np.cos(rot),
                        'hsin': np.sin(rot),
                        't': scene['t'][i],
                    })              

                    rot = Quaternion(scene['sur_quat'][3][i],scene['sur_quat'][0][i],scene['sur_quat'][1][i],scene['sur_quat'][2][i]).rotation_matrix
                    rot = np.arctan2(rot[1, 0], rot[0, 0])
                    scene2data[sname]['sur']['traj'].append({
                        'x': scene['sur_pos'][0][i],
                        'y': scene['sur_pos'][1][i],
                        'h' : rot,
                        'hcos': np.cos(rot),
                        'hsin': np.sin(rot),
                        't': scene['t'][i],
                    })    

        return self.post_process(scene2data)
    
    
    def read_fit_scenes(self,scenario_path):
        
        # scene_flist = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))
        # scene_list = []
        # for scene_fpath in scene_flist:
        
        scene_full_list = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))
        n = len(scene_full_list)

        # # --- Old split (50% / 16.7% / 33.3%) ---
        # half = n // 2
        # rest = n - half
        # val_num = rest // 3
        # test_num = rest - val_num
        #
        # if self.split == "train":
        #     scene_separated_list = scene_full_list[:half]
        # elif self.split == "val":
        #     scene_separated_list = scene_full_list[half:half+val_num]
        # elif self.split == "test":
        #     scene_separated_list = scene_full_list[half+val_num:]
        # elif self.split == "adv":
        #     scene_separated_list = scene_full_list
        # else:
        #     raise ValueError("split must be 'train', 'val', or 'test', or 'adv'")
        # --- End old split ---

        # --- New split (85% train / 10% val / 5% test) ---
        train_num = int(n * 0.85)
        val_num = int(n * 0.10)
        # test_num = remainder

        if self.split == "train":
            scene_separated_list = scene_full_list[:train_num]
        elif self.split == "val":
            scene_separated_list = scene_full_list[train_num:train_num+val_num]
        elif self.split == "test":
            scene_separated_list = scene_full_list[train_num+val_num:]
        elif self.split == "adv":
            scene_separated_list = scene_full_list
        else:
            raise ValueError("split must be 'train', 'val', or 'test', or 'adv'")
        
        # scene_separated_list = scene_full_list

        scene_list = []
        for scene_fpath in scene_separated_list:
            
            scene_name = scene_fpath.split('/')[-1][:-5]
            print('Loading %s...' % (scene_name))
            dict = None
            with open(scene_fpath, 'r') as f:
                dict = pd.read_excel(scene_fpath, sheet_name='driving_data').values.tolist()
                dict = np.array(dict).T
            if dict is None:
                print('Failed to load! Skipping')
                continue
            
            cur_scene = {
                'name' : scene_name,
                'dt' : self.dt
            }
            
            # cur_scene = {
            #     'name' : scene_name,
            #     'dt' : self.dt
            # }
            
            
            # cur_scene['t'] = torch.tensor(dict[0][:])
            # cur_scene['ego_pos'] = torch.tensor(dict[1:4][:])
            # cur_scene['ego_quat'] = torch.tensor(dict[4:8][:])
            # cur_scene['ego_twist_lin'] = torch.tensor(dict[8:11][:])
            # cur_scene['ego_twist_ang'] = torch.tensor(dict[11:14][:])
            # cur_scene['sur_pos'] = torch.tensor(dict[14:17][:])
            # cur_scene['sur_quat'] = torch.tensor(dict[17:21][:])
            # cur_scene['sur_twist_lin'] = torch.tensor(dict[21:24][:])
            # cur_scene['sur_twist_ang'] = torch.tensor(dict[24:27][:])
            
            start_idx = 0

            cur_scene['t'] = torch.tensor(dict[0][start_idx:])
            cur_scene['ego_pos'] = torch.tensor([row[start_idx:] for row in dict[1:4]])
            cur_scene['ego_quat'] = torch.tensor([row[start_idx:] for row in dict[4:8]])
            cur_scene['ego_twist_lin'] = torch.tensor([row[start_idx:] for row in dict[8:11]])
            cur_scene['ego_twist_ang'] = torch.tensor([row[start_idx:] for row in dict[11:14]])
            cur_scene['sur_pos'] = torch.tensor([row[start_idx:] for row in dict[14:17]])
            cur_scene['sur_quat'] = torch.tensor([row[start_idx:] for row in dict[17:21]])
            cur_scene['sur_twist_lin'] = torch.tensor([row[start_idx:] for row in dict[21:24]])
            cur_scene['sur_twist_ang'] = torch.tensor([row[start_idx:] for row in dict[24:27]])

            if len(dict[0])>10:
                scene_list.append(cur_scene)

        return scene_list


    def post_process(self, data):
        scene2info = {}
        print('Post-processing data...')
        drivable_raster = self.map_env.nusc_raster[:, 0]
        # print("drivable_raster is", type(drivable_raster))
        # print(torch.sum(drivable_raster))
        seq_map = [] # for deterministic iteration through data maps from data_idx -> (scene_name, start_idx)
        for scene in tqdm(data):
            scene2info[scene] = {}

            challenge_inst_samp_list = None
            # if self.use_challenge_splits:
            #     challenge_inst_samp_list = self.pred_challenge_scenes.get(scene, [])
            #     challenge_inst_samp_list = {chall_inst_samp : True for chall_inst_samp in challenge_inst_samp_list}

            # first process ego info so we know all timestamps (since always available)
            ego_data = data[scene]['ego']
            # map timestamps -> frame idx to aggregate other incomplete agents
            ego_t_map = {row['t'].item() : ridx for ridx, row in enumerate(ego_data['traj'])}
            T = len(ego_t_map)
            # state with no vel
            ego_x = np.array([[row['x'].item(), row['y'].item(), row['hcos'].item(), row['hsin'].item()]
                                    for row in ego_data['traj']])
            ego_h = np.array([row['h'].item() for row in ego_data['traj']]) # heading angle
            ego_t = np.array([row['t'].item() for row in ego_data['traj']])
            # compute speed
            ego_pos = ego_x[:, :2]
            ego_vel = nutils.velocity(ego_pos, ego_t)
            ego_s = np.linalg.norm(ego_vel, axis=1).reshape((-1, 1))
            ego_a = np.linalg.norm(nutils.velocity(ego_vel, ego_t), axis=1).reshape((-1, 1))
            # compute hdot
            ego_hdot = nutils.heading_change_rate(ego_h, ego_t).reshape((-1, 1))
            ego_ddh = nutils.heading_change_rate(ego_hdot.reshape((-1)), ego_t).reshape((-1, 1))
            # print(ego_pos)
            # print(ego_vel)
            # print(ego_hdot)
            # form state and record valid frames

            ego_traj = np.concatenate([ego_x, ego_s, ego_hdot], axis=1)
            ego_accel = np.concatenate([ego_a, ego_ddh], axis=1)
            ego_is_vis = np.logical_not(np.isnan(ego_s.flatten())).astype(np.int32)
            # vehicle attributes
            ego_lw = np.array([ego_data['l'], ego_data['w']])
            
            ego_info = None
            # if self.use_challenge_splits:
            #     # have to add dummy data for a few steps at the beginning b/c they use data at first few steps
            #     nadd_steps = 2 # NOTE assumes at least 4 past steps are needed, won't work for > 4
            #     ego_info = {
            #         'traj' : np.concatenate([np.ones((nadd_steps, 6))*np.nan, ego_traj], axis=0),
            #         'accel' : np.concatenate([np.ones((nadd_steps, 2))*np.nan, ego_accel], axis=0),
            #         'lw' :  ego_lw,
            #         'is_vis' : np.concatenate([np.zeros((nadd_steps), dtype=np.int32), ego_is_vis], axis=0),
            #         'k' : 'ego',
            #     }
            # else:
            ego_info = {
                'traj' : ego_traj,
                'accel' : ego_accel,
                'lw' :  ego_lw,
                'is_vis' : ego_is_vis,
                'k' : 'ego',
            }
                
            scene2info[scene]['ego'] = ego_info

            # now process other agents
            # first map onto same timeline as ego with nans at other spots
            #   also compute velocities.
            for name in data[scene]:
                if name == 'ego':
                    continue
                info = {}
                inst_tok = name
                t_list = [row['t'] for row in data[scene][name]['traj']]
                x_list = np.array([[row['x'], row['y'], row['hcos'], row['hsin']]
                                    for row in data[scene][name]['traj']])
                h_list = np.array([row['h'] for row in data[scene][name]['traj']])
                lw = np.array([data[scene][name]['l'], data[scene][name]['w']]) # veh attribs

                #  check if in challenge split, if so need to add to data no matter what
                in_chall_split = np.zeros((T), dtype=bool)


                valid_frame = np.ones((x_list.shape[0]), dtype=bool) # all valid by default
                if not self.use_challenge_splits or np.sum(in_chall_split) == 0: # if using challenge split, need all frames of any vehicles that we need to make a pred for.
                # #     # check if on drivable area (i.e. layer 0 of maps) at each frame
                    torch_xlist = torch.from_numpy(x_list).to(drivable_raster.device)
                    torch_lw = torch.from_numpy(lw).to(drivable_raster.device).unsqueeze(0).expand(x_list.shape[0], 2)
                    print(x_list.shape[0])
                    # mapixes =  torch.Tensor([self.scene2map[scene][1]]).to(drivable_raster.device).long().expand(x_list.shape[0])
                    mapixes =  torch.Tensor([0]).to(drivable_raster.device).long().expand(x_list.shape[0])                
                    drivable_frac = nutils.check_on_layer(drivable_raster,
                                                            self.map_env.nusc_dx,
                                                            torch_xlist,
                                                            torch_lw,
                                                            mapixes)
                    
                    # print("drivable_frac", drivable_frac)
                    
                    #---------------------------Valid Filtering------------------------------
                    # valid_frame = (drivable_frac >= 0.3).cpu().numpy()
                    valid_frame = np.ones_like(drivable_frac.cpu().numpy(), dtype=bool)
                    #---------------------------Valid Filtering------------------------------


                cur_x = np.ones_like(ego_x)*np.nan
                cur_h = np.ones_like(ego_h)*np.nan
                # only have values at observed frames

                for t, x, h, keep in zip(t_list, x_list, h_list, valid_frame):
                    t = t.item()
                    h = h.item()
      
                    if keep: # only keep frames on drivable and not in parking lot
                        cur_x[ego_t_map[t]] = x
                        cur_h[ego_t_map[t]] = h

                # if all frames nan (never on drivable surface), throw it out
                if np.sum(np.isnan(cur_x), axis=0)[0] == cur_x.shape[0]:
                    continue

                # compute speed
                pos = cur_x[:, :2]
                vel = nutils.velocity(pos, ego_t)
                s = np.linalg.norm(vel, axis=1).reshape((-1, 1))
                a = np.linalg.norm(nutils.velocity(vel, ego_t), axis=1).reshape((-1, 1))
                # compute hdot
                hdot = nutils.heading_change_rate(cur_h, ego_t).reshape((-1, 1))
                ddh = nutils.heading_change_rate(hdot.reshape((-1)), ego_t).reshape((-1, 1))
                # form state and record valid frames
                no_vis = np.isnan(s.flatten())
                # some position values might be available while vel is nan
                #       (when single random frame shows up)
                cur_x[no_vis] = np.nan
                is_vis = np.logical_not(no_vis).astype(np.float32)
                traj = np.concatenate([cur_x, s, hdot], axis=1)
                accel = np.concatenate([a, ddh], axis=1)
                # print(traj)
                info = None
                nadd_steps = 2
                if self.use_challenge_splits:
                    info = {
                        'traj' : np.concatenate([np.ones((nadd_steps, 6))*np.nan, traj], axis=0),
                        'accel' : np.concatenate([np.ones((nadd_steps, 2))*np.nan, accel], axis=0),
                        'lw' :  lw,
                        'is_vis' : np.concatenate([np.zeros((nadd_steps), dtype=np.int32), is_vis], axis=0),
                        'k' : data[scene][name]['k']
                    }
                else:
                    info = {
                        'traj' : traj,
                        'accel' : accel,
                        'lw' :  lw,
                        'is_vis' : is_vis,
                        'k' : data[scene][name]['k']
                    }
                scene2info[scene][name] = info

                
            if not self.use_challenge_splits:
                # update data map
                scene_seq = [(scene, start_idx) for start_idx in range(0, T - self.seq_len, self.seq_interval)]
                seq_map.extend(scene_seq)

        return scene2info, seq_map

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        idx_info = self.seq_map[idx]
        inst_tok = None
        if not self.use_challenge_splits:
            scene_name, sidx = idx_info
        else:
            scene_name, sidx, inst_tok = idx_info
        eidx = sidx + self.seq_len
        midx = sidx + self.npast
        # _, map_idx = self.scene2map[scene_name]
        # _, map_idx = ('HMCL_RACING',0)
        _, map_idx = ('boston_seaport',0)

        # NOTE only keep an agent in the sequence if it has an annotation
        #       at last frame of the past.
        #       This is not perfect since past/future-only agents will certainly affect traffic

        # always put ego at node 0
        ego_data = self.data[scene_name]['ego']
        past = [ego_data['traj'][sidx:midx, :]]
        future = [ego_data['traj'][midx:eidx, :]]
        sem = [self.cat2vec['car']] # one-hot vec
        lw = [ego_data['lw']]
        past_vis = [ego_data['is_vis'][sidx:midx]]
        fut_vis = [ego_data['is_vis'][midx:eidx]]
        

        # if self.use_challenge_splits:
        #     # prepend data for the agent we're making a prediction for
        #     # so ego is not at 0, challenge data is
        #     agent_data = self.data[scene_name][inst_tok]
        #     assert(np.isnan(agent_data['traj'][midx-1]).astype(np.int32).sum() == 0) # should not be nan if we're making a prediction for it
        #     past = [agent_data['traj'][sidx:midx, :]] + past
        #     future = [agent_data['traj'][midx:eidx, :]] + future
        #     sem = [self.cat2vec[agent_data['k']]] + sem # one-hot vec
        #     lw = [agent_data['lw']] + lw
        #     past_vis = [agent_data['is_vis'][sidx:midx]] + past_vis
        #     fut_vis = [agent_data['is_vis'][midx:eidx]] + fut_vis

        for agent in self.data[scene_name]:
            if agent == 'ego':
                continue
            # if self.use_challenge_splits and agent == inst_tok:
            #     continue
            agent_data = self.data[scene_name][agent]
            if np.isnan(agent_data['traj'][midx-1]).astype(np.int32).sum() > 0:
                continue
            if self.require_full_past and np.isnan(agent_data['traj'][:midx]).sum() > 0:
                # has some nan in past
                continue

            # have a valid agent, add info
            # may be nan at many frames, this must be dealt with in model
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
        # print(past)

        # normalize
        past_gt = self.normalizer.normalize(past) # gt past (no noise)
        past = self.normalizer.normalize(past)
        future_gt = self.normalizer.normalize(future) # gt future (used to compute err/loss)
        future = self.normalizer.normalize(future) # observed future (input to net)
        lw = self.veh_att_normalizer.normalize(lw)
        # fault = self.fault_normalizer.normalize(fault)
        # print(past)

        # # add noise if desired
        # if self.noise_std > 0:
        #     past += torch.randn_like(past)*self.noise_std
        #     future += torch.randn_like(future)*self.noise_std
        #     # make sure heading is still a unit vector
        #     past[:, :, 2:4] = past[:, :, 2:4] / torch.norm(past[:, :, 2:4], dim=-1, keepdim=True)
        #     future[:, :, 2:4] = future[:, :, 2:4] / torch.norm(future[:, :, 2:4], dim=-1, keepdim=True)
        #     # make sure position is still positive
        #     past[:, :, :2] = torch.clamp(past[:, :, :2], min=0.0)
        #     future[:, :, :2] = torch.clamp(future[:, :, :2], min=0.0)
        #     # also for vehicle attributes
        #     lw += torch.randn_like(lw)*self.noise_std
        #     # print(past)

        #  then build fully-connected scene graph
        NA = past.size(0)
        edge_index = None
        if NA > 1:
            node_list = range(NA)
            edge_index = list(itertools.product(node_list, node_list))
            edge_index_list = [(i, j) for i, j in edge_index if i != j]
            edge_index = torch.Tensor(edge_index_list).T.to(torch.long).contiguous()
        else:
            edge_index = torch.Tensor([[],[]]).long()

        graph_prop_dict = {
            'x' : torch.empty((NA,)),
            'pos' : torch.empty((NA,)),
            'edge_index' : edge_index,
            'past' : past,
            'past_gt' : past_gt,
            'future' : future,
            'future_gt' : future_gt,
            'sem' : sem,
            'lw' : lw,
            'past_vis' : past_vis,
            'future_vis' : fut_vis,
            'scene_name': scene_name
        }
        scene_graph = Graph(**graph_prop_dict)

        return scene_graph, map_idx
