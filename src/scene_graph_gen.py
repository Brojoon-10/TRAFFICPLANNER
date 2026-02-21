# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import tqdm
import torch
import numpy as np

from torch_geometric.data import DataLoader as GraphDataLoader

from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.logger import Logger
from utils.torch import get_device
from utils.common import mkdir

def run_one_epoch(data_loader):
    '''
    Run through dataset and find possible scenarios.
    '''
    pbar_data = tqdm.tqdm(data_loader)

    for i, data in enumerate(pbar_data):
        scene_graph, map_idx = data
        save_root = "./src/scene_graph/toolkit_dataset/half"      
        os.makedirs(save_root, exist_ok=True)
        torch.save(scene_graph, os.path.join(save_root, f"scene_{i:06d}.pt"))
        continue

def main():
    # Config values directly set
    split = 'val'
    val_size = 400
    seq_interval = 10
    shuffle = False
    num_workers = 0
    batch_size = 1
    
    # create output directory and logging

    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    # map and data path setup
    # data_path = '/home/hj/Carla_HB/carla-autoware-universe/src/carla_map/maps/vector_maps/lanelet2/HMCL_Racing_Odom_shifted.osm'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps', 'centerline_added_boston.osm')

    map_env = FITMapEnv(data_path,device=device)
    
    test_dataset = FITDataset(
        data_path,
        map_env,
        split=split,
        seq_interval=seq_interval,
        randomize_val=True,
        val_size=val_size,
        reduce_cats=False
    )

    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=lambda _: np.random.seed()
    )

    run_one_epoch(test_loader)

if __name__ == "__main__":
    main()
