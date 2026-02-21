import pandas as pd
import glob
import os
import torch
import numpy as np

def read_fit_scenes(scenario_path):
    scene_flist = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario*.xlsx')))
    scene_list = []
    for scene_fpath in scene_flist:
        scene_name = scene_fpath.split('/')[-1][:-5]
        print('Loading %s...' % (scene_name))
        dict = None
        with open(scene_fpath, 'r') as f:
            dict = pd.read_excel(scene_fpath, sheet_name='driving_data').values.tolist()
            dict = np.array(dict).T
            print(dict)
        if dict is None:
            print('Failed to load! Skipping')
            continue
        
        cur_scene = {
            'name' : scene_name,
            'dt' : 0.5
        }
        cur_scene['ego_pos'] = torch.tensor(dict[1:4][:])
        cur_scene['ego_quat'] = torch.tensor(dict[4:8][:])
        cur_scene['ego_twist_lin'] = torch.tensor(dict[8:11][:])
        cur_scene['ego_twist_ang'] = torch.tensor(dict[11:14][:])
        cur_scene['sur_pos'] = torch.tensor(dict[14:17][:])
        cur_scene['sur_quat'] = torch.tensor(dict[17:21][:])
        cur_scene['sur_twist_lin'] = torch.tensor(dict[21:24][:])
        cur_scene['sur_twist_ang'] = torch.tensor(dict[24:27][:])
        cur_scene['fault'] = torch.tensor(dict[27:29][:])
        scene_list.append(cur_scene)

    return scene_list

read_fit_scenes("/home/user/STRIVE/data/fit_scenarios")