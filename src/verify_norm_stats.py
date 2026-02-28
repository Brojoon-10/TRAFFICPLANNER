"""
검증: dataset post_process와 동일한 방식으로 state를 계산하고,
NUSC_NORM_STATS로 정규화한 결과를 확인.
dataset 로드 없이 직접 계산하여 교차 검증.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import glob
import pandas as pd
from pyquaternion import Quaternion
import datasets.nuscenes_utils as nutils
from datasets.utils import MeanStdNormalizer, NUSC_NORM_STATS

# === NUSC normalizer (현재 사용 중) ===
ninfo = NUSC_NORM_STATS[('car', 'truck')]
norm_mean = [ninfo['lscale'][0], ninfo['lscale'][0], ninfo['h'][0], ninfo['h'][0], ninfo['s'][0], ninfo['hdot'][0]]
norm_std = [ninfo['lscale'][1], ninfo['lscale'][1], ninfo['h'][1], ninfo['h'][1], ninfo['s'][1], ninfo['hdot'][1]]
normalizer = MeanStdNormalizer(torch.Tensor(norm_mean), torch.Tensor(norm_std))
print(f"NUSC normalizer:")
print(f"  mean: {norm_mean}")
print(f"  std:  {norm_std}")

# === Load data exactly like post_process ===
scenario_path = './data/race_scenarios/various_500_sample'
scene_files = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))[:100]

# State = (x, y, hcos, hsin, speed, hdot) — same as dataset
# Note: x, y are GLOBAL positions (the state before transform2frame)
# After transform2frame, they become LOCAL (relative to agent frame at that timestep)
# The normalizer divides x,y by lscale_std=15

# But in the actual training pipeline:
# 1. post_process: compute state (x_global, y_global, hcos, hsin, speed, hdot)
# 2. __getitem__: slices subsequence, normalizes with normalizer
# 3. normalize_scene_graph: transforms to local frame using transform2frame
# So x,y after normalization are LOCAL frame coords (relative displacement / lscale_std)

# To properly check, I need to see what the LOCAL frame x,y values look like
# Local frame: relative to last past position of ego

all_raw_speed = []
all_raw_hdot = []
all_raw_local_x = []  # after transform2frame, before normalization
all_raw_local_y = []

npast = 4
nfuture = 12
seq_len = npast + nfuture
dt = 0.5

for fidx, fpath in enumerate(scene_files):
    if fidx % 20 == 0:
        print(f"Processing {fidx}/{len(scene_files)}...")
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    t_arr = data[0]

    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            pos_rows, quat_rows = data[1:4], data[4:8]
        else:
            pos_rows, quat_rows = data[14:17], data[17:21]

        x_arr, y_arr = pos_rows[0], pos_rows[1]
        h_arr = np.zeros(T)
        hcos_arr = np.zeros(T)
        hsin_arr = np.zeros(T)
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])
            hcos_arr[i] = np.cos(h_arr[i])
            hsin_arr[i] = np.sin(h_arr[i])

        pos_2d = np.stack([x_arr, y_arr], axis=1)
        vel = nutils.velocity(pos_2d, np.array(t_arr))
        speed = np.linalg.norm(vel, axis=1)
        hdot = nutils.heading_change_rate(h_arr, np.array(t_arr))

        # Simulate subsequence extraction + local frame transform
        # For each valid subsequence of length seq_len
        seq_interval = 5
        for start in range(0, T - seq_len + 1, seq_interval):
            end = start + seq_len
            # Reference frame: last past step (index: start + npast - 1)
            ref_idx = start + npast - 1
            ref_x = x_arr[ref_idx]
            ref_y = y_arr[ref_idx]
            ref_hcos = hcos_arr[ref_idx]
            ref_hsin = hsin_arr[ref_idx]

            # Transform global positions to local frame
            for t in range(start, end):
                dx = x_arr[t] - ref_x
                dy = y_arr[t] - ref_y
                # Rotate to ref frame
                local_x = ref_hcos * dx + ref_hsin * dy
                local_y = -ref_hsin * dx + ref_hcos * dy
                all_raw_local_x.append(local_x)
                all_raw_local_y.append(local_y)

            valid = ~np.isnan(speed[start:end])
            all_raw_speed.extend(speed[start:end][valid].tolist())
            valid = ~np.isnan(hdot[start:end])
            all_raw_hdot.extend(hdot[start:end][valid].tolist())

all_raw_speed = np.array(all_raw_speed)
all_raw_hdot = np.array(all_raw_hdot)
all_raw_local_x = np.array(all_raw_local_x)
all_raw_local_y = np.array(all_raw_local_y)

def pr(name, arr):
    print(f"  {name}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
          f"min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
          f"p1={np.percentile(arr, 1):.4f}, p99={np.percentile(arr, 99):.4f}")

print(f"\n{'='*70}")
print(f"RAW VALUES (before normalization)")
print(f"{'='*70}")
pr("local_x (m)", all_raw_local_x)
pr("local_y (m)", all_raw_local_y)
pr("speed (m/s)", all_raw_speed)
pr("hdot (rad/s)", all_raw_hdot)

print(f"\n{'='*70}")
print(f"AFTER NUSC NORMALIZATION (current)")
print(f"{'='*70}")
# x,y: (val - 0) / 15
norm_x = all_raw_local_x / 15.0
norm_y = all_raw_local_y / 15.0
# speed: (val - 1.802) / 3.508
norm_speed = (all_raw_speed - 1.802009) / 3.507907
# hdot: (val - (-0.000037)) / 0.055684
norm_hdot = (all_raw_hdot - (-0.000037)) / 0.055684

pr("norm_x", norm_x)
pr("norm_y", norm_y)
pr("norm_speed", norm_speed)
pr("norm_hdot", norm_hdot)

print(f"\n{'='*70}")
print(f"IDEAL: all normalized values should be roughly in [-3, 3]")
print(f"{'='*70}")
for name, arr in [("norm_x", norm_x), ("norm_y", norm_y),
                   ("norm_speed", norm_speed), ("norm_hdot", norm_hdot)]:
    outside_3 = np.mean(np.abs(arr) > 3.0) * 100
    outside_5 = np.mean(np.abs(arr) > 5.0) * 100
    outside_10 = np.mean(np.abs(arr) > 10.0) * 100
    print(f"  {name}: {outside_3:.1f}% > |3|, {outside_5:.1f}% > |5|, {outside_10:.1f}% > |10|")

# === With CARLA stats ===
print(f"\n{'='*70}")
print(f"AFTER CARLA NORMALIZATION (proposed)")
print(f"{'='*70}")
# lscale: determine from local_x, local_y
# mean MUST be 0, use actual std of local displacements
all_local = np.concatenate([all_raw_local_x, all_raw_local_y])
carla_lscale_std = np.std(all_local)
print(f"  local position std: {carla_lscale_std:.4f} m")

# But wait - local_x has strong forward bias (agent moves forward)
# local_y is more symmetric
print(f"  local_x: mean={np.mean(all_raw_local_x):.4f}, std={np.std(all_raw_local_x):.4f}")
print(f"  local_y: mean={np.mean(all_raw_local_y):.4f}, std={np.std(all_raw_local_y):.4f}")
# The lscale_std should be large enough to capture the full range
# Using max(|x|, |y|) / 3 as a heuristic
max_range = max(np.percentile(np.abs(all_raw_local_x), 99),
                np.percentile(np.abs(all_raw_local_y), 99))
print(f"  p99 max(|x|, |y|): {max_range:.1f}m → suggested lscale_std = {max_range/3:.1f}m")

carla_norm_x = all_raw_local_x / carla_lscale_std
carla_norm_y = all_raw_local_y / carla_lscale_std
carla_norm_speed = (all_raw_speed - np.mean(all_raw_speed)) / np.std(all_raw_speed)
carla_norm_hdot = (all_raw_hdot - np.mean(all_raw_hdot)) / np.std(all_raw_hdot)

pr("carla_norm_x", carla_norm_x)
pr("carla_norm_y", carla_norm_y)
pr("carla_norm_speed", carla_norm_speed)
pr("carla_norm_hdot", carla_norm_hdot)

for name, arr in [("carla_norm_x", carla_norm_x), ("carla_norm_y", carla_norm_y),
                   ("carla_norm_speed", carla_norm_speed), ("carla_norm_hdot", carla_norm_hdot)]:
    outside_3 = np.mean(np.abs(arr) > 3.0) * 100
    outside_5 = np.mean(np.abs(arr) > 5.0) * 100
    print(f"  {name}: {outside_3:.1f}% > |3|, {outside_5:.1f}% > |5|")
