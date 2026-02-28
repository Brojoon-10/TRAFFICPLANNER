"""
CARLA 데이터의 실제 분포를 분석하여 NUSC_NORM_STATS와 비교.
100개 시나리오를 로드하여 speed, acc, hdot, ddh, position scale 등을 측정.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import glob
import pandas as pd
from pyquaternion import Quaternion
import datasets.nuscenes_utils as nutils

scenario_path = './data/race_scenarios/various_500_sample'
scene_files = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))

# Use first 100 files
scene_files = scene_files[:100]
print(f"Analyzing {len(scene_files)} scenarios...")

dt = 0.5  # 2 Hz

all_speed = []
all_hdot = []
all_acc = []  # speed derivative (scalar acceleration)
all_ddh = []  # hdot derivative
all_dx = []   # inter-step displacement x
all_dy = []   # inter-step displacement y
all_lscale = []  # relative displacement in local frame

for fpath in scene_files:
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T

    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            pos_rows = data[1:4]
            quat_rows = data[4:8]
        else:
            pos_rows = data[14:17]
            quat_rows = data[17:21]

        T = len(data[0])
        t_arr = np.array(data[0])

        # Position: x, y
        x_arr = pos_rows[0]
        y_arr = pos_rows[1]

        # Heading from quaternion
        h_arr = np.zeros(T)
        hcos_arr = np.zeros(T)
        hsin_arr = np.zeros(T)
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])
            hcos_arr[i] = np.cos(h_arr[i])
            hsin_arr[i] = np.sin(h_arr[i])

        # Compute speed (same as dataset code: velocity norm)
        pos_2d = np.stack([x_arr, y_arr], axis=1)  # (T, 2)
        vel = nutils.velocity(pos_2d, t_arr)  # (T, 2)
        speed = np.linalg.norm(vel, axis=1)  # (T,)

        # Compute hdot (same as dataset code)
        hdot = nutils.heading_change_rate(h_arr, t_arr)  # (T,)

        # Compute acc (speed derivative via velocity of velocity)
        acc_vec = nutils.velocity(vel, t_arr)  # (T, 2)
        acc = np.linalg.norm(acc_vec, axis=1)  # (T,)

        # Compute ddh
        ddh = nutils.heading_change_rate(hdot, t_arr)  # (T,)

        # Compute local frame displacement (lscale)
        for i in range(1, T):
            dx_global = x_arr[i] - x_arr[i-1]
            dy_global = y_arr[i] - y_arr[i-1]
            # Transform to local frame of previous step
            cos_h = hcos_arr[i-1]
            sin_h = hsin_arr[i-1]
            dx_local = cos_h * dx_global + sin_h * dy_global
            dy_local = -sin_h * dx_global + cos_h * dy_global
            all_lscale.append(dx_local)
            all_lscale.append(dy_local)

        # Filter out NaN
        valid = ~np.isnan(speed)
        all_speed.extend(speed[valid].tolist())

        valid = ~np.isnan(hdot)
        all_hdot.extend(hdot[valid].tolist())

        valid = ~np.isnan(acc)
        all_acc.extend(acc[valid].tolist())

        valid = ~np.isnan(ddh)
        all_ddh.extend(ddh[valid].tolist())

all_speed = np.array(all_speed)
all_hdot = np.array(all_hdot)
all_acc = np.array(all_acc)
all_ddh = np.array(all_ddh)
all_lscale = np.array(all_lscale)

print("\n" + "="*70)
print("CARLA DATA STATISTICS (100 scenarios, ego + sur)")
print("="*70)

def print_stats(name, arr):
    print(f"\n--- {name} ---")
    print(f"  count: {len(arr)}")
    print(f"  mean:  {np.mean(arr):.6f}")
    print(f"  std:   {np.std(arr):.6f}")
    print(f"  min:   {np.min(arr):.6f}")
    print(f"  max:   {np.max(arr):.6f}")
    print(f"  p1:    {np.percentile(arr, 1):.6f}")
    print(f"  p5:    {np.percentile(arr, 5):.6f}")
    print(f"  p25:   {np.percentile(arr, 25):.6f}")
    print(f"  p50:   {np.percentile(arr, 50):.6f}")
    print(f"  p75:   {np.percentile(arr, 75):.6f}")
    print(f"  p95:   {np.percentile(arr, 95):.6f}")
    print(f"  p99:   {np.percentile(arr, 99):.6f}")

print_stats("speed (m/s)", all_speed)
print_stats("hdot (rad/s)", all_hdot)
print_stats("acc (m/s²)", all_acc)
print_stats("ddh (rad/s²)", all_ddh)
print_stats("lscale (local frame disp, m)", all_lscale)

print("\n" + "="*70)
print("NUSC_NORM_STATS for comparison")
print("="*70)
print(f"  lscale: mean=0.0, std=15.0")
print(f"  h:      mean=0.0, std=1.0 (unit vector, no normalization)")
print(f"  s:      mean=1.802009, std=3.507907")
print(f"  hdot:   mean=-0.000037, std=0.055684")
print(f"  a:      mean=0.409074, std=1.045530")
print(f"  ddh:    mean=0.000046, std=0.075032")
print(f"  l:      mean=4.844294, std=1.084860")
print(f"  w:      mean=2.021752, std=0.299647")

print("\n" + "="*70)
print("NORMALIZED VALUES (using NUSC stats)")
print("="*70)

# How CARLA data looks after NUSC normalization
nusc_s_mean, nusc_s_std = 1.802009, 3.507907
nusc_hdot_mean, nusc_hdot_std = -0.000037, 0.055684
nusc_a_mean, nusc_a_std = 0.409074, 1.045530
nusc_ddh_mean, nusc_ddh_std = 0.000046, 0.075032
nusc_lscale_mean, nusc_lscale_std = 0.0, 15.0

speed_norm = (all_speed - nusc_s_mean) / nusc_s_std
hdot_norm = (all_hdot - nusc_hdot_mean) / nusc_hdot_std
acc_norm = (all_acc - nusc_a_mean) / nusc_a_std
ddh_norm = (all_ddh - nusc_ddh_mean) / nusc_ddh_std
lscale_norm = (all_lscale - nusc_lscale_mean) / nusc_lscale_std

print_stats("speed_normalized (NUSC)", speed_norm)
print_stats("hdot_normalized (NUSC)", hdot_norm)

print("\n" + "="*70)
print("INTENT ANALYSIS: acc/ddh in prototype space")
print("="*70)
# Intent uses: acc / a_std, yaw_rate / hdot_std
# Prototypes at [-1, 0, 1] × [-1, 0, 1]
intent_acc = all_acc / nusc_a_std  # NOT subtracting mean — check code
intent_yaw = all_hdot / nusc_hdot_std

print_stats("intent_acc (raw_acc / a_std)", intent_acc)
print_stats("intent_yaw (raw_hdot / hdot_std)", intent_yaw)

# What fraction falls outside [-1, 1] range
acc_outside = np.mean(np.abs(intent_acc) > 1.0) * 100
yaw_outside = np.mean(np.abs(intent_yaw) > 1.0) * 100
print(f"\n  % acc outside [-1,1]: {acc_outside:.1f}%")
print(f"  % yaw outside [-1,1]: {yaw_outside:.1f}%")

# What fraction falls outside [-1.5, 1.5] range
acc_outside_15 = np.mean(np.abs(intent_acc) > 1.5) * 100
yaw_outside_15 = np.mean(np.abs(intent_yaw) > 1.5) * 100
print(f"  % acc outside [-1.5,1.5]: {acc_outside_15:.1f}%")
print(f"  % yaw outside [-1.5,1.5]: {yaw_outside_15:.1f}%")

# Distribution across 9 intent classes
print("\n--- 9-class intent distribution ---")
acc_bins = [-float('inf'), -1/3, 1/3, float('inf')]  # 3 bins at prototype boundaries
yaw_bins = [-float('inf'), -1/3, 1/3, float('inf')]
acc_class = np.digitize(intent_acc, [-1/3, 1/3])  # 0, 1, 2
yaw_class = np.digitize(intent_yaw, [-1/3, 1/3])  # 0, 1, 2
intent_class = acc_class * 3 + yaw_class  # 0~8

labels = ['dec+left', 'dec+str', 'dec+right',
          'mnt+left', 'mnt+str', 'mnt+right',
          'acc+left', 'acc+str', 'acc+right']
for i in range(9):
    frac = np.mean(intent_class == i) * 100
    print(f"  class {i} ({labels[i]}): {frac:.1f}%")
