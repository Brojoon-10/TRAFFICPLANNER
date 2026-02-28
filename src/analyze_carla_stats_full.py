"""
CARLA 데이터 전체(500개)의 정확한 통계를 계산.
dataset의 post_process와 동일한 방식으로 speed, hdot, acc, ddh를 계산.

목표: CARLA_NORM_STATS 값 결정

주의:
- lscale는 mean=0 필수 (transform2frame에서 뺄셈 연산의 정확성 보장)
- h는 unit vector이므로 mean=0, std=1 유지
- acc/ddh는 부호 있는 (signed) 값이어야 함 → 현재 acc는 norm(velocity derivative)로 항상 ≥0 → signed acc 추가 계산
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
print(f"Analyzing ALL {len(scene_files)} scenarios...")

dt = 0.5

# Raw values (before any normalization)
all_speed = []       # scalar speed (m/s) — dataset dim 4
all_hdot = []        # heading change rate (rad/s) — dataset dim 5
all_acc_unsigned = [] # |velocity derivative| (m/s²) — dataset's acc
all_ddh = []         # hdot derivative (rad/s²) — dataset's ddh
all_lscale = []      # local frame displacement (m) — dataset dims 0,1

# Signed acceleration: speed_next - speed_cur / dt
all_signed_acc = []
# Vehicle attributes
all_length = []
all_width = []

for fidx, fpath in enumerate(scene_files):
    if fidx % 50 == 0:
        print(f"  Processing {fidx}/{len(scene_files)}...")
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

        x_arr = pos_rows[0]
        y_arr = pos_rows[1]

        h_arr = np.zeros(T)
        hcos_arr = np.zeros(T)
        hsin_arr = np.zeros(T)
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])
            hcos_arr[i] = np.cos(h_arr[i])
            hsin_arr[i] = np.sin(h_arr[i])

        # === Speed (same as dataset: norm of velocity) ===
        pos_2d = np.stack([x_arr, y_arr], axis=1)
        vel = nutils.velocity(pos_2d, t_arr)
        speed = np.linalg.norm(vel, axis=1)

        # === hdot (same as dataset) ===
        hdot = nutils.heading_change_rate(h_arr, t_arr)

        # === acc unsigned (same as dataset: norm of acceleration vector) ===
        acc_vec = nutils.velocity(vel, t_arr)
        acc_unsigned = np.linalg.norm(acc_vec, axis=1)

        # === ddh (same as dataset) ===
        ddh = nutils.heading_change_rate(hdot, t_arr)

        # === Signed acceleration: (speed[i+1] - speed[i]) / (t[i+1] - t[i]) ===
        signed_acc = np.full(T, np.nan)
        for i in range(T - 1):
            if not (np.isnan(speed[i]) or np.isnan(speed[i+1])):
                signed_acc[i] = (speed[i+1] - speed[i]) / (t_arr[i+1] - t_arr[i])
        # Last step: copy from second-to-last (same as velocity does)
        if T > 1 and not np.isnan(signed_acc[T-2]):
            signed_acc[T-1] = signed_acc[T-2]

        # === Local frame displacement (lscale) ===
        for i in range(1, T):
            dx_global = x_arr[i] - x_arr[i-1]
            dy_global = y_arr[i] - y_arr[i-1]
            cos_h = hcos_arr[i-1]
            sin_h = hsin_arr[i-1]
            dx_local = cos_h * dx_global + sin_h * dy_global
            dy_local = -sin_h * dx_global + cos_h * dy_global
            all_lscale.append(dx_local)
            all_lscale.append(dy_local)

        # Collect valid values
        valid = ~np.isnan(speed)
        all_speed.extend(speed[valid].tolist())
        valid = ~np.isnan(hdot)
        all_hdot.extend(hdot[valid].tolist())
        valid = ~np.isnan(acc_unsigned)
        all_acc_unsigned.extend(acc_unsigned[valid].tolist())
        valid = ~np.isnan(ddh)
        all_ddh.extend(ddh[valid].tolist())
        valid = ~np.isnan(signed_acc)
        all_signed_acc.extend(signed_acc[valid].tolist())

        # Vehicle attributes (hardcoded in dataset as l=4.084, w=1.73)
        all_length.append(4.084)
        all_width.append(1.73)

all_speed = np.array(all_speed)
all_hdot = np.array(all_hdot)
all_acc_unsigned = np.array(all_acc_unsigned)
all_ddh = np.array(all_ddh)
all_lscale = np.array(all_lscale)
all_signed_acc = np.array(all_signed_acc)

def print_stats(name, arr):
    print(f"\n--- {name} (count={len(arr)}) ---")
    print(f"  mean:  {np.mean(arr):.6f}")
    print(f"  std:   {np.std(arr):.6f}")
    print(f"  min:   {np.min(arr):.6f}  max: {np.max(arr):.6f}")
    print(f"  p1:    {np.percentile(arr, 1):.6f}  p99: {np.percentile(arr, 99):.6f}")
    print(f"  p5:    {np.percentile(arr, 5):.6f}  p95: {np.percentile(arr, 95):.6f}")
    print(f"  p25:   {np.percentile(arr, 25):.6f}  p75: {np.percentile(arr, 75):.6f}")
    print(f"  p50:   {np.percentile(arr, 50):.6f}")

print("\n" + "="*70)
print("CARLA RAW DATA STATISTICS (all 500 scenarios, ego+sur)")
print("="*70)

print_stats("speed (m/s) — state dim 4", all_speed)
print_stats("hdot (rad/s) — state dim 5", all_hdot)
print_stats("signed_acc (m/s²) — for intent", all_signed_acc)
print_stats("acc_unsigned (m/s²) — dataset's acc", all_acc_unsigned)
print_stats("ddh (rad/s²) — dataset's ddh", all_ddh)
print_stats("lscale (m) — local disp, state dim 0,1", all_lscale)

print("\n" + "="*70)
print("COMPARISON: NUSC vs CARLA")
print("="*70)

print(f"\n{'Field':<20} {'NUSC mean':>12} {'NUSC std':>12} {'CARLA mean':>12} {'CARLA std':>12} {'Ratio(std)':>12}")
print("-" * 80)

comparisons = [
    ('lscale (x,y)', 0.0, 15.0, np.mean(all_lscale), np.std(all_lscale)),
    ('h (unit vec)', 0.0, 1.0, 0.0, 1.0),  # no change needed
    ('speed', 1.802009, 3.507907, np.mean(all_speed), np.std(all_speed)),
    ('hdot', -0.000037, 0.055684, np.mean(all_hdot), np.std(all_hdot)),
    ('acc (signed)', 0.409074, 1.045530, np.mean(all_signed_acc), np.std(all_signed_acc)),
    ('ddh', 0.000046, 0.075032, np.mean(all_ddh), np.std(all_ddh)),
]

for name, nm, ns, cm, cs in comparisons:
    ratio = cs / ns if ns > 0 else 0
    print(f"{name:<20} {nm:>12.6f} {ns:>12.6f} {cm:>12.6f} {cs:>12.6f} {ratio:>12.2f}x")

print("\n" + "="*70)
print("PROPOSED CARLA_NORM_STATS")
print("="*70)

# lscale: MUST keep mean=0 (transform2frame assumption)
# Use CARLA std for better normalization
lscale_std = np.std(all_lscale)
print(f"\n  lscale: (0.0, {lscale_std:.6f})  ← mean MUST be 0")
print(f"    Check: mean(lscale) = {np.mean(all_lscale):.6f} (should be ~0)")
# Actually the lscale values are per-step local displacement
# In the dataset, state x,y are global coordinates, and normalization divides by lscale_std
# transform2frame does (x-x_ref)/std - that's why mean must be 0
# But the actual per-step displacement mean is NOT 0 because agents move forward
# The dataset normalizes GLOBAL position, not local displacement
# So we need to check what range global positions have

print(f"\n  h: (0.0, 1.0)  ← unchanged (unit vector)")
print(f"  speed: ({np.mean(all_speed):.6f}, {np.std(all_speed):.6f})")
print(f"  hdot: ({np.mean(all_hdot):.6f}, {np.std(all_hdot):.6f})")

print(f"\n  === For BIKE_PARAMS (signed acc/ddh) ===")
print(f"  a (signed acc): ({np.mean(all_signed_acc):.6f}, {np.std(all_signed_acc):.6f})")
print(f"  ddh: ({np.mean(all_ddh):.6f}, {np.std(all_ddh):.6f})")

print(f"\n  === Vehicle attributes (fixed in CARLA) ===")
print(f"  l: ({np.mean(all_length):.6f}, 0.0001)  ← fixed 4.084m, std≈0")
print(f"  w: ({np.mean(all_width):.6f}, 0.0001)  ← fixed 1.73m, std≈0")

print("\n" + "="*70)
print("INTENT ANALYSIS with CARLA stats")
print("="*70)

carla_signed_acc_mean = np.mean(all_signed_acc)
carla_signed_acc_std = np.std(all_signed_acc)
carla_hdot_mean = np.mean(all_hdot)
carla_hdot_std = np.std(all_hdot)

intent_acc_carla = (all_signed_acc - carla_signed_acc_mean) / carla_signed_acc_std
intent_yaw_carla = (all_hdot - carla_hdot_mean) / carla_hdot_std

print(f"\n  With CARLA stats, prototypes at [-1, 0, 1]:")
print(f"  intent_acc: p1={np.percentile(intent_acc_carla, 1):.2f}, p99={np.percentile(intent_acc_carla, 99):.2f}")
print(f"  intent_yaw: p1={np.percentile(intent_yaw_carla, 1):.2f}, p99={np.percentile(intent_yaw_carla, 99):.2f}")
acc_out = np.mean(np.abs(intent_acc_carla) > 1.0) * 100
yaw_out = np.mean(np.abs(intent_yaw_carla) > 1.0) * 100
print(f"  % acc outside [-1,1]: {acc_out:.1f}%  (NUSC: 43.5%)")
print(f"  % yaw outside [-1,1]: {yaw_out:.1f}%  (NUSC: 30.5%)")

# 9-class distribution with CARLA normalization
acc_class = np.digitize(intent_acc_carla, [-1/3, 1/3])
yaw_class = np.digitize(intent_yaw_carla, [-1/3, 1/3])
intent_class = acc_class * 3 + yaw_class

labels = ['dec+left', 'dec+str', 'dec+right',
          'mnt+left', 'mnt+str', 'mnt+right',
          'acc+left', 'acc+str', 'acc+right']
print(f"\n  9-class distribution (CARLA normalized):")
for i in range(9):
    frac = np.mean(intent_class == i) * 100
    print(f"    class {i} ({labels[i]}): {frac:.1f}%")

# Also check: what does the data look like for global position range?
print("\n" + "="*70)
print("GLOBAL POSITION RANGE CHECK")
print("="*70)
all_gx = []
all_gy = []
for fidx, fpath in enumerate(scene_files[:50]):
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    all_gx.extend(data[1].tolist())
    all_gy.extend(data[2].tolist())
    all_gx.extend(data[14].tolist())
    all_gy.extend(data[15].tolist())
all_gx = np.array(all_gx)
all_gy = np.array(all_gy)
print(f"  Global X: min={np.min(all_gx):.1f}, max={np.max(all_gx):.1f}, range={np.max(all_gx)-np.min(all_gx):.1f}")
print(f"  Global Y: min={np.min(all_gy):.1f}, max={np.max(all_gy):.1f}, range={np.max(all_gy)-np.min(all_gy):.1f}")
print(f"  Note: lscale normalizes these global coords. NUSC std=15 covers ±45m range")
print(f"  For CARLA, appropriate lscale_std depends on per-scenario position spread")

# Per-scenario position spread
spreads = []
for fidx, fpath in enumerate(scene_files[:50]):
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    for agent_rows in [(1, 2), (14, 15)]:
        xs = data[agent_rows[0]]
        ys = data[agent_rows[1]]
        # In local frame (relative to first position), range
        dx = xs - xs[0]
        dy = ys - ys[0]
        spread = np.sqrt(dx**2 + dy**2).max()
        spreads.append(spread)
spreads = np.array(spreads)
print(f"\n  Per-scenario max displacement from start:")
print(f"    mean={np.mean(spreads):.1f}m, p95={np.percentile(spreads, 95):.1f}m, max={np.max(spreads):.1f}m")
print(f"    → lscale_std should cover this range in ~3 sigma")
print(f"    → Suggested lscale_std = {np.percentile(spreads, 95)/3:.1f}m")
