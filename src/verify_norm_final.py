"""
최종 검증: NUSC_NORM_STATS가 CARLA 데이터에 적용될 때의 정확한 영향 분석.

데이터 흐름:
1. post_process: raw state (x_global, y_global, hcos, hsin, speed, hdot) 생성
2. __getitem__: normalizer.normalize(state) → 정규화된 state
3. model: 정규화된 state를 그대로 MLP, GCN, Transformer에 입력
4. decoder: bicycle model output(2dim: acc, ddh) → unnormalize → dynamics
5. loss: normalizer.unnormalize로 position 복원 (map attn soft label 등)

정규화가 영향 미치는 곳:
A. State normalizer: x,y,hcos,hsin,speed,hdot (6dim)
   - Dataset에서 normalize
   - Model encoder (MLP/GCN)에 직접 입력
   - Loss에서 unnormalize하여 GT position 복원
B. Bicycle params: a_stats(acc mean/std), ddh_stats(ddh mean/std)
   - Decoder output unnormalize: raw_acc = pred*std + mean
   - GT action 계산: norm_acc = (raw_acc - mean) / std
C. Intent CE loss: NUSC_NORM_STATS에서 a_std, hdot_std 직접 참조
   - acc / a_std, yaw_rate / hdot_std → prototype distance
D. Att normalizer: l,w (vehicle attributes)
   - Dataset에서 normalize, model에서 unnormalize
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import glob
import pandas as pd
from pyquaternion import Quaternion
import datasets.nuscenes_utils as nutils
from datasets.utils import NUSC_NORM_STATS

# === 전체 500개 데이터에서 통계 계산 ===
# dataset post_process와 100% 동일한 방식으로 계산

scenario_path = './data/race_scenarios/various_300_sample'
scene_files = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))
print(f"Processing {len(scene_files)} files...")

# State 6dim 각 차원의 raw 값 수집
all_x_global = []      # dim 0
all_y_global = []      # dim 1
all_hcos = []          # dim 2
all_hsin = []          # dim 3
all_speed = []         # dim 4
all_hdot = []          # dim 5

# Bicycle model 관련
all_signed_acc = []    # (speed[t+1] - speed[t]) / dt — bicycle output에 대응
all_ddh = []           # heading_change_rate(hdot, t) — dataset의 ddh

# Vehicle attributes
all_length = []
all_width = []

dt = 0.5

for fidx, fpath in enumerate(scene_files):
    if fidx % 50 == 0:
        print(f"  {fidx}/{len(scene_files)}...")
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    t_arr = np.array(data[0])

    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            pos_rows, quat_rows = data[1:4], data[4:8]
        else:
            pos_rows, quat_rows = data[14:17], data[17:21]

        x_arr = pos_rows[0].copy()
        y_arr = pos_rows[1].copy()

        h_arr = np.zeros(T)
        hcos_arr = np.zeros(T)
        hsin_arr = np.zeros(T)
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])
            hcos_arr[i] = np.cos(h_arr[i])
            hsin_arr[i] = np.sin(h_arr[i])

        # Speed: dataset과 동일 = norm(velocity(pos, t))
        pos_2d = np.stack([x_arr, y_arr], axis=1)
        vel = nutils.velocity(pos_2d, t_arr)
        speed = np.linalg.norm(vel, axis=1)

        # hdot: dataset과 동일 = heading_change_rate(h, t)
        hdot = nutils.heading_change_rate(h_arr, t_arr)

        # ddh: dataset과 동일 = heading_change_rate(hdot, t)
        ddh = nutils.heading_change_rate(hdot, t_arr)

        # Signed acc: bicycle model이 출력하는 acc에 대응
        # model._compute_gt_actions에서:
        #   raw_acc = (speed[t+1] - speed[t]) / dt
        #   norm_acc = (raw_acc - a_mean) / a_std
        # 여기서 speed는 unnormalized (raw m/s)
        signed_acc = np.full(T, np.nan)
        for i in range(T - 1):
            if not (np.isnan(speed[i]) or np.isnan(speed[i+1])):
                signed_acc[i] = (speed[i+1] - speed[i]) / (t_arr[i+1] - t_arr[i])
        if T > 1 and not np.isnan(signed_acc[T-2]):
            signed_acc[T-1] = signed_acc[T-2]

        # Collect valid values
        valid = ~np.isnan(speed)
        all_x_global.extend(x_arr[valid].tolist())
        all_y_global.extend(y_arr[valid].tolist())
        all_hcos.extend(hcos_arr[valid].tolist())
        all_hsin.extend(hsin_arr[valid].tolist())
        all_speed.extend(speed[valid].tolist())
        all_hdot.extend(hdot[valid].tolist())

        valid_ddh = ~np.isnan(ddh)
        all_ddh.extend(ddh[valid_ddh].tolist())

        valid_acc = ~np.isnan(signed_acc)
        all_signed_acc.extend(signed_acc[valid_acc].tolist())

        # Vehicle attributes (hardcoded in dataset: l=4.084, w=1.73)
        all_length.append(4.084)
        all_width.append(1.73)

# Convert to numpy
all_x = np.array(all_x_global)
all_y = np.array(all_y_global)
all_hcos = np.array(all_hcos)
all_hsin = np.array(all_hsin)
all_speed = np.array(all_speed)
all_hdot = np.array(all_hdot)
all_signed_acc = np.array(all_signed_acc)
all_ddh = np.array(all_ddh)

def pr(name, arr, ideal_range=None):
    m, s = np.mean(arr), np.std(arr)
    print(f"  {name}:")
    print(f"    mean={m:.6f}, std={s:.6f}")
    print(f"    min={np.min(arr):.4f}, max={np.max(arr):.4f}")
    print(f"    p1={np.percentile(arr, 1):.4f}, p99={np.percentile(arr, 99):.4f}")
    if ideal_range:
        out = np.mean((arr < ideal_range[0]) | (arr > ideal_range[1])) * 100
        print(f"    % outside [{ideal_range[0]}, {ideal_range[1]}]: {out:.1f}%")
    return m, s

print(f"\n{'='*70}")
print(f"A. STATE 6dim RAW STATISTICS ({len(all_speed)} samples)")
print(f"{'='*70}")
x_m, x_s = pr("dim0: x_global (m)", all_x)
y_m, y_s = pr("dim1: y_global (m)", all_y)
hc_m, hc_s = pr("dim2: hcos", all_hcos)
hs_m, hs_s = pr("dim3: hsin", all_hsin)
sp_m, sp_s = pr("dim4: speed (m/s)", all_speed)
hd_m, hd_s = pr("dim5: hdot (rad/s)", all_hdot)

print(f"\n{'='*70}")
print(f"B. BICYCLE PARAMS RAW STATISTICS ({len(all_signed_acc)} samples)")
print(f"{'='*70}")
acc_m, acc_s = pr("signed_acc (m/s²)", all_signed_acc)
ddh_m, ddh_s = pr("ddh (rad/s²)", all_ddh)

print(f"\n{'='*70}")
print(f"CURRENT: NUSC_NORM_STATS 적용 후")
print(f"{'='*70}")
ninfo = NUSC_NORM_STATS[('car', 'truck')]
# State normalizer: (x - mean) / std
nusc_state_mean = [0.0, 0.0, 0.0, 0.0, ninfo['s'][0], ninfo['hdot'][0]]
nusc_state_std = [ninfo['lscale'][1], ninfo['lscale'][1], 1.0, 1.0, ninfo['s'][1], ninfo['hdot'][1]]

dims = ['x', 'y', 'hcos', 'hsin', 'speed', 'hdot']
raw_arrays = [all_x, all_y, all_hcos, all_hsin, all_speed, all_hdot]

for i, (name, arr) in enumerate(zip(dims, raw_arrays)):
    norm_arr = (arr - nusc_state_mean[i]) / nusc_state_std[i]
    pr(f"norm_{name} (NUSC)", norm_arr, ideal_range=(-3, 3))

print(f"\n  Bicycle params after NUSC:")
nusc_a_mean, nusc_a_std = ninfo['a']
nusc_ddh_mean, nusc_ddh_std = ninfo['ddh']
norm_acc = (all_signed_acc - nusc_a_mean) / nusc_a_std
norm_ddh = (all_ddh - nusc_ddh_mean) / nusc_ddh_std
pr(f"norm_acc (NUSC)", norm_acc, ideal_range=(-3, 3))
pr(f"norm_ddh (NUSC)", norm_ddh, ideal_range=(-3, 3))

print(f"\n{'='*70}")
print(f"PROPOSED CARLA STATISTICS")
print(f"{'='*70}")

# lscale:
# x, y는 global 좌표이고 dataset에서 정규화 후 model에 들어감
# lscale mean MUST = 0 (transform2frame에서 뺄셈: (x_A-x_B)/std 가 정확하려면)
# lscale std: global position의 std가 아니라,
#   "model이 보는 좌표 범위를 [-3, 3] 정도에 맞추는 스케일"
# 현재 NUSC std=15: global x는 [-460, 712] 범위 → norm [-30, 47] → 너무 큼
# 하지만! GCN에서 pos는 agent간 상대적 관계에만 사용
# transform2frame에서 (x_A - x_B) / std → 상대 거리 / std
# 따라서 lscale_std는 "inter-agent distance scale"에 맞춰야 함
# 즉, 한 시나리오 내에서 agent 간 거리의 적정 스케일

# Per-scenario agent distance
print(f"\n  lscale 분석:")
print(f"    Global position은 절대 좌표이므로 mean/std가 의미 없음")
print(f"    중요한 건 agent 간 상대 거리를 정규화하는 스케일")
print(f"    현재 NUSC lscale_std=15: 15m = 1 단위")
print(f"    CARLA 시나리오에서 agent 간 거리:")

# Calculate inter-agent distances in scenarios
agent_dists = []
for fidx, fpath in enumerate(scene_files[:50]):
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    for t in range(T):
        dx = data[1][t] - data[14][t]  # ego_x - sur_x
        dy = data[2][t] - data[15][t]  # ego_y - sur_y
        agent_dists.append(np.sqrt(dx**2 + dy**2))
agent_dists = np.array(agent_dists)
print(f"    mean={np.mean(agent_dists):.1f}m, std={np.std(agent_dists):.1f}m")
print(f"    p5={np.percentile(agent_dists, 5):.1f}m, p95={np.percentile(agent_dists, 95):.1f}m")
print(f"    현재 정규화(÷15): 거리 15m → 1.0, 거리 50m → 3.3")

# lscale도 본 map crop 범위와 관련
print(f"    Map crop bounds: [-17, -38.5, 60, 38.5] → 77m × 77m")
print(f"    Max meaningful distance ≈ 60m (forward)")

# Final proposal
print(f"\n{'='*70}")
print(f"FINAL PROPOSED CARLA NORM STATS")
print(f"{'='*70}")

# lscale: mean=0 필수. std는 현재 15로도 agent 간 거리/map 범위에 대해 괜찮음
# 다만, 12 step × 0.5s = 6초간 이동거리 최대 ~55m. std=15면 norm=3.7 → 허용 범위 내
# 너무 바꾸면 map crop coordinate와 불일치 위험. 15 유지가 안전
lscale_std_proposed = 15.0  # 유지
print(f"  lscale: (0.0, {lscale_std_proposed})  ← 유지 (agent간 거리/map crop에 적합)")

print(f"  h: (0.0, 1.0)  ← 유지 (unit vector)")

print(f"  s (speed): ({sp_m:.6f}, {sp_s:.6f})")
print(f"    현재 NUSC: (1.802, 3.508)")
norm_speed_carla = (all_speed - sp_m) / sp_s
pr("    → norm_speed (CARLA)", norm_speed_carla, ideal_range=(-3, 3))

print(f"  hdot: ({hd_m:.6f}, {hd_s:.6f})")
print(f"    현재 NUSC: (-0.000037, 0.055684)")
norm_hdot_carla = (all_hdot - hd_m) / hd_s
pr("    → norm_hdot (CARLA)", norm_hdot_carla, ideal_range=(-3, 3))

print(f"\n  BIKE_PARAMS:")
print(f"  a_stats (signed acc): ({acc_m:.6f}, {acc_s:.6f})")
print(f"    현재 NUSC: (0.409074, 1.045530)")
norm_acc_carla = (all_signed_acc - acc_m) / acc_s
pr("    → norm_acc (CARLA)", norm_acc_carla, ideal_range=(-3, 3))

print(f"  ddh_stats: ({ddh_m:.6f}, {ddh_s:.6f})")
print(f"    현재 NUSC: (0.000046, 0.075032)")
norm_ddh_carla = (all_ddh - ddh_m) / ddh_s
pr("    → norm_ddh (CARLA)", norm_ddh_carla, ideal_range=(-3, 3))

# Vehicle attributes
print(f"\n  l: (4.084, 0.001)  ← 고정 차량, std≈0 → 작은 값 사용")
print(f"  w: (1.73, 0.001)  ← 고정 차량, std≈0 → 작은 값 사용")
print(f"  (주의: std=0이면 normalize시 division by zero → 작은 양수 사용)")

print(f"\n{'='*70}")
print(f"INTENT 분석: CARLA stats 적용 시")
print(f"{'='*70}")
# Intent CE loss에서 prototype은 [-1, 0, 1] × [-1, 0, 1]
# acc = raw_acc / a_std (mean은 안 뺌 — 코드 확인 필요)

# 코드 재확인: loss.py line 590
# ego_acc = raw_acc / a_std  ← mean을 빼지 않음!
# raw_acc = (speed[t+1] - speed[t]) / dt
# 이건 signed acc를 a_std로 나눈 것

print(f"\n  Intent CE loss 코드:")
print(f"    raw_acc = (speed_next - speed_cur) / dt")
print(f"    ego_acc = raw_acc / a_std")
print(f"    ego_yaw_rate = raw_hdot / hdot_std")
print(f"    → prototype [-1,0,1] 과 비교")

print(f"\n  현재 NUSC (a_std={nusc_a_std:.4f}, hdot_std={ninfo['hdot'][1]:.6f}):")
intent_acc_nusc = all_signed_acc / nusc_a_std
intent_yaw_nusc = all_hdot / ninfo['hdot'][1]
pr("    intent_acc (NUSC)", intent_acc_nusc, ideal_range=(-1.5, 1.5))
pr("    intent_yaw (NUSC)", intent_yaw_nusc, ideal_range=(-1.5, 1.5))

print(f"\n  제안 CARLA (a_std={acc_s:.4f}, hdot_std={hd_s:.6f}):")
intent_acc_carla = all_signed_acc / acc_s
intent_yaw_carla = all_hdot / hd_s
pr("    intent_acc (CARLA)", intent_acc_carla, ideal_range=(-1.5, 1.5))
pr("    intent_yaw (CARLA)", intent_yaw_carla, ideal_range=(-1.5, 1.5))

# 9-class distribution comparison
print(f"\n  9-class distribution:")
labels = ['dec+L', 'dec+S', 'dec+R', 'mnt+L', 'mnt+S', 'mnt+R', 'acc+L', 'acc+S', 'acc+R']
for tag, ia, iy in [("NUSC", intent_acc_nusc, intent_yaw_nusc),
                     ("CARLA", intent_acc_carla, intent_yaw_carla)]:
    ac = np.digitize(ia, [-1/3, 1/3])
    yc = np.digitize(iy, [-1/3, 1/3])
    ic = ac * 3 + yc
    dist = [f"{labels[i]}={np.mean(ic==i)*100:.0f}%" for i in range(9)]
    print(f"    {tag}: {', '.join(dist)}")
