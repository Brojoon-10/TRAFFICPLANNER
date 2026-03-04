"""
CARLA 데이터 정규화 통계 검증 스크립트.

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
   ★ 중요: a_stats/ddh_stats는 모델이 실제로 보는 값의 분포와 일치해야 함.
     모델은 _compute_gt_actions()에서 학습 subsequence의 future 구간만 사용:
       raw_acc = (speed[t+1] - speed[t]) / dt   (dt=0.5 고정)
     따라서 전체 trajectory 연속 프레임(가변 dt)이 아닌,
     FITDataset 학습 subsequence 기반으로 통계를 구해야 함.
C. Intent CE loss: CARLA_NORM_STATS에서 a_std, hdot_std 직접 참조
D. Att normalizer: l,w (vehicle attributes)
"""
import sys, os
# src/ 디렉토리를 path에 추가 (maps, datasets 등 import 위해)
_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)
# 프로젝트 루트로 이동 (상대 경로 data/ 접근 위해)
os.chdir(os.path.dirname(_src_dir))

import numpy as np
import glob
import pandas as pd
import torch
from pyquaternion import Quaternion
import datasets.nuscenes_utils as nutils
from datasets.utils import CARLA_NORM_STATS, CARLA_BIKE_PARAMS

# ============================================================
# Part A: State 6dim 통계 (전체 trajectory 기반 — state normalizer용)
# ============================================================
# State normalizer는 전체 데이터의 speed/hdot 분포를 사용.
# FITDataset.post_process()에서 계산되는 raw state 값의 통계.

scenario_path = './data/race_scenarios/various_driving_data_20260224'
scene_files = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))
print(f"Processing {len(scene_files)} files for state statistics...")

all_speed = []
all_hdot = []
dt_cfg = 0.5

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
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])

        pos_2d = np.stack([x_arr, y_arr], axis=1)
        vel = nutils.velocity(pos_2d, t_arr)
        speed = np.linalg.norm(vel, axis=1)
        hdot = nutils.heading_change_rate(h_arr, t_arr)

        valid = ~np.isnan(speed)
        all_speed.extend(speed[valid].tolist())
        valid_h = ~np.isnan(hdot)
        all_hdot.extend(hdot[valid_h].tolist())

all_speed = np.array(all_speed)
all_hdot = np.array(all_hdot)

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
print(f"A. STATE STATISTICS (전체 trajectory, {len(all_speed)} samples)")
print(f"{'='*70}")
sp_m, sp_s = pr("speed (m/s)", all_speed)
hd_m, hd_s = pr("hdot (rad/s)", all_hdot)

# ============================================================
# Part B: Bicycle params (a_stats, ddh_stats) — FITDataset 기반
# ============================================================
# ★ 핵심: _compute_gt_actions()와 동일한 방식으로 계산
#
# _compute_gt_actions()의 계산:
#   speed_seq = [last_past_speed, future_speed_0, ..., future_speed_11]  (13개)
#   raw_acc = (speed_seq[:, 1:] - speed_seq[:, :-1]) / dt              (12개, dt=0.5 고정)
#   hdot_seq도 동일
#
# 왜 전체 trajectory가 아닌 학습 subsequence인가:
#   1. 모델은 FITDataset.__getitem__()이 잘라주는 subsequence만 봄
#      (seq_interval=5로 샘플링, past_len=4 + future_len=12)
#   2. _compute_gt_actions()는 이 subsequence의 future 구간에서만 acc/ddh 계산
#   3. dt는 config의 0.5초 고정 (시뮬레이터의 가변 프레임 간격이 아님)
#   4. 전체 trajectory 기반 통계는 정차/출발 구간, 가변 dt 등으로 분포가 다름
#
# 따라서 a_stats/ddh_stats는 모델이 실제로 보는 분포 = FITDataset subsequence 기반이어야
# 정규화 후 mean=0, std=1이 보장됨.

print(f"\n{'='*70}")
print(f"B. BICYCLE PARAMS — FITDataset subsequence 재현 (_compute_gt_actions 동일)")
print(f"{'='*70}")

# FITDataset을 직접 로드하지 않고, 동일한 로직을 재현:
#   1. xlsx → raw position/heading 추출
#   2. nutils.velocity()로 speed/hdot 계산 (가변 dt, FITDataset.post_process와 동일)
#   3. seq_interval=5로 subsequence 시작점 생성
#   4. past[-1]과 future[0:12]로 speed_seq 구성
#   5. raw_acc = (speed[t+1] - speed[t]) / dt_cfg  (dt=0.5 고정)
#
# 이것이 _compute_gt_actions()와 동일한 이유:
#   - _compute_gt_actions()는 normalized speed를 unnormalize한 뒤 차분/dt로 acc 계산
#   - unnormalize(normalize(x)) = x 이므로 raw speed 차분/dt와 동일
#   - FITDataset.post_process()의 speed = nutils.velocity(pos, t_arr) 그대로 사용

SEQ_INTERVAL = 5
NPAST = 4
NFUTURE = 12
SEQ_LEN = NPAST + NFUTURE
SPLIT_RATIO = 0.85  # train split

all_raw_acc = []
all_raw_ddh = []
total_subseq = 0

# train split만 사용 (FITDataset과 동일)
train_n = int(len(scene_files) * SPLIT_RATIO)
train_files = scene_files[:train_n]
print(f"  Train split: {train_n}/{len(scene_files)} files")

for fidx, fpath in enumerate(train_files):
    if fidx % 100 == 0:
        print(f"  Part B: {fidx}/{len(train_files)}...")
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    t_arr = np.array(data[0])

    # 각 agent에 대해 speed/hdot 계산 (FITDataset.post_process 동일)
    agent_speeds = []
    agent_hdots = []
    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            pos_rows, quat_rows = data[1:4], data[4:8]
        else:
            pos_rows, quat_rows = data[14:17], data[17:21]

        x_arr = pos_rows[0].copy()
        y_arr = pos_rows[1].copy()

        h_arr = np.zeros(T)
        for i in range(T):
            rot = Quaternion(quat_rows[3][i], quat_rows[0][i],
                           quat_rows[1][i], quat_rows[2][i]).rotation_matrix
            h_arr[i] = np.arctan2(rot[1, 0], rot[0, 0])

        pos_2d = np.stack([x_arr, y_arr], axis=1)
        vel = nutils.velocity(pos_2d, t_arr)
        speed = np.linalg.norm(vel, axis=1)  # (T,)
        hdot = nutils.heading_change_rate(h_arr, t_arr)  # (T,)

        agent_speeds.append(speed)
        agent_hdots.append(hdot)

    # subsequence 생성 (FITDataset.__getitem__ 동일)
    for sidx in range(0, T - SEQ_LEN, SEQ_INTERVAL):
        midx = sidx + NPAST
        eidx = sidx + SEQ_LEN

        for a in range(2):  # ego, sur
            speed = agent_speeds[a]
            hdot_a = agent_hdots[a]

            # past[-1]이 NaN이면 skip (FITDataset과 동일: midx-1 시점이 유효해야 함)
            if np.isnan(speed[midx-1]):
                continue

            # speed_seq: [past[-1], future[0], ..., future[11]]  (13개)
            speed_seq = speed[midx-1:eidx]  # midx-1 ~ eidx-1, 즉 13개
            hdot_seq = hdot_a[midx-1:eidx]

            # NaN이 있으면 skip
            if np.any(np.isnan(speed_seq)) or np.any(np.isnan(hdot_seq)):
                continue

            # raw_acc = (speed[t+1] - speed[t]) / dt  (12개, dt=0.5 고정)
            raw_acc = (speed_seq[1:] - speed_seq[:-1]) / dt_cfg
            raw_ddh = (hdot_seq[1:] - hdot_seq[:-1]) / dt_cfg

            all_raw_acc.extend(raw_acc.tolist())
            all_raw_ddh.extend(raw_ddh.tolist())
            total_subseq += 1

all_raw_acc = np.array(all_raw_acc)
all_raw_ddh = np.array(all_raw_ddh)

print(f"  Total acc/ddh samples: {len(all_raw_acc)} ({total_subseq} subseqs × {NFUTURE} future steps)")
acc_m_fit, acc_s_fit = pr("raw_acc (m/s², FITDataset)", all_raw_acc)
ddh_m_fit, ddh_s_fit = pr("raw_ddh (rad/s², FITDataset)", all_raw_ddh)

# 정규화 후 mean=0, std=1 검증
norm_acc_fit = (all_raw_acc - acc_m_fit) / acc_s_fit
norm_ddh_fit = (all_raw_ddh - ddh_m_fit) / ddh_s_fit
print(f"\n  정규화 후 검증 (이 값으로 a_stats/ddh_stats 설정 시):")
pr("    norm_acc", norm_acc_fit, ideal_range=(-3, 3))
pr("    norm_ddh", norm_ddh_fit, ideal_range=(-3, 3))

# ============================================================
# Part C: 현재 config와 비교
# ============================================================

print(f"\n{'='*70}")
print(f"C. 현재 CARLA_BIKE_PARAMS vs FITDataset 실측")
print(f"{'='*70}")

cur_a = CARLA_BIKE_PARAMS['a_stats']
cur_ddh = CARLA_BIKE_PARAMS['ddh_stats']

print(f"  {'항목':<12} {'현재 config':>14} {'FITDataset 실측':>16} {'차이':>10} {'비율':>8}")
print(f"  {'─'*62}")
print(f"  {'acc mean':<12} {cur_a[0]:>14.6f} {acc_m_fit:>16.6f} {acc_m_fit-cur_a[0]:>+10.6f} {abs(acc_m_fit-cur_a[0])/max(abs(cur_a[0]),1e-9)*100:>7.1f}%")
print(f"  {'acc std':<12} {cur_a[1]:>14.6f} {acc_s_fit:>16.6f} {acc_s_fit-cur_a[1]:>+10.6f} {abs(acc_s_fit-cur_a[1])/cur_a[1]*100:>7.1f}%")
print(f"  {'ddh mean':<12} {cur_ddh[0]:>14.6f} {ddh_m_fit:>16.6f} {ddh_m_fit-cur_ddh[0]:>+10.6f} {abs(ddh_m_fit-cur_ddh[0])/max(abs(cur_ddh[0]),1e-9)*100:>7.1f}%")
print(f"  {'ddh std':<12} {cur_ddh[1]:>14.6f} {ddh_s_fit:>16.6f} {ddh_s_fit-cur_ddh[1]:>+10.6f} {abs(ddh_s_fit-cur_ddh[1])/cur_ddh[1]*100:>7.1f}%")

# 현재 config로 정규화하면 어떤 분포가 되는지
norm_acc_cur = (all_raw_acc - cur_a[0]) / cur_a[1]
norm_ddh_cur = (all_raw_ddh - cur_ddh[0]) / cur_ddh[1]
print(f"\n  현재 config로 정규화 시 (이상적: mean=0, std=1):")
pr("    norm_acc (현재 config)", norm_acc_cur, ideal_range=(-3, 3))
pr("    norm_ddh (현재 config)", norm_ddh_cur, ideal_range=(-3, 3))

# ============================================================
# Part D: State normalizer 검증
# ============================================================

print(f"\n{'='*70}")
print(f"D. STATE NORMALIZER 검증")
print(f"{'='*70}")

cinfo = CARLA_NORM_STATS[('car', 'truck')]
print(f"  speed: config=({cinfo['s'][0]:.6f}, {cinfo['s'][1]:.6f}), 실측=({sp_m:.6f}, {sp_s:.6f})")
print(f"    차이: mean {abs(sp_m-cinfo['s'][0]):.6f}, std {abs(sp_s-cinfo['s'][1]):.6f}")
print(f"  hdot:  config=({cinfo['hdot'][0]:.6f}, {cinfo['hdot'][1]:.6f}), 실측=({hd_m:.6f}, {hd_s:.6f})")
print(f"    차이: mean {abs(hd_m-cinfo['hdot'][0]):.6f}, std {abs(hd_s-cinfo['hdot'][1]):.6f}")

# ============================================================
# Part E: lscale 분석 (agent간 거리, local 변위)
# ============================================================

print(f"\n{'='*70}")
print(f"E. lscale 분석")
print(f"{'='*70}")

# Agent간 거리
agent_dists = []
for fidx, fpath in enumerate(scene_files[:50]):
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    for t in range(T):
        dx = data[1][t] - data[14][t]
        dy = data[2][t] - data[15][t]
        agent_dists.append(np.sqrt(dx**2 + dy**2))
agent_dists = np.array(agent_dists)
print(f"  Agent간 거리 (50 scenarios):")
print(f"    mean={np.mean(agent_dists):.1f}m, std={np.std(agent_dists):.1f}m")
print(f"    p5={np.percentile(agent_dists, 5):.1f}m, p95={np.percentile(agent_dists, 95):.1f}m")

# Raw trajectory에서 local 변위 (past[-1] 기준) — subsequence 기반
print(f"\n  Local 변위 (past[-1] 기준, subsequence에서 모델이 보는 값):")
lscale_std = cinfo['lscale'][1]
all_local_past = []
all_local_future = []
for fidx, fpath in enumerate(train_files[:200]):
    data_e = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data_e = np.array(data_e).T
    T_e = len(data_e[0])
    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            x_e, y_e = data_e[1], data_e[2]
        else:
            x_e, y_e = data_e[14], data_e[15]
        for sidx in range(0, T_e - SEQ_LEN, SEQ_INTERVAL):
            midx_e = sidx + NPAST
            eidx_e = sidx + SEQ_LEN
            ref_x = x_e[midx_e - 1]
            ref_y = y_e[midx_e - 1]
            for t in range(sidx, midx_e):
                d = np.sqrt((x_e[t]-ref_x)**2 + (y_e[t]-ref_y)**2)
                all_local_past.append(d)
            for t in range(midx_e, eidx_e):
                d = np.sqrt((x_e[t]-ref_x)**2 + (y_e[t]-ref_y)**2)
                all_local_future.append(d)

all_local_past = np.array(all_local_past)
all_local_future = np.array(all_local_future)
print(f"    Past local dist: mean={np.mean(all_local_past):.2f}m, p95={np.percentile(all_local_past, 95):.2f}m")
print(f"    Future local dist: mean={np.mean(all_local_future):.2f}m, p95={np.percentile(all_local_future, 95):.2f}m")
print(f"    Normalized (÷{lscale_std}): past p95={np.percentile(all_local_past, 95)/lscale_std:.2f}, future p95={np.percentile(all_local_future, 95)/lscale_std:.2f}")

# ============================================================
# Part F: 최종 제안값 출력
# ============================================================

print(f"\n{'='*70}")
print(f"F. CARLA_NORM_STATS / CARLA_BIKE_PARAMS 최종 값")
print(f"{'='*70}")
print(f"""
CARLA_BIKE_PARAMS = {{
    'maxs' : BIKE_MAXS,
    'maxhdot' : BIKE_MAXHDOT,
    'dt' : 0.5,
    'a_stats' : ({acc_m_fit:.6f}, {acc_s_fit:.6f}),
    'ddh_stats' : ({ddh_m_fit:.6f}, {ddh_s_fit:.6f})
}}

CARLA_NORM_STATS = {{
    ('car', 'truck') : {{
        'l' : (4.9017, 0.001),
        'w' : (2.1283, 0.001),
        's' : ({sp_m:.6f}, {sp_s:.6f}),
        'h' : (0.0, 1.0),
        'hdot' : ({hd_m:.6f}, {hd_s:.6f}),
        'lscale' : (0.0, 15.0),
        'a' : ({acc_m_fit:.6f}, {acc_s_fit:.6f}),
        'ddh' : ({ddh_m_fit:.6f}, {ddh_s_fit:.6f})
    }}
}}
""")

# ============================================================
# Part G: Intent 분석
# ============================================================

print(f"{'='*70}")
print(f"G. INTENT 분석 (CARLA stats 적용 시)")
print(f"{'='*70}")

# Intent는 raw_acc / a_std, raw_hdot / hdot_std 사용
# Part B에서 수집한 raw_acc와 별도로 future hdot을 수집
all_future_hdot = []
for fidx, fpath in enumerate(train_files):
    data_g = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data_g = np.array(data_g).T
    T_g = len(data_g[0])
    t_arr_g = np.array(data_g[0])
    for agent_type in ['ego', 'sur']:
        if agent_type == 'ego':
            pos_g, quat_g = data_g[1:4], data_g[4:8]
        else:
            pos_g, quat_g = data_g[14:17], data_g[17:21]
        h_arr_g = np.zeros(T_g)
        for i in range(T_g):
            rot_g = Quaternion(quat_g[3][i], quat_g[0][i], quat_g[1][i], quat_g[2][i]).rotation_matrix
            h_arr_g[i] = np.arctan2(rot_g[1, 0], rot_g[0, 0])
        hdot_g = nutils.heading_change_rate(h_arr_g, t_arr_g)
        for sidx in range(0, T_g - SEQ_LEN, SEQ_INTERVAL):
            midx_g = sidx + NPAST
            eidx_g = sidx + SEQ_LEN
            if np.isnan(hdot_g[midx_g-1]):
                continue
            fut_hdot = hdot_g[midx_g:eidx_g]
            if np.any(np.isnan(fut_hdot)):
                continue
            all_future_hdot.extend(fut_hdot.tolist())
all_future_hdot = np.array(all_future_hdot)

intent_acc = all_raw_acc / acc_s_fit
intent_yaw = all_future_hdot / hd_s

print(f"\n  Intent 분포 (acc/a_std, hdot/hdot_std):")
pr("  intent_acc", intent_acc, ideal_range=(-2, 2))
pr("  intent_yaw", intent_yaw, ideal_range=(-2, 2))

# 9-class distribution
labels = ['dec+L', 'dec+S', 'dec+R', 'mnt+L', 'mnt+S', 'mnt+R', 'acc+L', 'acc+S', 'acc+R']
n = min(len(intent_acc), len(intent_yaw))
ac = np.digitize(intent_acc[:n], [-1/3, 1/3])
yc = np.digitize(intent_yaw[:n], [-1/3, 1/3])
ic = ac * 3 + yc
dist = [f"{labels[i]}={np.mean(ic==i)*100:.0f}%" for i in range(9)]
print(f"\n  9-class: {', '.join(dist)}")
