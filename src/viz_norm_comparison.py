"""
nuScenes 정규화 기준 분포 vs CARLA 실측 분포 비교 시각화.

nuScenes 정규화 stats (mean, std)를 기준 정규분포(N(0,1))로 표시하고,
CARLA 7798 시나리오의 실측 데이터를 같은 정규화 공간에 매핑하여
기존 분포에서 얼마나 어긋나는지 시각적으로 보여줌.

출력: viz_norm_comparison.png
"""
import sys, os
_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)
os.chdir(os.path.dirname(_src_dir))

import numpy as np
import glob
import pandas as pd
from pyquaternion import Quaternion
import datasets.nuscenes_utils as nutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# ── 정규화 상수 ──
NUSC = {
    'speed': (1.802009, 3.507907),
    'hdot':  (-0.000037, 0.055684),
    'a':     (0.409074, 1.045530),
    'ddh':   (0.000046, 0.075032),
}
CARLA = {
    'speed': (6.225233, 2.673987),
    'hdot':  (0.051012, 0.233765),
    'a':     (0.252031, 1.923014),
    'ddh':   (0.003361, 0.371728),
}

# ── 데이터 로드 ──
scenario_path = './data/race_scenarios/various_driving_data_20260224'
scene_files = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))
print(f"총 {len(scene_files)}개 시나리오 처리 중...")

SEQ_INTERVAL = 5
NPAST = 4
NFUTURE = 12
SEQ_LEN = NPAST + NFUTURE
dt_cfg = 0.5

all_speed = []
all_hdot = []
all_raw_acc = []
all_raw_ddh = []

for fidx, fpath in enumerate(scene_files):
    if fidx % 200 == 0:
        print(f"  {fidx}/{len(scene_files)}...")
    data = pd.read_excel(fpath, sheet_name='driving_data').values.tolist()
    data = np.array(data).T
    T = len(data[0])
    t_arr = np.array(data[0])

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
        speed = np.linalg.norm(vel, axis=1)
        hdot = nutils.heading_change_rate(h_arr, t_arr)

        valid_s = ~np.isnan(speed)
        all_speed.extend(speed[valid_s].tolist())
        valid_h = ~np.isnan(hdot)
        all_hdot.extend(hdot[valid_h].tolist())

        agent_speeds.append(speed)
        agent_hdots.append(hdot)

    # subsequence 기반 acc/ddh (verify_norm_final.py와 동일)
    for sidx in range(0, T - SEQ_LEN, SEQ_INTERVAL):
        midx = sidx + NPAST
        eidx = sidx + SEQ_LEN
        for a in range(2):
            sp = agent_speeds[a]
            hd = agent_hdots[a]
            if np.isnan(sp[midx-1]):
                continue
            speed_seq = sp[midx-1:eidx]
            hdot_seq = hd[midx-1:eidx]
            if np.any(np.isnan(speed_seq)) or np.any(np.isnan(hdot_seq)):
                continue
            raw_acc = (speed_seq[1:] - speed_seq[:-1]) / dt_cfg
            raw_ddh = (hdot_seq[1:] - hdot_seq[:-1]) / dt_cfg
            all_raw_acc.extend(raw_acc.tolist())
            all_raw_ddh.extend(raw_ddh.tolist())

all_speed = np.array(all_speed)
all_hdot = np.array(all_hdot)
all_raw_acc = np.array(all_raw_acc)
all_raw_ddh = np.array(all_raw_ddh)

print(f"수집 완료: speed={len(all_speed)}, hdot={len(all_hdot)}, "
      f"acc={len(all_raw_acc)}, ddh={len(all_raw_ddh)}")

# ── nuScenes 정규화 공간으로 변환 ──
# CARLA raw 데이터를 nuScenes (mean, std)로 정규화
speed_nusc_norm = (all_speed - NUSC['speed'][0]) / NUSC['speed'][1]
hdot_nusc_norm  = (all_hdot  - NUSC['hdot'][0])  / NUSC['hdot'][1]
acc_nusc_norm   = (all_raw_acc - NUSC['a'][0])    / NUSC['a'][1]
ddh_nusc_norm   = (all_raw_ddh - NUSC['ddh'][0])  / NUSC['ddh'][1]

# CARLA raw 데이터를 CARLA (mean, std)로 정규화 (올바른 분포)
speed_carla_norm = (all_speed - CARLA['speed'][0]) / CARLA['speed'][1]
hdot_carla_norm  = (all_hdot  - CARLA['hdot'][0])  / CARLA['hdot'][1]
acc_carla_norm   = (all_raw_acc - CARLA['a'][0])    / CARLA['a'][1]
ddh_carla_norm   = (all_raw_ddh - CARLA['ddh'][0])  / CARLA['ddh'][1]

# ── Raw data arrays + stats ──
raw_data = {
    'speed': (all_speed, 'm/s'),
    'hdot':  (all_hdot,  'rad/s'),
    'a':     (all_raw_acc, 'm/s$^2$'),
    'ddh':   (all_raw_ddh, 'rad/s$^2$'),
}
var_keys = ['speed', 'hdot', 'a', 'ddh']
var_titles = ['Speed', 'Heading Rate', 'Acceleration', 'Heading Accel']

def kde_curve(data, x_grid, bw_method='scott', subsample=50000):
    """Compute KDE on subsampled data for speed."""
    if len(data) > subsample:
        rng = np.random.RandomState(42)
        data = rng.choice(data, subsample, replace=False)
    kde = gaussian_kde(data, bw_method=bw_method)
    return kde(x_grid)

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ===================================================================
# Layout: 1 row top (2D scatter), 1 row bottom (4x KDE with clipped y)
# ===================================================================
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 4, height_ratios=[1.3, 1], hspace=0.30, wspace=0.30)
fig.suptitle('nuScenes vs CARLA Normalization: Distribution Mismatch\n'
             '7798 CARLA scenarios  |  443k state samples  |  541k action samples',
             fontsize=17, fontweight='bold', y=0.99)

# ── Helper: draw sigma ellipses ──
def draw_sigma_ellipses(ax, mu_x, mu_y, std_x, std_y, color, sigmas=[1, 2, 3]):
    for s in sigmas:
        e = Ellipse((mu_x, mu_y), width=2*s*std_x, height=2*s*std_y,
                    fill=False, edgecolor=color,
                    linewidth=2.5 if s == 1 else 1.5,
                    linestyle='-' if s == 1 else ('--' if s == 2 else ':'),
                    alpha=0.9 if s == 1 else 0.6)
        ax.add_patch(e)
    ax.plot(mu_x, mu_y, '+', color=color, markersize=15, markeredgewidth=2.5)

# ── Top: 2D scatter (acc vs hdot) — wide, spanning all 4 cols ──
ax_2d = fig.add_subplot(gs[0, :])

rng = np.random.RandomState(42)
n_scatter = 30000
n = min(len(all_raw_acc), len(all_hdot))
idx = rng.choice(n, min(n_scatter, n), replace=False)
ax_2d.scatter(all_raw_acc[idx], all_hdot[idx], s=2, alpha=0.12, color='#4a90d9',
              rasterized=True)
draw_sigma_ellipses(ax_2d, NUSC['a'][0], NUSC['hdot'][0],
                    NUSC['a'][1], NUSC['hdot'][1], 'crimson')
draw_sigma_ellipses(ax_2d, CARLA['a'][0], CARLA['hdot'][0],
                    CARLA['a'][1], CARLA['hdot'][1], 'forestgreen')

legend_2d = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a90d9', markersize=8,
           label='CARLA data'),
    Line2D([0], [0], color='crimson', linewidth=2.5, label='nuScenes 1$\\sigma$'),
    Line2D([0], [0], color='crimson', linewidth=1.5, linestyle='--', label='nuScenes 2$\\sigma$'),
    Line2D([0], [0], color='crimson', linewidth=1.5, linestyle=':', label='nuScenes 3$\\sigma$'),
    Line2D([0], [0], color='forestgreen', linewidth=2.5, label='CARLA 1$\\sigma$'),
    Line2D([0], [0], color='forestgreen', linewidth=1.5, linestyle='--', label='CARLA 2$\\sigma$'),
    Line2D([0], [0], color='forestgreen', linewidth=1.5, linestyle=':', label='CARLA 3$\\sigma$'),
]
ax_2d.legend(handles=legend_2d, fontsize=10, loc='upper left', framealpha=0.9, ncol=2)
ax_2d.set_xlabel('Acceleration (m/s$^2$)', fontsize=13)
ax_2d.set_ylabel('Heading Rate (rad/s)', fontsize=13)
ax_2d.set_title('Acceleration vs Heading Rate — raw physical space', fontsize=15, fontweight='bold')
xlim = np.percentile(all_raw_acc, [0.5, 99.5])
ylim = np.percentile(all_hdot, [0.5, 99.5])
ax_2d.set_xlim(xlim[0]*1.3, xlim[1]*1.3)
ax_2d.set_ylim(ylim[0]*1.3, ylim[1]*1.3)
ax_2d.grid(True, alpha=0.15)

# ── Bottom: 4x KDE (y clipped to CARLA data scale) ──
kde_data = [
    ('Speed (m/s)',           all_speed,    NUSC['speed'], CARLA['speed']),
    ('Heading Rate (rad/s)',  all_hdot,     NUSC['hdot'],  CARLA['hdot']),
    ('Acceleration (m/s$^2$)', all_raw_acc, NUSC['a'],     CARLA['a']),
    ('Heading Accel (rad/s$^2$)', all_raw_ddh, NUSC['ddh'], CARLA['ddh']),
]

for i, (title, arr, n_stats, c_stats) in enumerate(kde_data):
    ax = fig.add_subplot(gs[1, i])
    n_mu, n_std = n_stats
    c_mu, c_std = c_stats

    p01, p99 = np.percentile(arr, [0.5, 99.5])
    pad = (p99 - p01) * 0.25
    lo, hi = p01 - pad, p99 + pad
    x = np.linspace(lo, hi, 500)

    # KDE of CARLA data
    y_kde = kde_curve(arr, x)
    ax.fill_between(x, y_kde, alpha=0.3, color='#4a90d9')
    ax.plot(x, y_kde, color='#2060b0', linewidth=2.5,
            label='CARLA data')

    # nuScenes Gaussian
    y_nusc = norm.pdf(x, n_mu, n_std)
    ax.plot(x, y_nusc, color='crimson', linewidth=2.5,
            label=f'nuScenes ($\\sigma$={n_std:.4f})')
    ax.fill_between(x, y_nusc, alpha=0.08, color='crimson')

    # CARLA Gaussian
    y_carla = norm.pdf(x, c_mu, c_std)
    ax.plot(x, y_carla, color='forestgreen', linewidth=2, linestyle='--',
            label=f'CARLA ($\\sigma$={c_std:.4f})')

    # y-axis: clip to CARLA data scale
    data_ymax = max(np.max(y_kde), np.max(y_carla))
    nusc_peak = np.max(y_nusc)
    ylim_top = data_ymax * 1.4
    ax.set_ylim(0, ylim_top)

    # If nuScenes clipped, show true peak
    if nusc_peak > ylim_top:
        peak_x = x[np.argmax(y_nusc)]
        ax.annotate(f'nuScenes peak={nusc_peak:.1f}\n({nusc_peak/data_ymax:.0f}x taller)',
                    xy=(peak_x, ylim_top * 0.97), fontsize=9,
                    color='crimson', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='crimson', alpha=0.9))

    # % outside nuScenes ±3σ
    outside = np.mean((arr < n_mu - 3*n_std) | (arr > n_mu + 3*n_std)) * 100
    if outside > 1:
        ax.text(0.97, 0.70, f'{outside:.0f}% outside\nnuScenes $\\pm$3$\\sigma$',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color='crimson', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='crimson', alpha=0.9))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Value', fontsize=10)
    if i == 0:
        ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right' if i != 0 else 'upper left', framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0.01, 1, 0.96])

out_path = './viz_norm_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close()
