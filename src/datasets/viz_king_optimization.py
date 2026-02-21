#!/usr/bin/env python3
"""
Visualize KING optimization results.
Uses global action fitting (avg error < 0.01m) and KING's BicycleModel simulation.
"""
import sys
sys.path.insert(0, '/home/hj/RACE_STRIVE/src')

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches

from king.proxy_simulator.motion_model import BicycleModel
from king.proxy_simulator.bm_policy import BMActionSequence
from king_augmentation import (
    KINGConfig, ExcelDataLoader, FixedEgoSimulator,
    compute_actions_from_trajectory_global
)
from datasets.fit_map_env import FITMapEnv


def draw_vehicle(ax, x, y, yaw, color, alpha=1.0, label=None):
    """Draw a vehicle rectangle at given position and orientation."""
    length = 4.4  # Full vehicle length (2.2 * 2)
    width = 1.8   # Full vehicle width (0.9 * 2)

    # Create rectangle centered at origin
    rect = FancyBboxPatch(
        (-length/2, -width/2), length, width,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        facecolor=color, edgecolor='black', alpha=alpha,
        linewidth=0.5
    )

    # Create transform: rotate then translate
    transform = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)

    return rect


def visualize_king_result(ego_traj, sur_traj_original, sur_traj_optimized,
                          timestamps, output_path=None, title="KING Optimization Result"):
    """
    Visualize original vs optimized surrounding vehicle trajectories.

    Args:
        ego_traj: dict with 'pos' (T, 2), 'yaw' (T, 1)
        sur_traj_original: dict with 'pos' (N, T, 2), 'yaw' (N, T, 1)
        sur_traj_optimized: dict with 'pos' (N, T, 2), 'yaw' (N, T, 1)
        timestamps: (T,)
        output_path: Path to save figure
        title: Figure title
    """
    T = ego_traj['pos'].shape[0]
    N = sur_traj_original['pos'].shape[0]

    # Convert to numpy
    ego_pos = ego_traj['pos'].cpu().numpy()
    ego_yaw = ego_traj['yaw'].cpu().numpy()

    sur_pos_orig = sur_traj_original['pos'].cpu().numpy()
    sur_yaw_orig = sur_traj_original['yaw'].cpu().numpy()

    sur_pos_opt = sur_traj_optimized['pos'].cpu().numpy()
    sur_yaw_opt = sur_traj_optimized['yaw'].cpu().numpy()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Colors
    ego_color = 'royalblue'
    sur_colors = ['crimson', 'orange', 'green', 'purple'][:N]

    # Left plot: Original trajectories
    ax1 = axes[0]
    ax1.set_title("Original Trajectories", fontsize=14, fontweight='bold')

    # Plot ego trajectory
    ax1.plot(ego_pos[:, 0], ego_pos[:, 1],
             color=ego_color, linewidth=2, label='Ego', zorder=2)

    # Plot surrounding vehicles (original)
    for n in range(N):
        ax1.plot(sur_pos_orig[n, :, 0], sur_pos_orig[n, :, 1],
                 color=sur_colors[n], linewidth=2, linestyle='--',
                 label=f'Sur{n+1} (Original)', zorder=2)

    # Draw vehicles at key timesteps
    key_times = [0, T//4, T//2, 3*T//4, T-1]
    for t in key_times:
        alpha = 0.3 + 0.7 * (t / (T-1))  # Fade in over time

        draw_vehicle(ax1, ego_pos[t, 0], ego_pos[t, 1],
                     ego_yaw[t, 0], ego_color, alpha=alpha)

        for n in range(N):
            draw_vehicle(ax1, sur_pos_orig[n, t, 0], sur_pos_orig[n, t, 1],
                         sur_yaw_orig[n, t, 0], sur_colors[n], alpha=alpha)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend(loc='best')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # Right plot: Optimized trajectories
    ax2 = axes[1]
    ax2.set_title("KING Optimized Trajectories", fontsize=14, fontweight='bold')

    # Plot ego trajectory (same as original)
    ax2.plot(ego_pos[:, 0], ego_pos[:, 1],
             color=ego_color, linewidth=2, label='Ego', zorder=2)

    # Plot surrounding vehicles (original as dashed, optimized as solid)
    for n in range(N):
        ax2.plot(sur_pos_orig[n, :, 0], sur_pos_orig[n, :, 1],
                 color=sur_colors[n], linewidth=1, linestyle=':',
                 alpha=0.5, label=f'Sur{n+1} (Original)', zorder=1)
        ax2.plot(sur_pos_opt[n, :, 0], sur_pos_opt[n, :, 1],
                 color=sur_colors[n], linewidth=2, linestyle='-',
                 label=f'Sur{n+1} (Optimized)', zorder=2)

    # Draw vehicles at key timesteps
    for t in key_times:
        alpha = 0.3 + 0.7 * (t / (T-1))

        draw_vehicle(ax2, ego_pos[t, 0], ego_pos[t, 1],
                     ego_yaw[t, 0], ego_color, alpha=alpha)

        for n in range(N):
            draw_vehicle(ax2, sur_pos_opt[n, t, 0], sur_pos_opt[n, t, 1],
                         sur_yaw_opt[n, t, 0], sur_colors[n], alpha=alpha)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend(loc='best')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()
    return fig


def run_king_optimization_with_viz(excel_path, config=None, output_dir=None):
    """
    Run KING optimization on Excel data and visualize results.

    Uses global action fitting to find action sequences that reproduce
    the original trajectory with < 0.01m average error.
    """
    if config is None:
        config = KINGConfig()

    device = config.device
    delta_t = config.delta_t

    print(f"=" * 60)
    print(f"KING Optimization with Visualization")
    print(f"=" * 60)
    print(f"Input: {excel_path}")
    print(f"Device: {device}")
    print(f"Delta_t: {delta_t}s")

    # Load data
    print("\n[1/4] Loading data...")
    loader = ExcelDataLoader(config)
    data = loader.load_and_interpolate(excel_path)

    if 'sur_state' not in data:
        print("No surrounding vehicles found!")
        return None

    ego_state = data['ego_state']
    sur_state = data['sur_state']
    timestamps = data['timestamps']

    T = ego_state['pos'].shape[0]
    N_sur = sur_state['pos'].shape[0]

    print(f"  Timesteps: {T}")
    print(f"  Duration: {T * delta_t:.1f}s")
    print(f"  Surrounding vehicles: {N_sur}")

    # Fit actions using global optimization
    print("\n[2/4] Global action fitting (target: < 0.01m error)...")

    bicycle_model = BicycleModel(delta_t=delta_t).to(device)

    fitted_actions = []
    for n in range(N_sur):
        print(f"\n  Fitting actions for Sur{n+1}...")

        pos = sur_state['pos'][n]  # (T, 2)
        yaw = sur_state['yaw'][n]  # (T, 1)
        vel = sur_state['vel'][n]  # (T, 2)

        throttle_seq, steer_seq, avg_error = compute_actions_from_trajectory_global(
            pos, yaw, vel, delta_t, device,
            opt_iters=3000, lr=0.01, verbose=True
        )

        if avg_error > 0.01:
            print(f"  WARNING: avg_error={avg_error:.4f}m > 0.01m threshold")
        else:
            print(f"  SUCCESS: avg_error={avg_error:.4f}m < 0.01m")

        fitted_actions.append({
            'throttle': throttle_seq,
            'steer': steer_seq
        })

    # Initialize KING policy with fitted actions
    print("\n[3/4] Running KING optimization...")

    # Create map environment
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(cur_dir, '..', 'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(map_data_path=map_path, device=device, load_lanegraph=False)

    # Create simulator
    simulator = FixedEgoSimulator(config, map_env, num_agents=N_sur)

    # Create policy
    args = config.to_args()
    args.num_agents = N_sur

    B = 1  # batch size
    adv_policy = BMActionSequence(
        args=args,
        batch_size=B,
        num_agents=N_sur,
        sim_horizon=T - 1,
    ).to(device)

    # Initialize with fitted actions
    action_seqs = []
    for b in range(B):
        batch_actions = []
        for n in range(N_sur):
            agent_actions = []
            for t in range(T - 1):
                agent_actions.append({
                    'throttle': fitted_actions[n]['throttle'][t].view(1),
                    'steer': fitted_actions[n]['steer'][t].item(),
                    'brake': torch.tensor([0.0], device=device),
                })
            batch_actions.append(agent_actions)
        action_seqs.append(batch_actions)

    adv_policy.initialize_non_critical_actions(action_seqs)

    # Store original trajectory (simulated with fitted actions, before KING optimization)
    sur_init = {
        'pos': sur_state['pos'][:, 0, :].unsqueeze(0).expand(B, N_sur, 2).clone(),
        'vel': sur_state['vel'][:, 0, :].unsqueeze(0).expand(B, N_sur, 2).clone(),
        'yaw': sur_state['yaw'][:, 0, :].unsqueeze(0).expand(B, N_sur, 1).clone(),
    }

    ego_traj = {
        'pos': ego_state['pos'],
        'yaw': ego_state['yaw'],
        'vel': ego_state['vel'],
    }

    # Get original simulated trajectory
    with torch.no_grad():
        result_orig = simulator.simulate(ego_traj, adv_policy, sur_init)
        sur_traj_original = {
            'pos': result_orig['sur_pos'][:, 0, :, :].clone(),  # (T, N, 2)
            'yaw': result_orig['sur_yaw'][:, 0, :, :].clone(),  # (T, N, 1)
        }
        # Reshape to (N, T, 2) for visualization
        sur_traj_original = {
            'pos': sur_traj_original['pos'].permute(1, 0, 2),
            'yaw': sur_traj_original['yaw'].permute(1, 0, 2),
        }
        print(f"  Original costs: ego_col={result_orig['ego_col'].item():.4f}, "
              f"adv_rd={result_orig['adv_rd'].item():.4f}")

    # Run KING optimization
    opt = torch.optim.Adam(
        adv_policy.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )

    best_cost = float('inf')
    best_sur_traj = None

    for iter_idx in range(config.opt_iters):
        opt.zero_grad(set_to_none=True)
        result = simulator.simulate(ego_traj, adv_policy, sur_init)
        loss = result['total_cost']
        loss.backward()

        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(adv_policy.parameters(), config.gradient_clip)

        opt.step()

        if loss.item() < best_cost:
            best_cost = loss.item()
            best_sur_traj = {
                'pos': result['sur_pos'][:, 0, :, :].detach().clone().permute(1, 0, 2),
                'yaw': result['sur_yaw'][:, 0, :, :].detach().clone().permute(1, 0, 2),
            }

        if iter_idx % 20 == 0 or iter_idx == config.opt_iters - 1:
            print(f"  Iter {iter_idx}/{config.opt_iters}: loss={loss.item():.4f}, "
                  f"ego_col={result['ego_col'].item():.4f}, "
                  f"adv_rd={result['adv_rd'].item():.4f}")

    print(f"\n  Best cost: {best_cost:.4f}")

    # Visualize
    print("\n[4/4] Visualizing results...")

    if output_dir is None:
        output_dir = os.path.dirname(excel_path)

    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_king_viz.png")

    visualize_king_result(
        ego_traj={'pos': ego_state['pos'], 'yaw': ego_state['yaw']},
        sur_traj_original=sur_traj_original,
        sur_traj_optimized=best_sur_traj,
        timestamps=timestamps,
        output_path=output_path,
        title=f"KING Optimization: {base_name}"
    )

    return {
        'ego_traj': ego_traj,
        'sur_traj_original': sur_traj_original,
        'sur_traj_optimized': best_sur_traj,
        'timestamps': timestamps,
        'best_cost': best_cost
    }


if __name__ == '__main__':
    # Use king_test dataset
    data_dir = '/home/hj/RACE_STRIVE/data/race_scenarios/king_test'
    excel_files = glob.glob(os.path.join(data_dir, 'driving_data_*.xlsx'))

    if not excel_files:
        print(f"No Excel files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(excel_files)} Excel files")
    print(f"Using first file: {excel_files[0]}")

    # Create config
    config = KINGConfig(
        opt_iters=151,
        learning_rate=0.005,
    )

    # Run optimization and visualization
    result = run_king_optimization_with_viz(excel_files[0], config)

    if result:
        print("\n" + "=" * 60)
        print("KING optimization complete!")
        print(f"Best cost: {result['best_cost']:.4f}")
        print("=" * 60)
