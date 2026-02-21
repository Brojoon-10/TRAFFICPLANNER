#!/usr/bin/env python3
"""
Test global action fitting - verify that simulated trajectory matches ground truth.
"""
import sys
sys.path.insert(0, '/home/hj/RACE_STRIVE/src')

import torch
import numpy as np
from king.proxy_simulator.motion_model import BicycleModel


def test_global_fitting():
    """Test that global action fitting can reproduce a trajectory."""
    from king_augmentation import compute_actions_from_trajectory_global

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    delta_t = 0.1
    T = 50  # 5 seconds at 10Hz

    # Create a realistic curved trajectory
    # Vehicle starts moving forward, then curves to the right
    bicycle_model = BicycleModel(delta_t=delta_t).to(device)

    # Generate ground truth trajectory by simulating with known actions
    print("\n=== Generating ground truth trajectory ===")
    gt_throttle = torch.ones(T - 1, device=device) * 0.3  # Constant throttle
    gt_steer = torch.linspace(0, 0.3, T - 1, device=device)  # Gradually turning right

    # Initial state
    init_pos = torch.tensor([0.0, 0.0], device=device)
    init_yaw = torch.tensor([0.0], device=device)
    init_vel = torch.tensor([5.0, 0.0], device=device)  # 5 m/s forward

    state = {
        'pos': init_pos.view(1, 1, 2),
        'vel': init_vel.view(1, 1, 2),
        'yaw': init_yaw.view(1, 1, 1),
    }

    # Store ground truth trajectory
    pos_gt = [init_pos.clone()]
    yaw_gt = [init_yaw.clone()]
    vel_gt = [init_vel.clone()]

    is_terminated = torch.zeros(1, device=device, dtype=torch.bool)

    for t in range(T - 1):
        actions = {
            'throttle': gt_throttle[t].view(1, 1, 1),
            'steer': gt_steer[t].view(1, 1, 1),
            'brake': torch.zeros(1, 1, 1, device=device),
        }
        state = bicycle_model(state, actions, is_terminated)

        pos_gt.append(state['pos'][0, 0].clone())
        yaw_gt.append(state['yaw'][0, 0].clone())
        vel_gt.append(state['vel'][0, 0].clone())

    pos_gt = torch.stack(pos_gt)  # (T, 2)
    yaw_gt = torch.stack(yaw_gt)  # (T, 1)
    vel_gt = torch.stack(vel_gt)  # (T, 2)

    print(f"Ground truth trajectory: {T} timesteps")
    print(f"  Start pos: {pos_gt[0].cpu().numpy()}")
    print(f"  End pos: {pos_gt[-1].cpu().numpy()}")
    print(f"  Total distance: {torch.norm(pos_gt[-1] - pos_gt[0]).item():.2f}m")

    # Now try to recover the actions using global fitting
    print("\n=== Global Action Fitting ===")
    recovered_throttle, recovered_steer, avg_error = compute_actions_from_trajectory_global(
        pos_gt, yaw_gt, vel_gt, delta_t, device,
        opt_iters=2000, lr=0.01, verbose=True
    )

    # Compare recovered actions with ground truth
    print("\n=== Action Comparison ===")
    throttle_diff = torch.abs(recovered_throttle - gt_throttle).mean().item()
    steer_diff = torch.abs(recovered_steer - gt_steer).mean().item()
    print(f"Mean throttle difference: {throttle_diff:.4f}")
    print(f"Mean steer difference: {steer_diff:.4f}")

    # Simulate with recovered actions and compare trajectory
    print("\n=== Trajectory Comparison (Simulated vs Ground Truth) ===")
    state = {
        'pos': init_pos.view(1, 1, 2),
        'vel': init_vel.view(1, 1, 2),
        'yaw': init_yaw.view(1, 1, 1),
    }

    simulated_pos = [init_pos.clone()]

    for t in range(T - 1):
        actions = {
            'throttle': recovered_throttle[t].view(1, 1, 1),
            'steer': recovered_steer[t].view(1, 1, 1),
            'brake': torch.zeros(1, 1, 1, device=device),
        }
        state = bicycle_model(state, actions, is_terminated)
        simulated_pos.append(state['pos'][0, 0].clone())

    simulated_pos = torch.stack(simulated_pos)

    # Compute errors at each timestep
    errors = torch.norm(simulated_pos - pos_gt, dim=1)
    print(f"Position errors:")
    print(f"  Mean: {errors.mean().item():.4f}m")
    print(f"  Max: {errors.max().item():.4f}m")
    print(f"  Final: {errors[-1].item():.4f}m")

    # Success criteria
    success = errors.mean().item() < 0.1  # Less than 10cm average error
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Average error {'<' if success else '>='} 0.1m")

    return success


def test_with_real_data():
    """Test with actual Excel data if available."""
    import os
    import glob

    data_dir = '/home/hj/RACE_STRIVE/data/race_scenarios'
    excel_files = glob.glob(os.path.join(data_dir, '*.xlsx'))

    if not excel_files:
        print("No Excel files found in data directory, skipping real data test")
        return True

    print(f"\n=== Testing with real data: {excel_files[0]} ===")

    from king_augmentation import ExcelDataLoader, compute_actions_from_trajectory_global

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    delta_t = 0.1

    loader = ExcelDataLoader(delta_t=delta_t, device=device)
    data = loader.load_and_interpolate(excel_files[0])

    if data is None:
        print("Failed to load data")
        return False

    # Test with first surrounding vehicle
    if data['sur_state']['pos'].shape[0] > 0:
        sur_pos = data['sur_state']['pos'][0]  # (T, 2)
        sur_yaw = data['sur_state']['yaw'][0]  # (T, 1)
        sur_vel = data['sur_state']['vel'][0]  # (T, 2)

        print(f"Surrounding vehicle trajectory: {sur_pos.shape[0]} timesteps")

        recovered_throttle, recovered_steer, avg_error = compute_actions_from_trajectory_global(
            sur_pos, sur_yaw, sur_vel, delta_t, device,
            opt_iters=2000, lr=0.01, verbose=True
        )

        print(f"Final average error: {avg_error:.4f}m")

        return avg_error < 1.0  # Allow up to 1m error for real data

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("GLOBAL ACTION FITTING TEST")
    print("=" * 60)

    success1 = test_global_fitting()

    print("\n" + "=" * 60)
    success2 = test_with_real_data()

    print("\n" + "=" * 60)
    print(f"Test 1 (Synthetic): {'PASSED' if success1 else 'FAILED'}")
    print(f"Test 2 (Real Data): {'PASSED' if success2 else 'FAILED'}")
