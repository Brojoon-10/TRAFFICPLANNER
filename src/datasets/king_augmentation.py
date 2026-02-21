# Copyright (c) 2024
# Data Augmentation using KING (Kinematic-INformed Gradient) optimization
# Reference: ECCV 2022 - KING: Generating Safety-Critical Driving Scenarios

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from argparse import Namespace

# Add parent directory to path
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

# KING components (NO CARLA dependency)
from king.proxy_simulator.motion_model import BicycleModel
from king.proxy_simulator.bm_policy import BMActionSequence
from king.proxy_simulator.driving_costs import BatchedPolygonCollisionCost, RouteDeviationCostRasterized

# Map environment for road deviation cost
from datasets.fit_map_env import FITMapEnv


@dataclass
class KINGConfig:
    """
    Configuration parameters from KING (generate_scenarios.py defaults).
    Reference: src/king/generate_scenarios.py:500-610

    Note: We use sim_tickrate=10 (dt=0.1s) for balance between precision and compute.
    This gives reasonable optimization steps for our 8-12 second data:
    - 8s data -> 80 steps (similar to KING default horizon)
    - 10s data -> 100 steps
    - 12s data -> 120 steps
    Horizon is set dynamically based on data length.
    """
    # Simulation parameters
    sim_tickrate: int = 10         # dt = 0.1s (KING default: 4 -> 0.25s)
    sim_horizon: int = 100         # Reference only - actual horizon from data length

    # Optimization parameters
    opt_iters: int = 151           # Line 500-503
    learning_rate: float = 0.005   # Line 505-508
    beta1: float = 0.9             # Line 597-599
    beta2: float = 0.999           # Line 601-604
    gradient_clip: float = 0.0     # Line 567-569

    # Cost weights
    w_ego_col: float = 1.0         # Line 577-579: ego-adversary collision weight
    w_adv_col: float = 0.0         # Line 581-584: adversary-adversary collision weight
    w_adv_rd: float = 1.0          # Line 592-594: road deviation weight
    adv_col_thresh: float = 1.25   # Line 587-589

    # Vehicle bounding box extent (half-lengths) from simulator.py:80
    # carla.Vector3D(2.20, .90, .755) -> (length/2, width/2, height/2)
    vehicle_extent_length: float = 2.20   # meters (half)
    vehicle_extent_width: float = 0.90    # meters (half)
    vehicle_extent_height: float = 0.755  # meters (half)

    # Batch size (usually 1 for our use case)
    batch_size: int = 1

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Derived properties
    @property
    def delta_t(self) -> float:
        return 1.0 / self.sim_tickrate  # 0.25s

    @property
    def vehicle_lw(self) -> Tuple[float, float]:
        """Full length and width of vehicle."""
        return (self.vehicle_extent_length * 2, self.vehicle_extent_width * 2)

    def to_args(self) -> Namespace:
        """Convert to argparse.Namespace for compatibility with KING code."""
        return Namespace(
            sim_tickrate=self.sim_tickrate,
            sim_horizon=self.sim_horizon,
            opt_iters=self.opt_iters,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            gradient_clip=self.gradient_clip,
            w_ego_col=self.w_ego_col,
            w_adv_col=self.w_adv_col,
            w_adv_rd=self.w_adv_rd,
            adv_col_thresh=self.adv_col_thresh,
            batch_size=self.batch_size,
            device=self.device,
        )


class ExcelDataLoader:
    """
    Load trajectory data from Excel files and interpolate to KING's timestep.

    Input Excel format (0.5s intervals):
        - ego_x, ego_y, ego_yaw (or quaternion)
        - sur_x, sur_y, sur_yaw (or quaternion)

    Output: Interpolated trajectories at 0.1s intervals (sim_tickrate=10)
    """

    def __init__(self, config: KINGConfig):
        self.config = config
        self.input_dt = 0.5   # Excel data interval
        self.output_dt = config.delta_t  # 0.1s interval (KING default is 0.25s)

    def load_and_interpolate(self, excel_path: str) -> Dict[str, torch.Tensor]:
        """
        Load Excel file and interpolate trajectories to KING timestep.

        Returns:
            dict with keys:
                - 'ego_state': dict with pos (T, 2), vel (T, 2), yaw (T, 1)
                - 'sur_state': dict with pos (N, T, 2), vel (N, T, 2), yaw (N, T, 1)
                - 'timestamps': (T,)
        """
        df = pd.read_excel(excel_path)

        # Original timestamps (0.5s intervals)
        n_original = len(df)
        t_original = np.arange(n_original) * self.input_dt

        # Target timestamps (0.25s intervals)
        t_end = t_original[-1]
        t_interp = np.arange(0, t_end + self.output_dt/2, self.output_dt)

        # Interpolate ego trajectory
        ego_data = self._interpolate_trajectory(
            df, t_original, t_interp,
            x_col='ego_x', y_col='ego_y', yaw_col='ego_yaw'
        )

        # Find surrounding vehicle columns
        sur_cols = self._find_sur_columns(df)
        sur_data_list = []

        for sur_idx, (x_col, y_col, yaw_col) in enumerate(sur_cols):
            sur_data = self._interpolate_trajectory(
                df, t_original, t_interp,
                x_col=x_col, y_col=y_col, yaw_col=yaw_col
            )
            sur_data_list.append(sur_data)

        # Convert to tensors
        device = self.config.device

        result = {
            'ego_state': {
                'pos': torch.tensor(ego_data['pos'], dtype=torch.float32, device=device),
                'yaw': torch.tensor(ego_data['yaw'], dtype=torch.float32, device=device),
                'vel': torch.tensor(ego_data['vel'], dtype=torch.float32, device=device),
            },
            'timestamps': torch.tensor(t_interp, dtype=torch.float32, device=device),
        }

        if sur_data_list:
            result['sur_state'] = {
                'pos': torch.stack([
                    torch.tensor(s['pos'], dtype=torch.float32, device=device)
                    for s in sur_data_list
                ]),  # (N, T, 2)
                'yaw': torch.stack([
                    torch.tensor(s['yaw'], dtype=torch.float32, device=device)
                    for s in sur_data_list
                ]),  # (N, T, 1)
                'vel': torch.stack([
                    torch.tensor(s['vel'], dtype=torch.float32, device=device)
                    for s in sur_data_list
                ]),  # (N, T, 2)
            }

        return result

    def _interpolate_trajectory(self, df: pd.DataFrame, t_orig: np.ndarray,
                                  t_interp: np.ndarray, x_col: str, y_col: str,
                                  yaw_col: str) -> Dict[str, np.ndarray]:
        """Interpolate a single vehicle trajectory."""
        x = df[x_col].values
        y = df[y_col].values

        # Handle yaw (might be quaternion in some datasets)
        if yaw_col in df.columns:
            yaw = df[yaw_col].values
        else:
            # Try to compute from quaternion if available
            yaw = self._extract_yaw_from_df(df, yaw_col.replace('yaw', ''))

        # Cubic interpolation for positions
        interp_x = interp1d(t_orig, x, kind='cubic', fill_value='extrapolate')
        interp_y = interp1d(t_orig, y, kind='cubic', fill_value='extrapolate')

        # Linear interpolation for yaw (unwrap first to handle wrap-around)
        yaw_unwrapped = np.unwrap(yaw)
        interp_yaw = interp1d(t_orig, yaw_unwrapped, kind='linear', fill_value='extrapolate')

        x_interp = interp_x(t_interp)
        y_interp = interp_y(t_interp)
        yaw_interp = interp_yaw(t_interp)

        # Wrap yaw back to [-pi, pi]
        yaw_interp = np.arctan2(np.sin(yaw_interp), np.cos(yaw_interp))

        # Compute velocity from positions
        vel = np.zeros((len(t_interp), 2))
        vel[1:, 0] = np.diff(x_interp) / self.output_dt
        vel[1:, 1] = np.diff(y_interp) / self.output_dt
        vel[0] = vel[1]  # Copy first velocity

        return {
            'pos': np.stack([x_interp, y_interp], axis=1),
            'yaw': yaw_interp.reshape(-1, 1),
            'vel': vel
        }

    def _find_sur_columns(self, df: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """Find surrounding vehicle column groups."""
        sur_cols = []

        # Pattern 1: sur1_x, sur1_y, sur1_yaw, sur2_x, ...
        i = 1
        while f'sur{i}_x' in df.columns or f'sur_{i}_x' in df.columns:
            prefix = f'sur{i}' if f'sur{i}_x' in df.columns else f'sur_{i}'
            sur_cols.append((f'{prefix}_x', f'{prefix}_y', f'{prefix}_yaw'))
            i += 1

        # Pattern 2: sur_x, sur_y (single surrounding vehicle)
        if not sur_cols and 'sur_x' in df.columns:
            sur_cols.append(('sur_x', 'sur_y', 'sur_yaw'))

        return sur_cols

    def _extract_yaw_from_df(self, df: pd.DataFrame, prefix: str) -> np.ndarray:
        """Extract yaw from quaternion columns if yaw column doesn't exist."""
        # Try multiple quaternion column naming conventions
        # Convention 1: qx, qy, qz, qw
        qw_col = f'{prefix}qw' if f'{prefix}qw' in df.columns else f'{prefix}_qw'
        qx_col = f'{prefix}qx' if f'{prefix}qx' in df.columns else f'{prefix}_qx'
        qy_col = f'{prefix}qy' if f'{prefix}qy' in df.columns else f'{prefix}_qy'
        qz_col = f'{prefix}qz' if f'{prefix}qz' in df.columns else f'{prefix}_qz'

        # Convention 2: ox, oy, oz, ow (used in race_scenarios data)
        if not all(c in df.columns for c in [qw_col, qx_col, qy_col, qz_col]):
            qw_col = f'{prefix}ow' if f'{prefix}ow' in df.columns else f'{prefix}_ow'
            qx_col = f'{prefix}ox' if f'{prefix}ox' in df.columns else f'{prefix}_ox'
            qy_col = f'{prefix}oy' if f'{prefix}oy' in df.columns else f'{prefix}_oy'
            qz_col = f'{prefix}oz' if f'{prefix}oz' in df.columns else f'{prefix}_oz'

        if all(c in df.columns for c in [qw_col, qx_col, qy_col, qz_col]):
            quats = np.stack([
                df[qw_col].values,
                df[qx_col].values,
                df[qy_col].values,
                df[qz_col].values
            ], axis=1)
            rot = R.from_quat(quats[:, [1, 2, 3, 0]])  # scipy uses xyzw order
            euler = rot.as_euler('xyz')
            return euler[:, 2]  # yaw is the z rotation
        else:
            raise ValueError(f"Cannot find yaw or quaternion columns for {prefix}")


def compute_actions_from_trajectory_stepwise(
    pos: torch.Tensor,      # (T, 2)
    yaw: torch.Tensor,      # (T, 1)
    vel: torch.Tensor,      # (T, 2)
    delta_t: float,
    device: str = 'cuda',
    opt_iters: int = 100,
    lr: float = 0.1,
    verbose: bool = False
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute throttle/steer actions from trajectory using step-by-step optimization.

    KEY INSIGHT: Start each step from the GROUND TRUTH state (not the predicted state).
    This prevents error accumulation. The actions found will make BicycleModel move
    from state[t] toward state[t+1], which is exactly what we need for initialization.

    For each timestep t:
        1. Start from ground truth state at t (pos[t], yaw[t], vel[t])
        2. Optimize action to minimize distance to state at t+1
        3. Save the best action found

    Args:
        pos: (T, 2) position trajectory
        yaw: (T, 1) yaw trajectory
        vel: (T, 2) velocity trajectory
        delta_t: simulation timestep
        device: cuda or cpu
        opt_iters: optimization iterations per timestep
        lr: learning rate for action optimization
        verbose: print progress

    Returns:
        List of action dicts for each timestep
    """
    from king.proxy_simulator.motion_model import BicycleModel

    T = pos.shape[0]
    action_list = []

    # Create BicycleModel for single-step forward
    bicycle_model = BicycleModel(delta_t=delta_t).to(device)
    bicycle_model.eval()

    total_pos_error = 0.0
    total_yaw_error = 0.0

    for t in range(T - 1):
        # Current ground truth state at t
        current_pos = pos[t]    # (2,)
        current_yaw = yaw[t]    # (1,)
        current_vel = vel[t]    # (2,)

        # Target state at t+1
        target_pos = pos[t + 1]  # (2,)
        target_yaw = yaw[t + 1]  # (1,)
        target_vel = vel[t + 1]  # (2,)

        # Learnable action parameters (use tanh to bound actions to [-1, 1])
        throttle_param = torch.nn.Parameter(torch.zeros(1, device=device))
        steer_param = torch.nn.Parameter(torch.zeros(1, device=device))

        optimizer = torch.optim.Adam([throttle_param, steer_param], lr=lr)

        best_loss = float('inf')
        best_throttle = 0.0
        best_steer = 0.0

        for i in range(opt_iters):
            optimizer.zero_grad()

            # Create state dict from GROUND TRUTH at time t
            state = {
                'pos': current_pos.view(1, 1, 2).clone(),
                'vel': current_vel.view(1, 1, 2).clone(),
                'yaw': current_yaw.view(1, 1, 1).clone(),
            }

            # Create action dict
            actions = {
                'throttle': torch.tanh(throttle_param).view(1, 1, 1),
                'steer': torch.tanh(steer_param).view(1, 1, 1),
                'brake': torch.zeros(1, 1, 1, device=device),
            }

            # Forward step
            is_terminated = torch.zeros(1, device=device, dtype=torch.bool)
            next_state = bicycle_model(state, actions, is_terminated)

            # Compute loss
            pred_pos = next_state['pos'][0, 0]  # (2,)
            pred_yaw = next_state['yaw'][0, 0]  # (1,)
            pred_vel = next_state['vel'][0, 0]  # (2,)

            # Position loss (most important)
            pos_loss = torch.sum((pred_pos - target_pos) ** 2)

            # Yaw loss (handle wrap-around)
            yaw_diff = pred_yaw - target_yaw
            yaw_loss = torch.sum((torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))) ** 2)

            # Velocity loss (help match dynamics)
            vel_loss = torch.sum((pred_vel - target_vel) ** 2)

            # Total loss
            loss = pos_loss + 0.5 * yaw_loss + 0.1 * vel_loss

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_throttle = torch.tanh(throttle_param).item()
                best_steer = torch.tanh(steer_param).item()

            # Early stopping if good enough
            if loss.item() < 1e-4:
                break

            loss.backward()
            optimizer.step()

        # Save action with best values
        action_list.append({
            'throttle': torch.tensor([[[best_throttle]]], device=device, dtype=torch.float32),
            'steer': best_steer,
            'brake': torch.tensor([[[0.0]]], device=device, dtype=torch.float32),
        })

        # Compute final error for this step (for reporting)
        with torch.no_grad():
            state = {
                'pos': current_pos.view(1, 1, 2),
                'vel': current_vel.view(1, 1, 2),
                'yaw': current_yaw.view(1, 1, 1),
            }
            actions = {
                'throttle': torch.tensor([[[best_throttle]]], device=device),
                'steer': torch.tensor([[[best_steer]]], device=device),
                'brake': torch.zeros(1, 1, 1, device=device),
            }
            is_terminated = torch.zeros(1, device=device, dtype=torch.bool)
            next_state = bicycle_model(state, actions, is_terminated)

            pos_error = torch.norm(next_state['pos'][0, 0] - target_pos).item()
            yaw_error = torch.abs(next_state['yaw'][0, 0, 0] - target_yaw[0]).item()
            total_pos_error += pos_error
            total_yaw_error += yaw_error

        if verbose and (t % 10 == 0 or t == T - 2):
            print(f"  Step {t}/{T-2}: pos_err={pos_error:.4f}m, yaw_err={np.degrees(yaw_error):.2f}deg, "
                  f"throttle={best_throttle:.3f}, steer={best_steer:.3f}")

    avg_pos_error = total_pos_error / (T - 1)
    avg_yaw_error = total_yaw_error / (T - 1)
    print(f"Action fitting complete: avg_pos_err={avg_pos_error:.4f}m, avg_yaw_err={np.degrees(avg_yaw_error):.2f}deg")

    return action_list


def compute_actions_from_trajectory_global(
    pos: torch.Tensor,      # (T, 2)
    yaw: torch.Tensor,      # (T, 1)
    vel: torch.Tensor,      # (T, 2)
    delta_t: float,
    device: str = 'cuda',
    opt_iters: int = 2000,
    lr: float = 0.01,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute throttle/steer actions from trajectory using GLOBAL optimization.

    KEY INSIGHT: Optimize ALL actions simultaneously so that the ENTIRE simulated
    trajectory matches the ground truth. This accounts for error accumulation
    during simulation.

    The optimization minimizes:
        sum_t ||simulated_pos[t] - gt_pos[t]||² + yaw_loss + vel_loss

    where simulated trajectory is computed by:
        state[0] = gt[0]
        for t in range(T-1):
            state[t+1] = BicycleModel(state[t], action[t])

    Args:
        pos: (T, 2) position trajectory
        yaw: (T, 1) yaw trajectory
        vel: (T, 2) velocity trajectory
        delta_t: simulation timestep
        device: cuda or cpu
        opt_iters: total optimization iterations
        lr: learning rate
        verbose: print progress

    Returns:
        throttle_seq: (T-1,) optimized throttle values
        steer_seq: (T-1,) optimized steer values
        final_error: average position error after optimization
    """
    import sys
    sys.path.insert(0, '/home/hj/RACE_STRIVE/src')
    from king.proxy_simulator.motion_model import BicycleModel

    T = pos.shape[0]

    # Create BicycleModel
    bicycle_model = BicycleModel(delta_t=delta_t).to(device)

    # Learnable action parameters for ALL timesteps
    # Use raw parameters, apply tanh during forward to bound to [-1, 1]
    throttle_params = torch.nn.Parameter(torch.zeros(T - 1, device=device))
    steer_params = torch.nn.Parameter(torch.zeros(T - 1, device=device))

    optimizer = torch.optim.Adam([throttle_params, steer_params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=verbose
    )

    best_loss = float('inf')
    best_throttle = throttle_params.detach().clone()
    best_steer = steer_params.detach().clone()

    # Ground truth tensors
    pos_gt = pos.to(device)
    yaw_gt = yaw.to(device)
    vel_gt = vel.to(device)

    for iter_idx in range(opt_iters):
        optimizer.zero_grad()

        # Initialize state from ground truth at t=0
        state = {
            'pos': pos_gt[0].view(1, 1, 2).clone(),
            'vel': vel_gt[0].view(1, 1, 2).clone(),
            'yaw': yaw_gt[0].view(1, 1, 1).clone(),
        }

        total_pos_loss = torch.tensor(0.0, device=device)
        total_yaw_loss = torch.tensor(0.0, device=device)
        total_vel_loss = torch.tensor(0.0, device=device)

        is_terminated = torch.zeros(1, device=device, dtype=torch.bool)

        # Simulate forward using current action parameters
        for t in range(T - 1):
            # Get bounded actions via tanh
            throttle_t = torch.tanh(throttle_params[t]).view(1, 1, 1)
            steer_t = torch.tanh(steer_params[t]).view(1, 1, 1)

            actions = {
                'throttle': throttle_t,
                'steer': steer_t,
                'brake': torch.zeros(1, 1, 1, device=device),
            }

            # Forward one step (state is modified in-place, need to handle carefully)
            state = {
                'pos': state['pos'].clone(),
                'vel': state['vel'].clone(),
                'yaw': state['yaw'].clone(),
            }
            state = bicycle_model(state, actions, is_terminated)

            # Compare with ground truth at t+1
            pred_pos = state['pos'][0, 0]  # (2,)
            pred_yaw = state['yaw'][0, 0]  # (1,)
            pred_vel = state['vel'][0, 0]  # (2,)

            gt_pos_t = pos_gt[t + 1]
            gt_yaw_t = yaw_gt[t + 1]
            gt_vel_t = vel_gt[t + 1]

            # Position loss
            total_pos_loss = total_pos_loss + torch.sum((pred_pos - gt_pos_t) ** 2)

            # Yaw loss (handle wrap-around)
            yaw_diff = pred_yaw - gt_yaw_t
            total_yaw_loss = total_yaw_loss + torch.sum(
                torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)) ** 2
            )

            # Velocity loss
            total_vel_loss = total_vel_loss + torch.sum((pred_vel - gt_vel_t) ** 2)

        # Total loss (weighted)
        loss = total_pos_loss + 0.5 * total_yaw_loss + 0.1 * total_vel_loss

        # Action smoothness regularization
        if T > 2:
            throttle_smooth = torch.sum((throttle_params[1:] - throttle_params[:-1]) ** 2)
            steer_smooth = torch.sum((steer_params[1:] - steer_params[:-1]) ** 2)
            loss = loss + 0.01 * (throttle_smooth + steer_smooth)

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_throttle = torch.tanh(throttle_params).detach().clone()
            best_steer = torch.tanh(steer_params).detach().clone()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([throttle_params, steer_params], max_norm=1.0)

        optimizer.step()
        scheduler.step(loss)

        if verbose and (iter_idx % 200 == 0 or iter_idx == opt_iters - 1):
            avg_pos_err = torch.sqrt(total_pos_loss / (T - 1)).item()
            print(f"  Iter {iter_idx}/{opt_iters}: loss={loss.item():.4f}, "
                  f"avg_pos_err={avg_pos_err:.4f}m")

    # Compute final trajectory error with best actions
    with torch.no_grad():
        state = {
            'pos': pos_gt[0].view(1, 1, 2),
            'vel': vel_gt[0].view(1, 1, 2),
            'yaw': yaw_gt[0].view(1, 1, 1),
        }
        is_terminated = torch.zeros(1, device=device, dtype=torch.bool)

        total_error = 0.0
        max_error = 0.0

        for t in range(T - 1):
            actions = {
                'throttle': best_throttle[t].view(1, 1, 1),
                'steer': best_steer[t].view(1, 1, 1),
                'brake': torch.zeros(1, 1, 1, device=device),
            }
            state = bicycle_model(state, actions, is_terminated)

            error = torch.norm(state['pos'][0, 0] - pos_gt[t + 1]).item()
            total_error += error
            max_error = max(max_error, error)

        avg_error = total_error / (T - 1)
        print(f"Global action fitting complete: avg_err={avg_error:.4f}m, max_err={max_error:.4f}m")

    return best_throttle, best_steer, avg_error


def compute_actions_from_trajectory(
    pos: torch.Tensor,      # (T, 2)
    yaw: torch.Tensor,      # (T, 1)
    vel: torch.Tensor,      # (T, 2)
    delta_t: float,
    device: str = 'cuda'
) -> List[Dict[str, torch.Tensor]]:
    """
    Compute actions using GLOBAL optimization (simulates full trajectory).
    Returns in the format expected by BMActionSequence.initialize_non_critical_actions()
    """
    throttle_seq, steer_seq, avg_error = compute_actions_from_trajectory_global(
        pos, yaw, vel, delta_t, device,
        opt_iters=2000, lr=0.01, verbose=True
    )

    # Convert to list of dicts format
    action_list = []
    for t in range(len(throttle_seq)):
        action_list.append({
            'throttle': throttle_seq[t].view(1),
            'steer': steer_seq[t].item(),
            'brake': torch.tensor([0.0], device=device),
        })

    return action_list


class FixedEgoSimulator:
    """
    Simulator that keeps ego trajectory fixed while optimizing surrounding vehicles.

    Uses KING's BicycleModel interface:
        - forward(state, actions, is_terminated)
        - state: dict with 'pos', 'vel', 'yaw'
        - actions: dict with 'throttle', 'steer', 'brake'
    """

    def __init__(self, config: KINGConfig, map_env: FITMapEnv, num_agents: int):
        self.config = config
        self.map_env = map_env
        self.device = torch.device(config.device)
        self.num_agents = num_agents

        # Create args namespace for KING cost functions
        # They need: batch_size, num_agents, device
        self.args = config.to_args()
        self.args.num_agents = num_agents

        # Motion model from KING - requires delta_t
        # Reference: motion_model.py:10
        self.motion_model = BicycleModel(delta_t=config.delta_t).to(self.device)

        # Cost functions from KING (driving_costs.py)
        # Uses KING's original interface
        self.collision_cost = BatchedPolygonCollisionCost(self.args)
        self.road_deviation_cost = RouteDeviationCostRasterized(self.args)

        # Vehicle extent from KING config (simulator.py:471-480)
        # Shape: (B, 1, 2) for ego, (B, N, 2) for adv
        # Note: KING uses (width, length) order in extent
        # Reference: simulator.py:471-480
        self.ego_extent = torch.tensor(
            [config.vehicle_extent_width, config.vehicle_extent_length],
            device=self.device, dtype=torch.float32
        ).view(1, 1, 2).expand(config.batch_size, 1, 2)  # (B, 1, 2)
        self.adv_extent = torch.tensor(
            [config.vehicle_extent_width, config.vehicle_extent_length],
            device=self.device, dtype=torch.float32
        ).view(1, 1, 2).expand(config.batch_size, num_agents, 2)  # (B, N, 2)

        # Store map raster for road deviation cost
        # map_env.nusc_raster: (1, C, H, W) -> take first channel
        # Map shape: (H, W) = (3438, 5086)
        #
        # KING's driving_costs.py has been modified:
        # - crop_map(row, col, h_extent, w_extent) with map[row_range, col_range]
        # - h_extent = size(0) = H, w_extent = size(1) = W
        # - pos[0] = row (bounded by H), pos[1] = col (bounded by W)
        self.road_raster = map_env.nusc_raster[0, 0, :, :].to(self.device)  # (H, W) = (3438, 5086)

    def world_to_pix(self, world_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to pixel coordinates for KING's RouteDeviationCostRasterized.

        Modified KING's driving_costs.py uses:
        - crop_map(row, col, h_extent, w_extent) with map[row_range, col_range]
        - h_extent = size(0) = H = 3438, w_extent = size(1) = W = 5086
        - pos[0] = row (bounded by H), pos[1] = col (bounded by W)

        Boston transform: pixel_x (col) = x/0.25 + 2164, pixel_y (row) = 1809 - y/0.25

        Args:
            world_pos: (N, 2) in world frame [x, y]

        Returns:
            pixel_pos: (N, 2) where pos[0]=row (pixel_y), pos[1]=col (pixel_x)
        """
        # Boston Seaport transform from nuscenes_utils.py:268-269
        pixel_x = world_pos[:, 0] / 0.25 + 2164  # col, valid: 0~5085
        pixel_y = 1809 - world_pos[:, 1] / 0.25  # row, valid: 0~3437

        # Map shape: (H, W) = (3438, 5086)
        # Clamp to ensure crop window stays in bounds (±32 pixels)
        H, W = 3438, 5086
        pixel_y = torch.clamp(pixel_y, 32, H - 33)  # row, bounded by h_extent=H=3438
        pixel_x = torch.clamp(pixel_x, 32, W - 33)  # col, bounded by w_extent=W=5086

        pixel_pos = torch.zeros_like(world_pos)
        pixel_pos[:, 0] = pixel_y  # pos[0] = row, bounded by h_extent=size(0)=H=3438
        pixel_pos[:, 1] = pixel_x  # pos[1] = col, bounded by w_extent=size(1)=W=5086
        return pixel_pos

    def simulate(self, ego_trajectory: Dict[str, torch.Tensor],
                 adv_policy: BMActionSequence,
                 sur_init_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run simulation with fixed ego and optimizable surrounding vehicles.

        Args:
            ego_trajectory: Pre-recorded ego trajectory
                - pos: (T, 2)
                - yaw: (T, 1)
                - vel: (T, 2)
            adv_policy: BMActionSequence for surrounding vehicles
            sur_init_state: Initial state of surrounding vehicles
                - pos: (B, N, 2)
                - vel: (B, N, 2)
                - yaw: (B, N, 1)

        Returns:
            Dictionary with simulation results including costs
        """
        T = ego_trajectory['pos'].shape[0]
        B = sur_init_state['pos'].shape[0]  # batch size
        N_sur = sur_init_state['pos'].shape[1]  # num agents

        # Initialize state dict for BicycleModel
        # Shape: (B, N, dim) where N = num_agents
        state = {
            'pos': sur_init_state['pos'].clone(),  # (B, N, 2)
            'vel': sur_init_state['vel'].clone(),  # (B, N, 2)
            'yaw': sur_init_state['yaw'].clone(),  # (B, N, 1)
        }

        # Termination mask - no termination in our case
        is_terminated = torch.zeros(B, device=self.device, dtype=torch.bool)

        # Cost lists for aggregation (matching KING's generate_scenarios.py:337)
        cost_dict = {"ego_col": [], "adv_col": [], "adv_rd": []}

        # Trajectory lists (collect in same loop to avoid duplicate simulation)
        sur_pos_list = [state['pos'].clone()]
        sur_yaw_list = [state['yaw'].clone()]

        # Simulation loop (reference: generate_scenarios.py:352-386)
        for t in range(T):
            # Get ego state at this timestep
            # KING format: (B, 1, dim)
            ego_state = {
                'pos': ego_trajectory['pos'][t:t+1].unsqueeze(0).expand(B, 1, 2),  # (B, 1, 2)
                'yaw': ego_trajectory['yaw'][t:t+1].unsqueeze(0).expand(B, 1, 1),  # (B, 1, 1)
            }

            # Get adv state at this timestep
            # KING format: (B, N, dim)
            adv_state = {
                'pos': state['pos'],  # (B, N, 2)
                'yaw': state['yaw'],  # (B, N, 1)
            }

            # Compute costs using KING's cost functions
            # Reference: generate_scenarios.py:401-405, compute_cost()
            ego_col_cost, adv_col_cost, _ = self.collision_cost(
                ego_state,
                self.ego_extent,  # (B, 1, 2)
                adv_state,
                self.adv_extent,  # (B, N, 2)
            )

            # Handle empty adv_col_cost (reference: generate_scenarios.py:407-409)
            if adv_col_cost.size(-1) == 0:
                adv_col_cost = torch.zeros(1, 1, device=self.device)

            # Clamp adv_col_cost (reference: generate_scenarios.py:411-413)
            adv_col_cost = torch.minimum(
                adv_col_cost,
                torch.tensor([self.config.adv_col_thresh], device=self.device)
            )

            # Road deviation cost
            # Reference: generate_scenarios.py:415-418
            # crop_center should be (2,) - single ego position
            adv_rd_cost = self.road_deviation_cost(
                self.road_raster,           # (H, W)
                adv_state['pos'],           # (B, N, 2)
                adv_state['yaw'],           # (B, N, 1)
                ego_state['pos'][0, 0, :],  # (2,) crop center
                self.world_to_pix           # world to pixel function
            )

            cost_dict["ego_col"].append(ego_col_cost)
            cost_dict["adv_col"].append(adv_col_cost)
            cost_dict["adv_rd"].append(adv_rd_cost)

            # Step to next state (except for last timestep)
            if t < T - 1:
                observations = {'timestep': t}
                actions = adv_policy(observations)
                state = self.motion_model(state, actions, is_terminated)
                # Collect trajectory
                sur_pos_list.append(state['pos'].clone())
                sur_yaw_list.append(state['yaw'].clone())

        # Aggregate costs (reference: generate_scenarios.py:131-154)
        # ego_col: mean over time, min over agents
        ego_col_stacked = torch.stack(cost_dict["ego_col"], dim=1)  # (B, T, ...)
        ego_col_agg = torch.min(
            torch.mean(ego_col_stacked, dim=1),
            dim=1
        )[0]  # (B,)

        # adv_col: min over time and agents
        adv_col_stacked = torch.stack(cost_dict["adv_col"], dim=1)
        adv_col_agg = torch.min(
            torch.min(adv_col_stacked, dim=1)[0],
            dim=1
        )[0]  # (B,)

        # adv_rd: mean over time
        adv_rd_stacked = torch.stack(cost_dict["adv_rd"], dim=1)
        adv_rd_agg = torch.mean(adv_rd_stacked, dim=1)  # (B,)

        # Total objective (reference: generate_scenarios.py:150-154)
        total_objective = (
            self.config.w_ego_col * ego_col_agg.mean() +
            self.config.w_adv_rd * adv_rd_agg.mean() +
            -1 * self.config.w_adv_col * adv_col_agg.mean()  # negative to encourage spread
        )

        # Stack trajectories
        sur_pos_traj = torch.stack(sur_pos_list, dim=0)  # (T, B, N, 2)
        sur_yaw_traj = torch.stack(sur_yaw_list, dim=0)  # (T, B, N, 1)

        return {
            'sur_pos': sur_pos_traj,
            'sur_yaw': sur_yaw_traj,
            'ego_col': ego_col_agg.mean(),
            'adv_col': adv_col_agg.mean(),
            'adv_rd': adv_rd_agg.mean(),
            'total_cost': total_objective
        }

class DirectTrajectoryOptimizer:
    """
    Directly optimize trajectory positions without BicycleModel simulation.

    This avoids the dynamics mismatch between BicycleModel (tuned for CARLA)
    and our Excel data. Instead, we treat the trajectory as a learnable parameter
    and add kinematic feasibility constraints as regularization.
    """

    def __init__(self, config: KINGConfig, map_env: FITMapEnv, num_agents: int):
        self.config = config
        self.map_env = map_env
        self.device = torch.device(config.device)
        self.num_agents = num_agents

        self.args = config.to_args()
        self.args.num_agents = num_agents

        # Cost functions
        self.collision_cost = BatchedPolygonCollisionCost(self.args)
        self.road_deviation_cost = RouteDeviationCostRasterized(self.args)

        # Vehicle extents
        self.ego_extent = torch.tensor(
            [config.vehicle_extent_width, config.vehicle_extent_length],
            device=self.device, dtype=torch.float32
        ).view(1, 1, 2).expand(config.batch_size, 1, 2)
        self.adv_extent = torch.tensor(
            [config.vehicle_extent_width, config.vehicle_extent_length],
            device=self.device, dtype=torch.float32
        ).view(1, 1, 2).expand(config.batch_size, num_agents, 2)

        # Road raster
        self.road_raster = map_env.nusc_raster[0, 0, :, :].to(self.device)

    def world_to_pix(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to pixel coordinates."""
        pixel_x = world_pos[:, 0] / 0.25 + 2164
        pixel_y = 1809 - world_pos[:, 1] / 0.25
        H, W = 3438, 5086
        pixel_y = torch.clamp(pixel_y, 32, H - 33)
        pixel_x = torch.clamp(pixel_x, 32, W - 33)
        pixel_pos = torch.zeros_like(world_pos)
        pixel_pos[:, 0] = pixel_y
        pixel_pos[:, 1] = pixel_x
        return pixel_pos

    def compute_cost(self, ego_pos, ego_yaw, adv_pos, adv_yaw):
        """Compute collision and road deviation costs at a single timestep."""
        B = 1

        ego_state = {
            'pos': ego_pos.view(B, 1, 2),
            'yaw': ego_yaw.view(B, 1, 1),
        }
        adv_state = {
            'pos': adv_pos.view(B, self.num_agents, 2),
            'yaw': adv_yaw.view(B, self.num_agents, 1),
        }

        # Collision cost
        ego_col_cost, adv_col_cost, _ = self.collision_cost(
            ego_state, self.ego_extent, adv_state, self.adv_extent
        )

        # Road deviation cost
        adv_rd_cost = self.road_deviation_cost(
            self.road_raster,
            adv_state['pos'],
            adv_state['yaw'],
            ego_state['pos'][0, 0, :],
            self.world_to_pix
        )

        return ego_col_cost, adv_rd_cost

    def optimize(self, ego_traj, sur_traj_init, timestamps):
        """
        Directly optimize surrounding vehicle trajectory.

        Args:
            ego_traj: dict with 'pos' (T, 2), 'yaw' (T, 1)
            sur_traj_init: dict with 'pos' (N, T, 2), 'yaw' (N, T, 1)
            timestamps: (T,)

        Returns:
            Optimized sur_traj
        """
        T = ego_traj['pos'].shape[0]
        N = sur_traj_init['pos'].shape[0]

        # Make trajectory learnable
        sur_pos = sur_traj_init['pos'].clone().requires_grad_(True)  # (N, T, 2)
        sur_yaw = sur_traj_init['yaw'].clone().requires_grad_(True)  # (N, T, 1)

        optimizer = torch.optim.Adam([sur_pos, sur_yaw], lr=self.config.learning_rate)

        original_pos = sur_traj_init['pos'].clone()  # For regularization

        best_cost = float('inf')
        best_pos = sur_pos.detach().clone()
        best_yaw = sur_yaw.detach().clone()

        for iter_idx in range(self.config.opt_iters):
            optimizer.zero_grad()

            total_ego_col = 0.0
            total_rd = 0.0

            for t in range(T):
                ego_col, rd = self.compute_cost(
                    ego_traj['pos'][t],
                    ego_traj['yaw'][t],
                    sur_pos[:, t, :],  # (N, 2)
                    sur_yaw[:, t, :]   # (N, 1)
                )
                total_ego_col += ego_col
                total_rd += rd

            # Average over time
            avg_ego_col = total_ego_col / T
            avg_rd = total_rd / T

            # Smoothness regularization (penalize jerky motion)
            if T > 2:
                accel = sur_pos[:, 2:, :] - 2 * sur_pos[:, 1:-1, :] + sur_pos[:, :-2, :]
                smoothness = torch.mean(accel ** 2)
            else:
                smoothness = torch.tensor(0.0, device=self.device)

            # Deviation from original (don't go too far)
            deviation = torch.mean((sur_pos - original_pos) ** 2)

            # Total objective
            loss = (
                self.config.w_ego_col * avg_ego_col +
                self.config.w_adv_rd * avg_rd +
                0.01 * smoothness +
                0.001 * deviation
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_cost:
                best_cost = loss.item()
                best_pos = sur_pos.detach().clone()
                best_yaw = sur_yaw.detach().clone()

            if iter_idx % 20 == 0:
                print(f"Iter {iter_idx}: loss={loss.item():.4f}, "
                      f"ego_col={avg_ego_col.item():.4f}, "
                      f"rd={avg_rd.item():.4f}")

        print(f"Direct optimization complete. Best cost: {best_cost:.4f}")

        return {
            'pos': best_pos,
            'yaw': best_yaw
        }


class KINGDataAugmenter:
    """
    Main class for data augmentation using KING optimization.
    """

    def __init__(self, config: Optional[KINGConfig] = None,
                 map_data_path: Optional[str] = None):
        self.config = config or KINGConfig()
        self.device = torch.device(self.config.device)
        self.args = self.config.to_args()

        # Initialize data loader
        self.data_loader = ExcelDataLoader(self.config)

        # Initialize map environment
        # FITMapEnv needs path to OSM file for get_fit_maps()
        # Default: src/maps/centerline_added_boston.osm
        if map_data_path is None:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            map_data_path = os.path.join(cur_dir, '..', 'maps', 'centerline_added_boston.osm')

        self.map_env = FITMapEnv(
            map_data_path=map_data_path,
            device=self.config.device,
            load_lanegraph=False
        )

        # Simulator will be created per-scenario with correct num_agents
        self.simulator = None

    def augment_scenario(self, excel_path: str,
                         output_path: Optional[str] = None,
                         use_direct_optimization: bool = True) -> pd.DataFrame:
        """
        Augment a single scenario from Excel file.

        Args:
            excel_path: Path to input Excel file
            output_path: Path to save augmented Excel (optional)
            use_direct_optimization: If True, optimize trajectory directly without BicycleModel

        Returns:
            DataFrame with augmented trajectories
        """
        # Load and interpolate data
        data = self.data_loader.load_and_interpolate(excel_path)

        # Check if we have surrounding vehicles
        if 'sur_state' not in data:
            print(f"No surrounding vehicles in {excel_path}, skipping...")
            return None

        ego_state = data['ego_state']
        sur_state = data['sur_state']
        timestamps = data['timestamps']

        N_sur = sur_state['pos'].shape[0]
        T = ego_state['pos'].shape[0]

        print(f"Optimizing trajectory for {T} timesteps, {N_sur} surrounding vehicles")

        if use_direct_optimization:
            # Use direct trajectory optimization (avoids BicycleModel dynamics mismatch)
            optimizer = DirectTrajectoryOptimizer(self.config, self.map_env, num_agents=N_sur)

            ego_traj = {
                'pos': ego_state['pos'],
                'yaw': ego_state['yaw'],
            }
            sur_traj_init = {
                'pos': sur_state['pos'],  # (N, T, 2)
                'yaw': sur_state['yaw'],  # (N, T, 1)
            }

            optimized_sur = optimizer.optimize(ego_traj, sur_traj_init, timestamps)

            best_sur_traj = {
                'pos': optimized_sur['pos'].unsqueeze(1),  # (N, 1, T, 2) -> need (T, B, N, 2)
                'yaw': optimized_sur['yaw'].unsqueeze(1),
            }
            # Reshape: (N, T, 2) -> (T, 1, N, 2)
            best_sur_traj = {
                'pos': optimized_sur['pos'].permute(1, 0, 2).unsqueeze(1),  # (T, 1, N, 2)
                'yaw': optimized_sur['yaw'].permute(1, 0, 2).unsqueeze(1),  # (T, 1, N, 1)
            }

        else:
            # Original BicycleModel-based optimization (may have dynamics mismatch)
            B = self.config.batch_size

            self.simulator = FixedEgoSimulator(self.config, self.map_env, num_agents=N_sur)

            sur_init = {
                'pos': sur_state['pos'][:, 0, :].unsqueeze(0).expand(B, N_sur, 2).clone(),
                'vel': sur_state['vel'][:, 0, :].unsqueeze(0).expand(B, N_sur, 2).clone(),
                'yaw': sur_state['yaw'][:, 0, :].unsqueeze(0).expand(B, N_sur, 1).clone(),
            }

            adv_policy = BMActionSequence(
                args=self.args,
                batch_size=B,
                num_agents=N_sur,
                sim_horizon=T - 1,
            ).to(self.device)

            action_seqs = []
            for b in range(B):
                batch_actions = []
                for n in range(N_sur):
                    agent_actions = compute_actions_from_trajectory(
                        pos=sur_state['pos'][n],
                        yaw=sur_state['yaw'][n],
                        vel=sur_state['vel'][n],
                        delta_t=self.config.delta_t,
                        device=self.config.device
                    )
                    batch_actions.append(agent_actions)
                action_seqs.append(batch_actions)

            adv_policy.initialize_non_critical_actions(action_seqs)
            print(f"Initialized policy with {len(action_seqs[0][0])} timesteps of actions")

            opt = torch.optim.Adam(
                adv_policy.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2)
            )

            ego_traj = {
                'pos': ego_state['pos'],
                'yaw': ego_state['yaw'],
                'vel': ego_state['vel'],
            }

            best_cost = float('inf')
            best_sur_traj = None

            for iter_idx in range(self.config.opt_iters):
                opt.zero_grad(set_to_none=True)
                result = self.simulator.simulate(ego_traj, adv_policy, sur_init)
                loss = result['total_cost']
                loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(adv_policy.parameters(),
                        self.config.gradient_clip)

                opt.step()

                if loss.item() < best_cost:
                    best_cost = loss.item()
                    best_sur_traj = {
                        'pos': result['sur_pos'].detach().clone(),
                        'yaw': result['sur_yaw'].detach().clone()
                    }

                if iter_idx % 20 == 0:
                    print(f"Iter {iter_idx}: loss={loss.item():.4f}, "
                          f"ego_col={result['ego_col'].item():.4f}, "
                          f"adv_col={result['adv_col'].item():.4f}, "
                          f"adv_rd={result['adv_rd'].item():.4f}")

            print(f"Optimization complete. Best cost: {best_cost:.4f}")

        # Convert results to DataFrame (take first batch)
        output_df = self._trajectories_to_dataframe(
            ego_traj, best_sur_traj, timestamps
        )

        # Downsample back to 0.5s intervals
        output_df = self._downsample_to_original(output_df)

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_excel(output_path, index=False)
            print(f"Saved augmented scenario to {output_path}")

        return output_df

    def _trajectories_to_dataframe(self, ego_traj: Dict, sur_traj: Dict,
                                    timestamps: torch.Tensor) -> pd.DataFrame:
        """Convert trajectory tensors to DataFrame."""
        T = timestamps.shape[0]

        data = {
            'timestamp': timestamps.cpu().numpy(),
            'ego_x': ego_traj['pos'][:, 0].cpu().numpy(),
            'ego_y': ego_traj['pos'][:, 1].cpu().numpy(),
            'ego_yaw': ego_traj['yaw'][:, 0].cpu().numpy(),
        }

        # Add surrounding vehicle trajectories (take first batch, index 0)
        # sur_traj['pos']: (T, B, N, 2) -> take [:,0,:,:]
        N_sur = sur_traj['pos'].shape[2]
        for n in range(N_sur):
            data[f'sur{n+1}_x'] = sur_traj['pos'][:, 0, n, 0].cpu().numpy()
            data[f'sur{n+1}_y'] = sur_traj['pos'][:, 0, n, 1].cpu().numpy()
            data[f'sur{n+1}_yaw'] = sur_traj['yaw'][:, 0, n, 0].cpu().numpy()

        return pd.DataFrame(data)

    def _downsample_to_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample from 0.1s to 0.5s intervals."""
        # Take every 5th row (0.1s * 5 = 0.5s)
        return df.iloc[::5].reset_index(drop=True)

    def augment_directory(self, input_dir: str, output_dir: str):
        """
        Augment all Excel files in a directory.

        Args:
            input_dir: Directory containing input Excel files
            output_dir: Directory to save augmented files
        """
        os.makedirs(output_dir, exist_ok=True)

        excel_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

        for i, fname in enumerate(excel_files):
            print(f"\n[{i+1}/{len(excel_files)}] Processing {fname}")

            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, f"aug_{fname}")

            try:
                self.augment_scenario(input_path, output_path)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue


def main():
    """Main entry point for data augmentation."""
    import argparse

    parser = argparse.ArgumentParser(description='Data augmentation using KING optimization')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input Excel files')
    parser.add_argument('--output_dir', type=str,
                        default='./data/king_augmented_datasets',
                        help='Directory to save augmented files')
    parser.add_argument('--map_data_path', type=str, default='./data/nuscenes',
                        help='Path to map data')
    parser.add_argument('--opt_iters', type=int, default=151,
                        help='Number of optimization iterations')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate for optimization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Create config with any overrides
    config = KINGConfig(
        opt_iters=args.opt_iters,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Initialize augmenter
    augmenter = KINGDataAugmenter(config=config, map_data_path=args.map_data_path)

    # Run augmentation
    augmenter.augment_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
