"""
End-to-end alignment verification for 57x57 grid.
Proves: loss soft label, viz soft label, drivable fraction, GT dots all align with road.
"""
import sys, os
os.chdir('/home/hj/RACE_STRIVE')
sys.path.insert(0, 'src')

import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

from models.trafficplanner_model import TrafficPlannerModel
from datasets.fit_dataset import FITDataset
from datasets.fit_map_env import FITMapEnv
from utils.common import dict2obj
from utils.torch import load_state


def render_map_bg(map_obs_np):
    colors = ['darkgray', 'coral', 'orange', 'gold', 'lightblue', 'lightblue']
    alphas = [1.0, 0.6, 0.6, 0.6, 1.0, 0.5]
    disp_h, disp_w = map_obs_np.shape[2], map_obs_np.shape[1]
    img = np.ones((disp_h, disp_w, 3))
    for i in range(map_obs_np.shape[0]):
        c = np.array(mcolors.to_rgba(colors[i % len(colors)])[:3])
        a = alphas[i % len(alphas)]
        mask = map_obs_np[i].T
        for ch in range(3):
            img[:,:,ch] = img[:,:,ch] * (1 - mask*a) + c[ch] * mask * a
    return img


def overlay(ax, map_bg, grid_data, title, rf_s, rf_o, pix, bounds):
    dh, dw = map_bg.shape[:2]
    ax.imshow(map_bg, origin='lower', extent=[0, dw, 0, dh])
    gd = grid_data.T
    gs_h, gs_w = gd.shape
    up = zoom(gd, (4, 4), order=1)
    vmax = up.max()
    if vmax > 0: up = up / vmax
    hm = plt.cm.jet(up)
    hm[:,:,3] = up * 0.7
    half = rf_s * 0.5
    x0 = (rf_o - half) / pix * dw
    x1 = (rf_o + (gs_w-1)*rf_s + half) / pix * dw
    y0 = (rf_o - half) / pix * dh
    y1 = (rf_o + (gs_h-1)*rf_s + half) / pix * dh
    ax.imshow(hm, origin='lower', extent=[x0, x1, y0, y1])
    ax.plot((0-bounds[0])/(bounds[2]-bounds[0])*dw,
            (0-bounds[1])/(bounds[3]-bounds[1])*dh, 'w*', markersize=12)
    ax.set_title(title, fontsize=9); ax.axis('off')
    ax.set_xlim(0, dw); ax.set_ylim(0, dh)


def drivable_fraction(map_obs, gs, rf_s, rf_o):
    drv = map_obs[0]
    result = np.zeros((gs, gs))
    half = int(rf_s * 1.5)
    H, W = drv.shape
    for li in range(gs):
        for wi in range(gs):
            pl, pw = int(rf_o + li*rf_s), int(rf_o + wi*rf_s)
            patch = drv[max(0,pl-half):min(H,pl+half+1), max(0,pw-half):min(W,pw+half+1)]
            result[li, wi] = patch.mean() if patch.size > 0 else 0
    return result


def loss_style_soft_label(gt_l, gt_w, gs, bounds, pix, rf_s, rf_o, remaining=6, sigma_m=3.0):
    result = np.zeros((gs, gs))
    for k in range(min(remaining, len(gt_l))):
        pix_l = (gt_l[k] - bounds[0]) / (bounds[2]-bounds[0]) * pix
        pix_w = (gt_w[k] - bounds[1]) / (bounds[3]-bounds[1]) * pix
        tok_l = (pix_l - rf_o) / rf_s
        tok_w = (pix_w - rf_o) / rf_s
        LI, WI = np.meshgrid(np.arange(gs), np.arange(gs), indexing='ij')
        sig = sigma_m / (bounds[2]-bounds[0]) * pix / rf_s
        result += np.exp(-((LI-tok_l)**2 + (WI-tok_w)**2) / (2*sig**2))
    v = result.max()
    if v > 0: result /= v
    return result


def main():
    # Load config directly from YAML
    with open('configs/train_trafficplanner.cfg') as f:
        raw = yaml.safe_load(f)

    # Set defaults for missing keys
    defaults = dict(
        map_obs_bounds=[-17.0, -38.5, 60.0, 38.5],
        map_obs_size_pix=256,
        map_layers=['drivable_area', 'solid_line', 'dashed_line'],
        agent_types=['car', 'truck'],
        past_len=4, future_len=12, dt=0.5,
        reduce_cats=False, model_output_bicycle=True,
        map_feat_size=64, past_feat_size=64, future_feat_size=64,
        latent_size=32, z_local_size=32,
        conv_kernel_list=[7,5,5,3,3,3],
        conv_stride_list=[2,2,1,2,2,2],
        conv_filter_list=[16,32,64,64,128,128],
        num_intents=9, sur_pred_dim=2,
        map_recrop=True,
        trans_num_layers=4, trans_d_model=128, trans_nhead=8,
        trans_ffn_dim=512, trans_dropout=0.1,
        use_ego_z_local=True, use_sur_z_local=True,
        use_a2a_rel_bias=False, num_z_queries=4,
        context_num_layers=2, map_summary_tokens=8,
    )
    for k, v in defaults.items():
        if k not in raw:
            raw[k] = v
    cfg = dict2obj(raw)

    device = 'cuda:0'
    out_dir = 'out/train_trafficplanner_out_trans_encoder_no_tf'
    ckpt_path = os.path.join(out_dir, 'checkpoints_trafficplanner/epoch_00000040_model.pth')
    bounds = cfg.map_obs_bounds
    pix = cfg.map_obs_size_pix

    # Map env + dataset
    data_path = os.path.join('src', 'maps', 'centerline_added_boston.osm')
    map_env = FITMapEnv(data_path, bounds=bounds, L=pix, W=pix,
                        layers=cfg.map_layers, device=device)
    dataset = FITDataset(data_path, map_env, split='val',
                         categories=cfg.agent_types,
                         npast=cfg.past_len, nfuture=cfg.future_len,
                         dt=cfg.dt, reduce_cats=cfg.reduce_cats)

    # Model
    model = TrafficPlannerModel(
        cfg.past_len, cfg.future_len, pix, len(dataset.categories),
        map_feat_size=cfg.map_feat_size,
        past_feat_size=cfg.past_feat_size,
        future_feat_size=cfg.future_feat_size,
        latent_size=cfg.latent_size,
        z_local_size=cfg.z_local_size,
        output_bicycle=cfg.model_output_bicycle,
        dt=cfg.dt,
        conv_channel_in=map_env.num_layers,
        conv_kernel_list=cfg.conv_kernel_list,
        conv_stride_list=cfg.conv_stride_list,
        conv_filter_list=cfg.conv_filter_list,
        map_recrop=cfg.map_recrop,
        trans_num_layers=cfg.trans_num_layers,
        trans_d_model=cfg.trans_d_model,
        trans_nhead=cfg.trans_nhead,
        trans_ffn_dim=cfg.trans_ffn_dim,
        trans_dropout=cfg.trans_dropout,
        use_ego_z_local=cfg.use_ego_z_local,
        use_sur_z_local=cfg.use_sur_z_local,
        use_a2a_rel_bias=cfg.use_a2a_rel_bias,
        num_z_queries=cfg.num_z_queries,
        context_num_layers=cfg.context_num_layers,
        map_summary_tokens=cfg.map_summary_tokens,
        num_intents=getattr(cfg, 'num_intents', 9),
        sur_pred_dim=getattr(cfg, 'sur_pred_dim', 2),
    ).to(device)

    load_state(ckpt_path, model, map_location=device)
    model.set_normalizer(dataset.get_state_normalizer())
    model.set_att_normalizer(dataset.get_att_normalizer())
    from datasets.utils import CARLA_BIKE_PARAMS
    if cfg.model_output_bicycle:
        model.set_bicycle_params(CARLA_BIKE_PARAMS)
    model.eval()

    state_norm = dataset.get_state_normalizer()
    rf_s = model.map_rf_stride
    rf_o = model.map_rf_offset
    gs = model.map_token_spatial
    print(f"\n=== grid={gs}x{gs}, rf_stride={rf_s}, rf_offset={rf_o} ===")

    from torch_geometric.data import DataLoader as GraphDataLoader
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)

    for scene_i, (scene_graph, map_idx) in enumerate(loader):
        if scene_i >= 3: break
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)

        ego_mask = scene_graph.sem[:, 0] == 1
        ego_idx = ego_mask.nonzero(as_tuple=True)[0]
        if ego_idx.numel() == 0: continue

        with torch.no_grad():
            ego_pos = state_norm.unnormalize(scene_graph.past[ego_idx[0], -1:]).squeeze(0).cpu().numpy()
            gt_future = state_norm.unnormalize(scene_graph.future[ego_idx[0]]).cpu().numpy()

        ego_t = torch.tensor(ego_pos[:4], device=device, dtype=torch.float32).unsqueeze(0)
        mapix = map_idx[scene_graph.batch[ego_idx[0]]].unsqueeze(0)
        map_obs = map_env.get_map_crop_pos(ego_t, mapix).cpu().numpy()[0]

        FT = gt_future.shape[0]
        rem = min(6, FT)
        hcos, hsin = ego_pos[2], ego_pos[3]
        gt_l, gt_w = np.zeros(rem), np.zeros(rem)
        for k in range(rem):
            dx, dy = gt_future[k,0]-ego_pos[0], gt_future[k,1]-ego_pos[1]
            gt_l[k] = dx*hcos + dy*hsin
            gt_w[k] = -dx*hsin + dy*hcos

        print(f"\nScene {scene_i}: ego=({ego_pos[0]:.1f},{ego_pos[1]:.1f})")
        print(f"  GT local: " + ", ".join(f"({gt_l[k]:.1f},{gt_w[k]:.1f})" for k in range(rem)))

        drv = drivable_fraction(map_obs, gs, rf_s, rf_o)
        lsl = loss_style_soft_label(gt_l, gt_w, gs, bounds, pix, rf_s, rf_o, remaining=rem)

        from viz_attn_intent import compute_soft_label_grid
        ego_frames = np.zeros((FT+1, 4))
        ego_frames[0] = ego_pos[:4]
        ego_frames[1:] = gt_future[:, :4]
        viz_sl = compute_soft_label_grid(gt_future, ego_frames, grid_size=gs,
                                          bounds=bounds, pix_size=pix,
                                          rf_stride=rf_s, rf_offset=rf_o,
                                          sigma_d=1.0, decay_lambda=0.075)
        vsl_t0 = viz_sl[0].reshape(gs, gs) if 0 in viz_sl else None

        gt_dots = np.zeros((gs, gs))
        for k in range(rem):
            pl = (gt_l[k]-bounds[0])/(bounds[2]-bounds[0])*pix
            pw = (gt_w[k]-bounds[1])/(bounds[3]-bounds[1])*pix
            tl, tw = int(round((pl-rf_o)/rf_s)), int(round((pw-rf_o)/rf_s))
            for dl in range(-1,2):
                for dw in range(-1,2):
                    if 0<=tl+dl<gs and 0<=tw+dw<gs:
                        gt_dots[tl+dl,tw+dw] = max(gt_dots[tl+dl,tw+dw], 0.5)
            if 0<=tl<gs and 0<=tw<gs: gt_dots[tl,tw] = 1.0

        map_bg = render_map_bg(map_obs)
        ncols = 5 if vsl_t0 is not None else 4
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))

        dh, dw = map_bg.shape[:2]
        axes[0].imshow(map_bg, origin='lower')
        for k in range(rem):
            px = (gt_l[k]-bounds[0])/(bounds[2]-bounds[0])*dw
            py = (gt_w[k]-bounds[1])/(bounds[3]-bounds[1])*dh
            axes[0].plot(px, py, 'ro', markersize=5)
            axes[0].annotate(str(k), (px,py), color='white', fontsize=7, ha='center')
        axes[0].plot((0-bounds[0])/(bounds[2]-bounds[0])*dw,
                     (0-bounds[1])/(bounds[3]-bounds[1])*dh, 'w*', markersize=15)
        axes[0].set_title(f'Map + GT traj (scene {scene_i})\ngrid={gs}, stride={rf_s}, offset={rf_o}', fontsize=9)
        axes[0].axis('off')

        overlay(axes[1], map_bg, drv, 'Drivable fraction\n(per-token RF)', rf_s, rf_o, pix, bounds)
        overlay(axes[2], map_bg, lsl, 'Loss soft label\n(_make_soft_label)', rf_s, rf_o, pix, bounds)
        overlay(axes[3], map_bg, gt_dots, 'GT position dots\n(token grid)', rf_s, rf_o, pix, bounds)
        if vsl_t0 is not None:
            overlay(axes[4], map_bg, vsl_t0, 'Viz soft label t=0\n(compute_soft_label_grid)', rf_s, rf_o, pix, bounds)

        plt.suptitle(f'E2E Alignment — scene {scene_i} — {gs}x{gs}, stride={rf_s}, offset={rf_o}', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(out_dir, 'viz_attn_intent_epoch00040',
                                 f'e2e_alignment_scene{scene_i}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    print("\nDone. If all panels align with the road, the 57x57 pipeline is correct.")


if __name__ == '__main__':
    main()
