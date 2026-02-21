# import torch

# # 저장된 scene graph 파일 경로
# file_path = "/home/hj/RACE_STRIVE/src/scene_graph/pass/scene_000000.pt"

# # 불러오기
# data = torch.load(file_path)

# # 확인 가능한 기본 정보 출력
# print(data)
# print(data.keys)  # 어떤 속성들이 들어 있는지
# print(data.pos)



# import torch
# from pprint import pprint
# from torch_geometric.data import Data

# # 학습 폴더의 첫 파일 경로
# f = "/home/hj/RACE_STRIVE/src/scene_graph/toolkit_dataset/toolkit_dataset_train/scene_000400.pt"

# g = torch.load(f)
# print(hasattr(g, 'batch'), hasattr(g, 'ptr'))

# raw = torch.load(f)
# print("=== 타입 확인 ===")
# print(type(raw))

# if isinstance(raw, tuple):
#     print("\n이 파일은 (graph, map_idx) tuple 입니다.")
#     graph, map_idx = raw
# else:
#     print("\n이 파일은 graph 하나만 저장되어 있습니다.")
#     graph, map_idx = raw, None       # map_idx 없음

# # ───────── graph 내용 살펴보기 ─────────
# print("\n--- graph 속성 / shape ---")
# pprint(graph.__dict__.keys())        # 어떤 필드가 있는지
# for attr in ('past', 'future', 'sem', 'batch', 'ptr'):
#     if hasattr(graph, attr):
#         val = getattr(graph, attr)
#         print(f"{attr}: shape {tuple(val.shape)} | dtype {val.dtype}")



#!/usr/bin/env python3
import os, sys
import glob
import torch

PROJECT_ROOT = "/home/hj/RACE_STRIVE"
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# ──────────────────────────────────────────────────────


def check_pt_files(pt_dir, num_samples=5):
    """
    Load a handful of .pt files, inspect for state_normalizer attributes,
    and print summary information.
    """
    pt_paths = sorted(glob.glob(os.path.join(pt_dir, '*.pt')))
    if not pt_paths:
        print(f"No .pt files found in {pt_dir}")
        return

    total = len(pt_paths)
    sample_paths = pt_paths[:min(num_samples, total)]
    print(f"Found {total} .pt files. Inspecting {len(sample_paths)} samples:\n")

    for path in sample_paths:
        data = torch.load(path)
        print(f"--- {os.path.basename(path)} ---")
        if hasattr(data, 'state_normalizer'):
            sn = data.state_normalizer
            print(f"state_normalizer: {type(sn).__name__}")
            try:
                mean = getattr(sn, 'mean_vals', getattr(sn, 'mean', None))
                std  = getattr(sn, 'std_vals',  getattr(sn, 'std',  None))
                if mean is not None and std is not None:
                    print(f"  mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")
                    print(f"  mean first 5: {mean.flatten()[:5].tolist()}")
                    print(f"  std  first 5: {std.flatten()[:5].tolist()}")
                else:
                    print("  Could not find mean/std attributes on normalizer")
            except Exception as e:
                print(f"  Error reading mean/std: {e}")
        else:
            print("No state_normalizer attribute found.")

        if hasattr(data, 'att_normalizer'):
            an = data.att_normalizer
            print(f"att_normalizer: {type(an).__name__}")
            try:
                mean = getattr(an, 'mean_vals', getattr(an, 'mean', None))
                std  = getattr(an, 'std_vals',  getattr(an, 'std',  None))
                if mean is not None and std is not None:
                    print(f"  att mean shape: {tuple(mean.shape)}, std shape: {tuple(std.shape)}")
                else:
                    print("  Could not find mean/std on att_normalizer")
            except Exception as e:
                print(f"  Error reading att mean/std: {e}")
        else:
            print("No att_normalizer attribute found.")

        print()

if __name__ == "__main__":
    PT_DIR = "/home/hj/RACE_STRIVE/src/scene_graph/toolkit_dataset/toolkit_dataset_normal"
    check_pt_files(PT_DIR, num_samples=5)