import glob, os

scenario_path = "/home/hj/RACE_STRIVE/data/race_scenarios/boston_junction_1"
scene_full_list = sorted(glob.glob(os.path.join(scenario_path, 'driving_data_scenario_*.xlsx')))

n = len(scene_full_list)
half = n // 2
rest = n - half
val_num = rest // 3
test_num = rest - val_num

# split
train_list = scene_full_list[:half]
val_list   = scene_full_list[half:half+val_num]
test_list  = scene_full_list[half+val_num:]

def print_split(name, file_list):
    print(f"\n=== {name.upper()} split ({len(file_list)} scenes) ===")
    for idx, f in enumerate(file_list):
        base = os.path.basename(f)
        orig_num = int(base.split('_')[-1].split('.')[0])
        print(f"{name} idx {idx:3d} -> scenario {orig_num}")

print_split("train", train_list)
print_split("val", val_list)
print_split("test", test_list)

def query(split_name, indices):
    if split_name == "train":
        target = train_list
    elif split_name == "val":
        target = val_list
    elif split_name == "test":
        target = test_list
    else:
        raise ValueError("split_name must be 'train', 'val', or 'test'")
    
    for idx in indices:
        if idx < len(target):
            base = os.path.basename(target[idx])
            orig_num = int(base.split('_')[-1].split('.')[0])
            print(f"{split_name} idx {idx} → origin scenario {orig_num}")
        else:
            print(f"{split_name} idx {idx} → out of range (len {len(target)})")

query("val", [6, 20, 30])
