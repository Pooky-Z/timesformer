import os
import glob
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare TAD dataset CSVs")
    parser.add_argument("--data_root", required=True, help="Path to TAD dataset root (containing frames/)")
    parser.add_argument("--output_dir", required=True, help="Directory to save train.csv and val.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set")
    args = parser.parse_args()

    frames_root = os.path.join(args.data_root, "frames")
    if not os.path.exists(frames_root):
        print(f"Error: {frames_root} does not exist.")
        return

    # Classes: normal=0, abnormal=1
    # Structure:
    # frames/
    #   abnormal/
    #     video1/
    #   normal/
    #     video2/
    
    samples = []
    
    # Abnormal
    abnormal_dir = os.path.join(frames_root, "abnormal")
    if os.path.exists(abnormal_dir):
        for vid in os.listdir(abnormal_dir):
            vid_path = os.path.join(abnormal_dir, vid)
            if os.path.isdir(vid_path):
                # Check if it has images
                if len(glob.glob(os.path.join(vid_path, "*.jpg"))) > 0:
                    # Store relative path from data_root or absolute path?
                    # The dataset class joins PATH_PREFIX with path.
                    # If PATH_PREFIX is empty, we need absolute path or relative to cwd.
                    # Let's use absolute path for simplicity, and set PATH_PREFIX to "" in config.
                    # Or relative to data_root.
                    # Let's use relative path to data_root.
                    rel_path = os.path.relpath(vid_path, args.data_root)
                    samples.append((rel_path, 1))
    
    # Normal
    normal_dir = os.path.join(frames_root, "normal")
    if os.path.exists(normal_dir):
        for vid in os.listdir(normal_dir):
            vid_path = os.path.join(normal_dir, vid)
            if os.path.isdir(vid_path):
                if len(glob.glob(os.path.join(vid_path, "*.jpg"))) > 0:
                    rel_path = os.path.relpath(vid_path, args.data_root)
                    samples.append((rel_path, 0))

    print(f"Found {len(samples)} samples.")
    
    random.shuffle(samples)
    
    val_size = int(len(samples) * args.val_ratio)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "train.csv"), "w") as f:
        for path, label in train_samples:
            f.write(f"{path} {label}\n")
            
    with open(os.path.join(args.output_dir, "val.csv"), "w") as f:
        for path, label in val_samples:
            f.write(f"{path} {label}\n")
            
    print(f"Saved {len(train_samples)} training samples and {len(val_samples)} validation samples to {args.output_dir}")

if __name__ == "__main__":
    main()
