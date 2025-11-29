import torch
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from torchvision import transforms

from timesformer.utils.parser import load_config
from timesformer.models.build import build_model
from timesformer.utils.checkpoint import load_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description="TimeSformer Inference")
    parser.add_argument("--cfg", dest="cfg_file", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (.pyth)")
    parser.add_argument("--video_path", required=True, help="Path to video file or frame folder")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--sampling_rate", type=int, default=None, help="Sampling rate (stride) between frames. If None, uses config default.")
    parser.add_argument("--uniform_sampling", action="store_true", help="Use uniform sampling (like training) instead of sliding window.")
    parser.add_argument("opts", help="See timesformer/config/defaults.py for all options", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def load_frames_from_video(video_path, num_frames=8, sampling_rate=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame is BGR, keep it BGR to match training
        frames.append(frame)
    cap.release()
    return frames

def load_frames_from_folder(folder_path):
    import glob
    files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")))
    # cv2.imread reads BGR, keep it BGR
    frames = [cv2.imread(f) for f in files]
    return frames

def transform_frames(frames, cfg):
    # T H W C -> C T H W
    # But first resize/crop
    
    # Simple inference transform: Resize -> CenterCrop -> ToTensor -> Normalize
    # Note: TimeSformer expects specific normalization
    
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    crop_size = cfg.DATA.TEST_CROP_SIZE
    
    # Resize such that shorter side is crop_size (or slightly larger? usually 256 for 224 crop)
    # Let's follow standard val transform: Resize(256) -> CenterCrop(224)
    # scale_size = 256
    # In TAD dataset val mode, it uses TEST_CROP_SIZE for min_scale, so we should match that.
    scale_size = cfg.DATA.TEST_CROP_SIZE
    
    processed_frames = []
    for img in frames:
        # img is numpy HWC
        h, w, _ = img.shape
        if h < w:
            new_h = scale_size
            new_w = int(w * (scale_size / h))
        else:
            new_w = scale_size
            new_h = int(h * (scale_size / w))
            
        img = cv2.resize(img, (new_w, new_h))
        
        # Center crop
        start_x = (new_w - crop_size) // 2
        start_y = (new_h - crop_size) // 2
        img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        processed_frames.append(img)
        
    # Stack: T H W C
    tensor = torch.tensor(np.stack(processed_frames)).float()
    
    # Normalize
    tensor = tensor / 255.0
    tensor = tensor - torch.tensor(mean)
    tensor = tensor / torch.tensor(std)
    
    # Permute to C T H W
    tensor = tensor.permute(3, 0, 1, 2)
    
    return tensor

def sample_clips(frames, num_frames, sampling_rate, uniform_sampling=False):
    total_frames = len(frames)
    clips = []
    indices = []

    if uniform_sampling:
        # Uniform sampling: divide video into num_frames segments and take center
        seg_size = float(total_frames - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)
        
        # Clamp
        seq = [max(0, min(total_frames - 1, s)) for s in seq]
        
        clip_frames = [frames[i] for i in seq]
        clips.append(clip_frames)
        indices.append(seq[0]) # Start frame
        return clips, indices

    # Sliding window approach
    clip_len = num_frames * sampling_rate
    stride = clip_len // 2 # 50% overlap
    
    if total_frames < clip_len:
        # Pad or just take what we have? 
        # TimeSformer needs fixed T.
        # Let's loop the video if too short, or just take indices with modulo
        start_indices = [0]
    else:
        start_indices = range(0, total_frames - clip_len + 1, stride)
        
    for start in start_indices:
        # Select specific frames
        # seq = [start, start+rate, start+2*rate...]
        seq = [start + i * sampling_rate for i in range(num_frames)]
        # Handle boundary if any (though range check above handles it mostly)
        seq = [min(i, total_frames - 1) for i in seq]
        
        clip_frames = [frames[i] for i in seq]
        clips.append(clip_frames)
        indices.append(start)
        
    return clips, indices

def main():
    args = get_args()
    cfg = load_config(args)
    cfg.NUM_GPUS = 1 # Force 1 GPU for inference
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    # We can use timesformer utils or torch.load directly
    # The checkpoint format in TimeSformer usually has 'model_state'
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load video
    if os.path.isdir(args.video_path):
        frames = load_frames_from_folder(args.video_path)
    else:
        frames = load_frames_from_video(args.video_path)
        
    print(f"Loaded {len(frames)} frames.")
    
    if len(frames) == 0:
        print("No frames found.")
        return

    # Prepare clips
    num_frames = cfg.DATA.NUM_FRAMES
    if args.sampling_rate is not None:
        sampling_rate = args.sampling_rate
    else:
        sampling_rate = cfg.DATA.SAMPLING_RATE
    
    raw_clips, start_indices = sample_clips(frames, num_frames, sampling_rate, uniform_sampling=args.uniform_sampling)
    print(f"Generated {len(raw_clips)} clips (uniform={args.uniform_sampling}).")
    
    results = []
    
    with torch.no_grad():
        for i, clip_frames in enumerate(raw_clips):
            # Transform
            input_tensor = transform_frames(clip_frames, cfg)
            
            # Pack pathway (TimeSformer usually just needs the tensor, but SlowFast needs list)
            # Check utils.pack_pathway_output logic
            if cfg.MODEL.ARCH in ['vit']:
                # TimeSformer (ViT) usually takes tensor directly [B, C, T, H, W]
                inputs = input_tensor.unsqueeze(0).to(device)
            else:
                # SlowFast etc
                # We need to replicate pack_pathway_output logic if using other archs
                # For TimeSformer divST, it's 'vit' arch in config usually?
                # Config says ARCH: vit
                inputs = [input_tensor.unsqueeze(0).to(device)]
                
            # Predict
            preds = model(inputs)
            # preds is [B, NumClasses]
            probs = torch.softmax(preds, dim=1)
            prob = probs[0].cpu().numpy()
            
            # Assuming class 1 is Abnormal, 0 is Normal
            abnormal_score = prob[1]
            results.append(abnormal_score)
            
            print(f"Clip {i} (Start frame {start_indices[i]}): Abnormal Prob = {abnormal_score:.4f}")

    # Aggregate
    avg_score = np.mean(results)
    max_score = np.max(results)
    
    print("-" * 30)
    print(f"Video Prediction Summary:")
    print(f"Average Abnormal Score: {avg_score:.4f}")
    print(f"Max Abnormal Score: {max_score:.4f}")
    
    threshold = 0.5
    if max_score > threshold:
        print("Result: ABNORMAL / ACCIDENT DETECTED")
    else:
        print("Result: Normal")

if __name__ == "__main__":
    main()
