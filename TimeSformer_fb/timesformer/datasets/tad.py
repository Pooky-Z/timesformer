import os
import random
import torch
import torch.utils.data
import numpy as np
from fvcore.common.file_io import PathManager
import glob

import timesformer.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Tad(torch.utils.data.Dataset):
    """
    TAD (Traffic Accident Detection) dataset loader.
    Assumes data is organized in folders of frames.
    """
    def __init__(self, cfg, mode, num_retries=10):
        self.cfg = cfg
        self.mode = mode
        self._num_retries = num_retries
        
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
            
        logger.info("Constructing TAD {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} not found".format(path_to_file)
        
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, line in enumerate(f.read().splitlines()):
                line = line.strip()
                if not line: continue
                # Support space or comma separated
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split(' ')
                
                path = parts[0].strip()
                label = int(parts[1].strip())
                
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                    
        logger.info(f"Constructed TAD dataloader with {len(self._path_to_videos)} clips")

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        
        label = self._labels[index]
        video_dir = self._path_to_videos[index]
        
        # List frames
        frames_paths = sorted(glob.glob(os.path.join(video_dir, "*")))
        # Filter for images
        frames_paths = [p for p in frames_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(frames_paths) == 0:
            logger.warning(f"No frames found in {video_dir}")
            # Fallback: try to return a random other sample? 
            # Or just raise error. For now, let's raise error to be safe.
            raise RuntimeError(f"No frames found in {video_dir}")

        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(frames_paths)
        
        # Temporal sampling
        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)
        
        # Clamp indices
        seq = [max(0, min(video_length - 1, s)) for s in seq]
        
        image_paths = [frames_paths[s] for s in seq]
        
        frames = torch.as_tensor(
            utils.retry_load_images(
                image_paths,
                self._num_retries,
            )
        )
        
        # Normalize
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        
        # T H W C -> C T H W
        frames = frames.permute(3, 0, 1, 2)
        
        # Spatial sampling
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        
        # Pack pathway
        if not self.cfg.MODEL.ARCH in ['vit']:
            frames = utils.pack_pathway_output(self.cfg, frames)
        else:
            # Perform temporal sampling from the fast pathway.
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )
        
        return frames, label, index, {}

    def __len__(self):
        return len(self._path_to_videos)
