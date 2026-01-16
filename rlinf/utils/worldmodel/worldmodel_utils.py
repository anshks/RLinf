# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for world model integration in RLinf.

This module provides utilities for loading and working with DiT-based world models
from the world-model-eval package, adapted for RLinf's training infrastructure.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Import world model utilities from world-model-eval package
# Using local installation at /projects/work/yang-lab2/as20482/RLinf/world-model-eval
try:
    from world_model_eval.utils import predict, rescale_bridge_action, discover_trials
    from world_model_eval.world_model import WorldModel
except ImportError as e:
    print(f"Warning: can't import world_model_eval: {e}")
    print("Please install world-model-eval from /projects/work/yang-lab2/as20482/RLinf/world-model-eval")
    predict = None
    rescale_bridge_action = None
    discover_trials = None
    WorldModel = None


def load_png_to_tensor(png_path: str, target_size: int = 256) -> torch.Tensor:
    """Load PNG file and convert to tensor format expected by world model.

    Args:
        png_path: Path to PNG file
        target_size: Target image size (default 256x256)

    Returns:
        Tensor of shape (H, W, C) with values in [0, 1] range
    """
    img = Image.open(png_path).convert("RGB")
    img = img.resize((target_size, target_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array)


def worldmodel_frame_to_obs_dict(frame: torch.Tensor, camera_name: str = "wrist_camera") -> dict:
    """Convert world model output frame to RLinf observation format.

    This function converts world model generated frames into the observation format
    expected by RLinf's environment infrastructure.

    Args:
        frame: Tensor from world model, shape can be:
            - (H, W, C) for single frame
            - (1, H, W, C) with batch dimension
            - (1, 1, H, W, C) with batch and time dimensions
        camera_name: Name of the camera view (default "wrist_camera")

    Returns:
        Dictionary with camera image in uint8 [0, 255] range
    """
    # Handle different input shapes
    if frame.dim() == 5:  # (batch, time, H, W, C)
        frame_np = frame[0, 0].cpu().numpy()
    elif frame.dim() == 4:  # (batch, H, W, C)
        frame_np = frame[0].cpu().numpy()
    elif frame.dim() == 3:  # (H, W, C)
        frame_np = frame.cpu().numpy()
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    # Convert to uint8 [0, 255] range
    frame_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

    # Return in RLinf observation format
    return {
        camera_name: frame_uint8
    }


def get_worldmodel_checkpoint_config(checkpoint_path: str) -> dict:
    """Get configuration for a specific world model checkpoint.

    Different world model checkpoints may require different configurations for
    pixel RoPE (Rotary Position Embeddings) and CFG (Classifier-Free Guidance).

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with 'use_pixel_rope' and 'default_cfg' values
    """
    # Checkpoint-specific configurations
    CHECKPOINTS_TO_KWARGS = {
        "bridge_v2_ckpt.pt": {
            "use_pixel_rope": True,
            "default_cfg": 1.0,
        },
        "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
            "use_pixel_rope": False,
            "default_cfg": 3.0,
        },
        "ckpt_000480000.pt": {  # LIBERO world model
            "use_pixel_rope": False,
            "default_cfg": 3.0,
        },
    }

    # Extract just the filename if full path is provided
    checkpoint_file = Path(checkpoint_path).name

    # Return config if found, otherwise use conservative defaults
    return CHECKPOINTS_TO_KWARGS.get(
        checkpoint_file,
        {
            "use_pixel_rope": False,
            "default_cfg": 1.0,
        },
    )


def load_world_model(checkpoint_path: str, rank: int):
    """Load and initialize DiT world model from checkpoint.

    This function loads a Diffusion Transformer (DiT) world model with VAE
    from the world-model-eval package, applying checkpoint-specific configurations.

    Args:
        checkpoint_path: Path to world model checkpoint file
        rank: GPU rank for device placement

    Returns:
        Initialized WorldModel instance

    Raises:
        ImportError: If world-model-eval package is not installed
    """
    try:
        from world_model_eval.world_model import WorldModel
    except ImportError as e:
        raise ImportError(
            "world-model-eval package not found. "
            "Please install it from SimpleVLA-RL or run: "
            "cd /scratch/as20482/SimpleVLA-RL/world-model-eval && pip install -e ."
        ) from e

    # Get checkpoint-specific configuration
    config = get_worldmodel_checkpoint_config(checkpoint_path)

    # Load world model with checkpoint path and configuration
    world_model = WorldModel(
        checkpoint_path=checkpoint_path,
        use_pixel_rope=config["use_pixel_rope"],
        default_cfg=config["default_cfg"],
        rank=rank,
    )

    return world_model
