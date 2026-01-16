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

"""DiT-based world model backend for RLinf.

This module implements a world model backend using Diffusion Transformer (DiT)
architecture from the world-model-eval package for video generation-based RL training.
Follows the SimpleVLA-RL implementation pattern where GPT evaluation happens outside
the world model backend after rollout completion.
"""

from typing import Any
import numpy as np
import torch

from rlinf.models.worldmodel.base_fake_model import BaseFakeModelInference
from rlinf.utils.worldmodel import (
    load_world_model,
    worldmodel_frame_to_obs_dict,
)


class DiTWorldModelInference(BaseFakeModelInference):
    """DiT-based world model backend that generates video predictions.

    This class extends BaseFakeModelInference to use a Diffusion Transformer world model
    for generating future frames conditioned on robot actions.

    Following SimpleVLA-RL design:
    - World model only generates frames during rollout
    - Rewards are computed externally (via GPT-4o) after full rollout completion
    - The world model backend doesn't handle reward calculation

    Configuration keys:
        world_model_checkpoint (str): Path to DiT checkpoint
        action_dim (int): Expected action dimension (e.g., 20 for cloth folding with Pi0.5)
        batch_size (int): Number of parallel episodes
        gen_num_image_each_step (int): Number of frames to generate per action step
        max_episode_steps (int): Maximum steps per episode
    """

    def __init__(self, cfg: dict[str, Any], dataset: Any, device: Any):
        """Initialize DiT world model backend.

        Args:
            cfg: Configuration dictionary with world model settings
            dataset: Dataset providing initial frames and instructions
            device: Device for computation
        """
        # Initialize base class
        super().__init__(cfg, dataset, device)

        print(f"[DiT WorldModel] Initialized with action_dim={self.dataset.action_dim}")
        print(f"[DiT WorldModel] Batch size: {self.batch_size}")
        print(f"[DiT WorldModel] Gen steps per action: {self.gen_num_image_each_step}")

    def _load_model(self) -> None:
        """Load DiT world model from checkpoint."""
        checkpoint_path = self.cfg["world_model_checkpoint"]
        rank = torch.cuda.current_device() if torch.cuda.is_available() else 0

        # Load world model using utility function
        self.world_model = load_world_model(checkpoint_path, rank)

        # Set chunk size to 1 for single-step generation
        self.world_model.chunk_size = 1

        print(f"[DiT WorldModel] Loaded from {checkpoint_path}")
        print(f"[DiT WorldModel] World model chunk size: {self.world_model.chunk_size}")

    def _infer_next_frames(self, actions: torch.Tensor) -> list[list[dict[str, Any]]]:
        """Generate next frames using DiT world model.

        Args:
            actions: Action tensor of shape (batch_size, action_dim)
                    For Pi0.5 with cloth folding: action_dim = 20

        Returns:
            List of observation lists for each generated time step
        """
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device)
        else:
            actions = actions.to(self.device)
        if actions.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            actions = actions.float()

        # Validate action dimensions
        assert actions.shape[0] == self.batch_size, \
            f"Batch size mismatch: got {actions.shape[0]}, expected {self.batch_size}"
        assert actions.shape[1] == self.dataset.action_dim, \
            f"Action dim mismatch: got {actions.shape[1]}, expected {self.dataset.action_dim}"

        # Expand actions to match gen_num_image_each_step
        # Shape: (batch_size, gen_num_image_each_step, action_dim)
        actions_expanded = actions.unsqueeze(1).expand(
            -1, self.gen_num_image_each_step, -1
        )

        # Generate frames using world model
        # The world model's generate_chunk expects (batch_size, num_chunks, action_dim)
        generated_frames_list = []

        for frame_idx, frames in self.world_model.generate_chunk(actions_expanded):
            # frames shape: (batch_size, 1, H, W, C) in [0, 1] range
            generated_frames_list.append(frames)

        # Verify we generated the expected number of frames
        assert len(generated_frames_list) == self.gen_num_image_each_step, \
            f"Generated {len(generated_frames_list)} frames, expected {self.gen_num_image_each_step}"

        # Convert generated frames to RLinf observation format
        return_obs_list = []
        for frames in generated_frames_list:
            # frames: (batch_size, 1, H, W, C) in [0, 1]
            obs_list = []
            for i in range(self.batch_size):
                # Get single frame: (1, H, W, C)
                frame_single = frames[i : i + 1]

                # Convert to observation dict matching RLinf format
                obs_dict = self._convert_frame_to_obs(frame_single, i)
                obs_list.append(obs_dict)

            return_obs_list.append(obs_list)

        return return_obs_list

    def _convert_frame_to_obs(
        self, frame: torch.Tensor, batch_idx: int
    ) -> dict[str, Any]:
        """Convert world model frame to RLinf observation format.

        Args:
            frame: Generated frame of shape (1, H, W, C) in [0, 1] range
            batch_idx: Index in batch to get task description

        Returns:
            Observation dictionary with camera images, state, and task
        """
        # Remove batch/time dimensions: (1, 1, H, W, C) -> (H, W, C)
        frame_squeezed = frame.squeeze(0)
        if frame_squeezed.ndim == 4 and frame_squeezed.shape[0] == 1:
            frame_squeezed = frame_squeezed[0]

        # Convert to uint8 [0, 255] range
        frame_np = frame_squeezed.cpu().numpy()
        frame_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

        # Create observation dict for each camera
        obs = {}
        for camera_name in self.camera_names:
            obs[camera_name] = torch.from_numpy(frame_uint8).to(self.device)

        # Add dummy state (not used for WorldGym, but required by RLinf format)
        obs["observation.state"] = torch.zeros(
            self.dataset.action_dim, dtype=torch.float32, device=self.device
        )

        # Add task description from episode data
        obs["task"] = self.episodes[batch_idx]["task"]

        return obs

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment and world model state.

        Args:
            seed: Random seed
            options: Reset options including episode_id

        Returns:
            Initial observations and info dict
        """
        # Call parent reset to initialize episodes
        obs, info = super().reset(seed, options)

        # Initialize world model with initial frames from dataset
        # The world model needs to be primed with the initial frames
        initial_frames = []
        for i in range(self.batch_size):
            # Get initial frame from episode
            episode = self.episodes[i]
            start_frame = episode["start_items"][-1]  # Use last start item

            # Extract camera image (use first camera)
            camera_name = self.camera_names[0]
            frame_tensor = start_frame[camera_name]  # (H, W, C) uint8

            # Convert to float [0, 1] and add batch dim
            frame_float = frame_tensor.float() / 255.0  # (H, W, C)
            frame_batched = frame_float.unsqueeze(0)  # (1, H, W, C)

            initial_frames.append(frame_batched)

        # Stack initial frames: (batch_size, H, W, C)
        initial_frames_stacked = torch.cat(initial_frames, dim=0).to(self.device)

        # Prime world model with initial frames
        self.world_model.reset(initial_frames_stacked)

        return obs, info
