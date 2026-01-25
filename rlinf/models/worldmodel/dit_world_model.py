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
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from PIL import Image

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
    - Maintains robot state (position, rotation, gripper) for closed-loop control

    Configuration keys:
        world_model_checkpoint (str): Path to DiT checkpoint
        action_dim (int): Expected action dimension (e.g., 20 for cloth folding with Pi0.5)
        batch_size (int): Number of parallel episodes
        gen_num_image_each_step (int): Number of frames to generate per action step
        max_episode_steps (int): Maximum steps per episode
    """

    @staticmethod
    def _r6_to_rotation(r6: np.ndarray) -> R:
        """Convert r6 representation back to a scipy Rotation.

        r6 is produced by: rot_matrix[:, :2].T.flatten()
        So r6 reshaped to (2, 3) gives row vectors that are the first two columns transposed.
        """
        cols = r6.reshape(2, 3).T  # (3, 2) - first two columns of rotation matrix
        # Gram-Schmidt orthogonalization
        a1 = cols[:, 0].copy()
        a1 = a1 / np.linalg.norm(a1)
        a2 = cols[:, 1].copy()
        a2 = a2 - np.dot(a2, a1) * a1
        a2 = a2 / np.linalg.norm(a2)
        a3 = np.cross(a1, a2)
        rot_matrix = np.column_stack([a1, a2, a3])
        return R.from_matrix(rot_matrix)

    @staticmethod
    def _rotation_to_r6(rot: R) -> np.ndarray:
        """Convert scipy Rotation to r6 representation."""
        matrix = rot.as_matrix()
        return matrix[:, :2].T.flatten()

    def __init__(self, cfg: dict[str, Any], dataset: Any, device: Any):
        """Initialize DiT world model backend.

        Args:
            cfg: Configuration dictionary with world model settings
            dataset: Dataset providing initial frames and instructions
            device: Device for computation
        """
        # Initialize base class
        super().__init__(cfg, dataset, device)

        # Initialize state tracking for closed-loop control
        # States are tracked per batch element: [batch_size, state_dim]
        self.current_states = None

        # Frame saving for visualization (only env 0)
        self.save_dir = Path("/scratch/nl2752/RLinf_inspect")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counter = 0

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
        print(f"[DiT WorldModel] _infer_next_frames called - actions shape: {actions.shape}, frame_counter={self.frame_counter}")
        print(f"[DiT WorldModel] BEFORE world model [env 0]: pos_delta={actions[0,:3]}, gripper_L={actions[0,9]:.3f}, gripper_R={actions[0,19]:.3f}")
        print(f"[DiT WorldModel] Actions [env 0] full: {actions[0]}")
        print(f"[DiT WorldModel] Actions range: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
        
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
        
        print(f"[DiT WorldModel] actions_expanded shape: {actions_expanded.shape}")
        print(f"[DiT WorldModel] Calling world_model.generate_chunk...")
        print(f"[DiT WorldModel] World model state before: curr_frame={self.world_model.curr_frame}, xs.shape={self.world_model.xs.shape if hasattr(self.world_model, 'xs') else 'N/A'}")

        # Generate frames using world model
        # The world model's generate_chunk expects (batch_size, num_chunks, action_dim)
        generated_frames_list = []

        for frame_idx, frames in self.world_model.generate_chunk(actions_expanded):
            # frames shape: (batch_size, 1, H, W, C) in [0, 1] range
            print(f"[DiT WorldModel] Generated frame {frame_idx}: shape={frames.shape}, range=[{frames.min():.3f}, {frames.max():.3f}]")
            generated_frames_list.append(frames)

        # Verify we generated the expected number of frames
        assert len(generated_frames_list) == self.gen_num_image_each_step, \
            f"Generated {len(generated_frames_list)} frames, expected {self.gen_num_image_each_step}"
        
        print(f"[DiT WorldModel] Generated {len(generated_frames_list)} frames, frame_counter before saving: {self.frame_counter}")

        # Convert generated frames to RLinf observation format
        return_obs_list = []
        for frames in generated_frames_list:
            # frames: (batch_size, 1, H, W, C) in [0, 1]
            obs_list = []
            for i in range(self.batch_size):
                # Get single frame: (1, H, W, C)
                frame_single = frames[i : i + 1]

                # Save frame for env 0
                if i == 0:
                    frame_uint8 = (frame_single.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    save_path = self.save_dir / f"frame_{self.frame_counter:04d}.png"
                    Image.fromarray(frame_uint8).save(save_path)
                    print(f"[DiT WorldModel] Saved frame {self.frame_counter} to {save_path.name}")
                    self.frame_counter += 1

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

        # Add robot state for closed-loop control
        if self.current_states is not None:
            obs["observation.state"] = self.current_states[batch_idx].clone()
            if batch_idx == 0:  # Only print for first batch to avoid spam
                print(f"[DiT WorldModel] Returning REAL state [env {batch_idx}]: {obs['observation.state'].cpu().numpy()[:6]}... (first 6 dims)")
        else:
            # Fallback to zeros if states not initialized
            obs["observation.state"] = torch.zeros(
                self.dataset.action_dim, dtype=torch.float32, device=self.device
            )
            if batch_idx == 0:
                print(f"[DiT WorldModel] WARNING: Returning ZERO state [env {batch_idx}] - states not initialized!")

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

        # Save initial frame for env 0
        self.frame_counter = 0
        initial_frame_uint8 = (initial_frames[0].squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(initial_frame_uint8).save(self.save_dir / f"frame_{self.frame_counter:04d}_initial.png")
        print(f"[DiT WorldModel] Saved initial frame: frame_{self.frame_counter:04d}_initial.png")
        self.frame_counter += 1

        # Load initial robot states from dataset episodes
        self._load_initial_states_from_episodes()

        return obs, info

    def _load_initial_states_from_episodes(self) -> None:
        """Load initial robot states from dataset episodes.

        For Pi0.5 with dual-arm robots, the state is:
        [left_pos(3), left_r6(6), left_gripper(1), right_pos(3), right_r6(6), right_gripper(1)] = 20D
        """
        # Initialize state tensor: [batch_size, state_dim]
        self.current_states = torch.zeros(
            self.batch_size, self.dataset.action_dim, dtype=torch.float32, device=self.device
        )

        for i in range(self.batch_size):
            episode = self.episodes[i]
            # Try to get initial state from episode data
            if "initial_state" in episode:
                # If episode has initial state, use it
                initial_state = episode["initial_state"]
                if isinstance(initial_state, np.ndarray):
                    self.current_states[i] = torch.from_numpy(initial_state).to(self.device)
                elif isinstance(initial_state, torch.Tensor):
                    self.current_states[i] = initial_state.to(self.device)
            else:
                # Otherwise, use default initial pose
                # This is a reasonable default for dual-arm manipulation
                self._set_default_initial_state(i)

        print(f"[DiT WorldModel] Loaded initial states for {self.batch_size} environments")
        print(f"[DiT WorldModel] Initial state [env 0]: {self.current_states[0].cpu().numpy()[:10]}... (first 10 dims)")

    def _set_default_initial_state(self, batch_idx: int) -> None:
        """Set a default initial state for dual-arm robot.

        Args:
            batch_idx: Index in batch to set initial state for
        """
        state = np.zeros(self.dataset.action_dim, dtype=np.float32)

        # Default left arm pose
        left_pos = np.array([0.35646688, 0.0382356, 0.92677665])
        left_quat_wxyz = np.array([-0.08122765, 0.70717267, 0.69836195, -0.07482959])
        left_quat_xyzw = np.array([left_quat_wxyz[1], left_quat_wxyz[2], left_quat_wxyz[3], left_quat_wxyz[0]])
        left_rot = R.from_quat(left_quat_xyzw)
        left_rot_matrix = left_rot.as_matrix()
        left_r6 = left_rot_matrix[:, :2].T.flatten()

        state[0:3] = left_pos
        state[3:9] = left_r6
        state[9] = 1.0  # Left gripper open

        # Default right arm pose
        right_pos = np.array([0.35906339, -0.45805741, 0.92072002])
        right_quat_xyzw = np.array([0.70221996, 0.70186009, -0.07412596, -0.09372766])
        right_rot = R.from_quat(right_quat_xyzw)
        right_rot_matrix = right_rot.as_matrix()
        right_r6 = right_rot_matrix[:, :2].T.flatten()

        state[10:13] = right_pos
        state[13:19] = right_r6
        state[19] = 1.0  # Right gripper open

        self.current_states[batch_idx] = torch.from_numpy(state).to(self.device)

    def _update_states_with_actions(self, actions: torch.Tensor) -> None:
        """Update robot states by integrating actions.

        Actions are in delta format for position and rotation, absolute for gripper.
        For dual-arm robots with r6 rotation representation:
        - Positions: current_pos += action_pos (additive)
        - Rotations: current_rot = current_rot @ action_rot (compositional)
        - Grippers: current_gripper = action_gripper (absolute)

        Args:
            actions: Action tensor of shape [batch_size, action_dim]
        """
        print(f"[DiT WorldModel] _update_states_with_actions called with shape: {actions.shape}")
        
        if self.current_states is None:
            return

        # Convert to numpy for easier manipulation with scipy
        import torch
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        if isinstance(self.current_states, torch.Tensor):
            states_np = self.current_states.cpu().numpy()
        else:
            states_np = self.current_states
        
        # # Debug: print state before update for first environment
        # print(f"[DiT WorldModel] State update - Action [env 0]: pos_delta={actions_np[0][:3]}, gripper_L_action={actions_np[0][9]:.3f}, gripper_R_action={actions_np[0][19]:.3f}")
        # print(f"[DiT WorldModel] State update - Before [env 0]: pos={states_np[0][:3]}, gripper_L={states_np[0][9]:.3f}, gripper_R={states_np[0][19]:.3f}")

        for i in range(self.batch_size):
            action = actions_np[i]
            state = states_np[i]

            # Update left arm
            # Position: additive delta
            state[0:3] += action[0:3]

            # Rotation: compositional update with r6 representation
            left_rot_current = self._r6_to_rotation(state[3:9])
            left_rot_delta = self._r6_to_rotation(action[3:9])
            left_rot_new = left_rot_current * left_rot_delta
            state[3:9] = self._rotation_to_r6(left_rot_new)

            # Gripper: absolute value
            state[9] = action[9]

            # Update right arm
            # Position: additive delta
            state[10:13] += action[10:13]

            # Rotation: compositional update with r6 representation
            right_rot_current = self._r6_to_rotation(state[13:19])
            right_rot_delta = self._r6_to_rotation(action[13:19])
            right_rot_new = right_rot_current * right_rot_delta
            state[13:19] = self._rotation_to_r6(right_rot_new)

            # Gripper: absolute value
            state[19] = action[19]

            states_np[i] = state

        # Update stored states
        self.current_states = torch.from_numpy(states_np).to(self.device)
        
        # Debug: print state after update for first environment
        print(f"[DiT WorldModel] State update - After [env 0]: pos={states_np[0][:3]}, gripper_L={states_np[0][9]:.3f}, gripper_R={states_np[0][19]:.3f}")

