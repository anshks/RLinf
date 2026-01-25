"""
Pi0.5 Inference Class for LeRobot

This module provides a simple interface for loading and running inference
with a trained Pi0.5 policy model. It supports both raw observation format
(with images dict and qpos) and LeRobot format, with automatic transformation
of actions from r6 rotation representation to quaternions.

Installation:
    See pi05_training and lerobot
    
Example usage (matching sample policy interface):
    from pi05_inference import Pi05InferencePolicy

    # Load the policy
    policy = Pi05InferencePolicy(
        checkpoint_path="/path/to/checkpoint",
        device="cuda"
    )

    # Warmup the model to trigger torch.compile (recommended for first use)
    policy.warmup()

    # Create observation dictionary (raw format)
    observation = {
        "images": {
            "cam_high": np.array(...),  # (H, W, 3) RGB image
            "cam_left": np.array(...),  # (H, W, 3) RGB image
        },
        "depth": np.array(...),  # (H, W) or (H, W, 1) depth frame (optional)
        "qpos": np.array(...),  # (20,) state vector [left_arm(10), right_arm(10)]
        "task": "fold the cloth"  # optional task description
    }

    # Generate action in quaternion format (16D output)
    action = policy.forward(observation)
    # Returns: [left_quat(4), left_pos(3), right_quat(4), right_pos(3),
    #           left_gripper(1), right_gripper(1)]

Alternative usage (LeRobot format):
    observation = {
        "observation.state": torch.tensor(...),
        "observation.images.cam_high": torch.tensor(...),  # (3, H, W)
        "task": "fold the cloth"
    }

    # Get action in model's native format (20D with r6 rotations)
    action = policy.predict(observation)

    # Or get action transformed to quaternion format
    action_quat = policy.predict(observation, transform_to_quat=True)
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from collections import deque

# Ensure imports work regardless of how the script is run
_repo_root = Path(__file__).parent
if _repo_root not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Try importing from lerobot package first, fall back to local imports
try:
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
except (ImportError, ModuleNotFoundError):
    try:
        from policies.pi05.configuration_pi05 import PI05Config
        from policies.pi05.modeling_pi05 import PI05Policy
        from processor.pipeline import PolicyProcessorPipeline
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import LeRobot modules. Make sure you're running from the lerobot repository "
            "or have lerobot installed. Error: " + str(e)
        )


class Pi05InferencePolicy:
    """
    Inference wrapper for Pi0.5 policy model.

    This class handles:
    - Loading pretrained Pi0.5 model from checkpoint
    - Loading preprocessing and postprocessing pipelines
    - Running inference to generate actions from observations

    The checkpoint directory should contain:
        - config.json: Model configuration
        - model.safetensors: Model weights
        - policy_preprocessor.json: Preprocessing pipeline config
        - policy_preprocessor_step_2_normalizer_processor.safetensors: Normalizer state
        - policy_postprocessor.json: Postprocessing pipeline config
        - policy_postprocessor_step_0_unnormalizer_processor.safetensors: Unnormalizer state
        - train_config.json: Training configuration (optional)

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to run inference on ('cuda', 'cpu', etc.)
        verbose: Print loading information
        primary_camera: Which camera to use when model expects single image (e.g., "cam_high").
                       If None and model expects single image, uses first available camera.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        verbose: bool = True,
        primary_camera: str | None = None,
        is_delta_action: bool = True,
        # execution_horizon: int = 10,
        control_freq: float = 30.0,
        inference_delay: int = 3
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.verbose = verbose
        self.primary_camera = primary_camera  # Which camera to use if model expects single image

        self.is_delta_action = is_delta_action

        # Async RTC parameters
        # self.execution_horizon = execution_horizon  # s: actions executed between inferences
        self.prev_chunk = None  # Previous action chunk
        self.control_freq = control_freq  # Robot control frequency (Hz)
        self.inference_delay = inference_delay  # d: inference delay in action steps (manual config)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_path}"
            )

        # Load configuration
        if self.verbose:
            print(f"Loading Pi0.5 model from {self.checkpoint_path}")

        # Load policy using from_pretrained (handles config and weights)
        self.policy = self._load_policy()
        self.config = self.policy.config  # Extract config from loaded policy
        self.policy.to(self.device)
        self.policy.eval()

        # Load preprocessor and postprocessor
        self.preprocessor = self._load_preprocessor()
        self.postprocessor = self._load_postprocessor()

        # Load training config if available (optional, for reference)
        self.train_config = self._load_train_config()

        # Reset internal state
        self.reset()

        if self.verbose:
            print("✓ Pi0.5 model loaded successfully")
            self._print_model_info()

        # for testing only, to be deleted
        self.first_inference = True

    def _load_policy(self) -> PI05Policy:
        """Load policy using from_pretrained method."""
        if self.verbose:
            print(f"  Loading policy from {self.checkpoint_path}")

        # Use the standard from_pretrained method which handles config and weights
        policy = PI05Policy.from_pretrained(str(self.checkpoint_path))

        return policy

    def _load_train_config(self) -> dict | None:
        """Load training configuration if available."""
        train_config_path = self.checkpoint_path / "train_config.json"
        if not train_config_path.exists():
            return None

        with open(train_config_path, "r") as f:
            return json.load(f)

    def _load_preprocessor(self) -> PolicyProcessorPipeline:
        """Load preprocessing pipeline using from_pretrained."""
        if self.verbose:
            print(f"  Loading preprocessor from {self.checkpoint_path}")

        # Use from_pretrained which handles config and state file loading automatically
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint_path),
            config_filename="policy_preprocessor.json"
        )

        return preprocessor

    def _load_postprocessor(self) -> PolicyProcessorPipeline:
        """Load postprocessing pipeline using from_pretrained."""
        if self.verbose:
            print(f"  Loading postprocessor from {self.checkpoint_path}")

        # Use from_pretrained which handles config and state file loading automatically
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint_path),
            config_filename="policy_postprocessor.json"
        )

        return postprocessor

    def _print_model_info(self):
        """Print model information."""
        print("\nModel Information:")
        print(f"  Config: {self.config.__class__.__name__}")

        expected_img_keys = self._get_expected_image_keys()
        if expected_img_keys:
            print(f"  Expected image keys: {expected_img_keys}")
            if "observation.image" in expected_img_keys:
                primary_cam = self.primary_camera or "(first available)"
                print(f"  Primary camera: {primary_cam}")

        if hasattr(self.config, "input_features"):
            print(f"  Input features: {self.config.input_features}")
        if hasattr(self.config, "output_features"):
            print(f"  Output features: {self.config.output_features}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Device: {self.device}")

    def reset(self):
        """Reset policy internal state and async RTC tracking."""
        self.policy.reset()

        # Async RTC state tracking
        self.prev_chunk = None  # Previous action chunk (full chunk_size)
        self.prev_chunk_send_time = None  # Timestamp when chunk was sent
        self.chunk_idx = 0  # Chunk counter

    def warmup(self):
        """
        Warmup the model with dummy inputs to trigger torch.compile compilation.

        This method creates dummy observations with the correct shapes based on
        the model configuration and runs a forward pass to trigger compilation.
        Once completed, prints a notification message signaling compilation is done.

        The shapes are automatically extracted from the checkpoint configuration files.
        """
        # if self.verbose:
        if not self.config.compile_model:
            print("\n" + "=" * 60)
            print("Model is not configured to use torch.compile. Skipping warmup.")
            print("=" * 60)
            return
        print("\n" + "=" * 60)
        print(f"Starting model warmup (triggering torch.compile with mode {self.config.compile_mode})...")
        print("=" * 60)

        # Create dummy observation with correct shapes from config
        dummy_obs = {}

        # Add state if expected
        if self.expected_state_dim > 0:
            dummy_obs["observation.state"] = torch.randn(
                self.expected_state_dim,
                device=self.device
            )
            if self.verbose:
                print(f"  Created dummy state: shape={dummy_obs['observation.state'].shape}")

        # Add images for each expected camera
        if self.expected_image_keys:
            for img_key in self.expected_image_keys:
                # Get shape from config
                if hasattr(self.config, "input_features") and img_key in self.config.input_features:
                    img_shape = self.config.input_features[img_key].shape
                    dummy_obs[img_key] = torch.randn(1, *img_shape, device=self.device)
                    if self.verbose:
                        print(f"  Created dummy image '{img_key}': shape={dummy_obs[img_key].shape}")

        # Add depth if expected
        if hasattr(self.config, "input_features") and "observation.depth" in self.config.input_features:
            depth_shape = self.config.input_features["observation.depth"].shape
            dummy_obs["observation.depth"] = torch.randn(1, *depth_shape, device=self.device)
            if self.verbose:
                print(f"  Created dummy depth: shape={dummy_obs['observation.depth'].shape}")

        # Add task if model uses it
        # if hasattr(self.config, "use_language") and self.config.use_language:
        dummy_obs["task"] = "warmup task"
        if self.verbose:
            print(f"  Added dummy task string")

        if self.verbose:
            print("\nRunning warmup inference pass...")

        # Run inference to trigger compilation
        try:
            with torch.inference_mode():
                _ = self.predict(dummy_obs)

            if self.verbose:
                print("\n" + "=" * 60)
                print("✓ Model warmup complete! Compilation finished.")
                print("=" * 60)
                print()
        except Exception as e:
            print(f"\n✗ Warmup failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def r6_to_quat(self, delta_r6: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Convert 6D delta rotation representation to quaternion.

        This function converts a 6D rotation representation (r6) which represents
        a delta rotation, adds it to the current rotation from state, and returns
        the resulting quaternion in wxyz format.

        Args:
            delta_r6: Delta rotation in 6D representation, shape (6,) or (B, 6)
            state: Current state containing rotation in 6D, shape (10,) or (B, 10)
                   Format: [pos(3), r6(6), gripper(1)]

        Returns:
            Quaternion in wxyz format, shape (4,) or (B, 4)
        """
        if state.ndim == 1:
            state = state[None]
        if delta_r6.ndim == 1:
            delta_r6 = delta_r6[None]

        # Add delta to current rotation
        r6 = delta_r6 + state[..., 3:9]

        # Convert r6 to rotation matrix
        r1 = r6[..., :3]
        r2 = r6[..., 3:6]

        # Gram-Schmidt orthonormalization
        b1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
        b2 = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)

        # Stack to rotation matrix
        mat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)

        # Get previous rotation matrix
        prev_r1 = state[..., 3:6]
        prev_r2 = state[..., 6:9]
        prev_b1 = prev_r1 / np.linalg.norm(prev_r1, axis=-1, keepdims=True)
        prev_b2 = prev_r2 - np.sum(prev_b1 * prev_r2, axis=-1, keepdims=True) * prev_b1
        prev_b2 = prev_b2 / np.linalg.norm(prev_b2, axis=-1, keepdims=True)
        prev_b3 = np.cross(prev_b1, prev_b2)

        prev_mat = np.stack([prev_b1, prev_b2, prev_b3], axis=-1)  # (..., 3, 3)

        # Compute relative rotation
        # relative_mat = np.transpose(prev_mat, (0, 2, 1)) @ mat  # (..., 3, 3)
        relative_mat = mat @ np.transpose(prev_mat, (0, 2, 1))  # (..., 3, 3), in the world frame

        # Convert to quaternion (wxyz format)
        relative_quat = R.from_matrix(relative_mat).as_quat(scalar_first=True)  # (..., 4) wxyz

        return relative_quat.squeeze() if relative_quat.shape[0] == 1 else relative_quat
    
    def r6_absolute_to_quat(self, r6: np.ndarray) -> np.ndarray:
        """
        Convert absolute 6D rotation representation to quaternion.

        This function converts a 6D rotation representation (r6) directly to
        quaternion in wxyz format.

        Args:
            r6: Rotation in 6D representation, shape (6,) or (B, 6)

        Returns:
            Quaternion in wxyz format, shape (4,) or (B, 4)
        """
        if r6.ndim == 1:
            r6 = r6[None]

        # Convert r6 to rotation matrix
        r1 = r6[..., :3]
        r2 = r6[..., 3:6]

        # Gram-Schmidt orthonormalization
        b1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
        b2 = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)

        # Stack to rotation matrix
        mat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)

        # Convert to quaternion (wxyz format)
        quat = R.from_matrix(mat).as_quat(scalar_first=True)  # (..., 4) wxyz

        return quat.squeeze() if quat.shape[0] == 1 else quat

    def _get_expected_image_keys(self) -> list[str]:
        """Get expected image keys from model config."""
        if hasattr(self.config, "input_features"):
            # Check for observation.images.* keys
            obs_images = self.config.input_features.get("observation.images", {})
            if obs_images:
                return [f"observation.images.{k}" for k in obs_images.keys()]

            # Check for single observation.image key
            if "observation.image" in self.config.input_features:
                return ["observation.image"]

        return []

    def _transform_observation_to_lerobot_format(
        self,
        obs: dict[str, Any],
        image_size: tuple[int, int] = (224, 224)
    ) -> dict[str, Any]:
        """
        Transform raw observation to LeRobot expected format.

        Input format (from robot):
            {
                "images": {
                    "cam_high": np.array (H, W, 3),
                    "cam_left": np.array (H, W, 3),
                    ...
                },
                "depth": np.array (H, W) or (H, W, 1),  # Optional depth frame
                "qpos": np.array (20,)  # [left_arm(10), right_arm(10)]
            }

        Output format (for LeRobot):
            {
                "observation.state": torch.Tensor,
                "observation.image": torch.Tensor (3, H, W),  # if single image
                or
                "observation.images.cam_high": torch.Tensor (3, H, W),  # if multiple
                ...
                "observation.depth": torch.Tensor (1, H, W),  # if depth provided
                "task": str (optional)
            }

        Args:
            obs: Raw observation dictionary
            image_size: Target image size (height, width)

        Returns:
            Transformed observation dictionary
        """
        lerobot_obs = {}

        # Transform state (qpos)
        if "qpos" in obs:
            qpos = obs["qpos"]
            if isinstance(qpos, np.ndarray):
                lerobot_obs["observation.state"] = torch.from_numpy(qpos).float()
            else:
                lerobot_obs["observation.state"] = qpos

        # Transform images - check what keys the model expects
        if "images" in obs:
            expected_keys = self._get_expected_image_keys()

            # Helper to process image
            def process_image(img):
                if isinstance(img, np.ndarray):
                    # Resize is handled within the policy class
                    # if img.shape[:2] != image_size:
                    #     img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
                    # Convert to torch tensor (H, W, C) -> (C, H, W)
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    return torch.from_numpy(img).float().permute(2, 0, 1)
                return img

            # Check if model expects single image or multiple
            if expected_keys and "observation.image" in expected_keys:
                # Model expects single observation.image - use specified camera or first available
                if self.primary_camera and self.primary_camera in obs["images"]:
                    img = obs["images"][self.primary_camera]
                    lerobot_obs["observation.image"] = process_image(img)
                else:
                    # Use first camera
                    camera_images = list(obs["images"].items())
                    if camera_images:
                        cam_name, img = camera_images[0]
                        lerobot_obs["observation.image"] = process_image(img)
                        if self.verbose and self.primary_camera:
                            warnings.warn(
                                f"Primary camera '{self.primary_camera}' not found. "
                                f"Using '{cam_name}' instead."
                            )
            else:
                # Model expects observation.images.{cam_name} format
                for cam_name, img in obs["images"].items():
                    img_tensor = process_image(img)
                    lerobot_obs[f"observation.images.{cam_name}"] = img_tensor

        # Transform depth image if present
        if "depth" in obs:
            depth = obs["depth"]
            if isinstance(depth, np.ndarray):
                # Resize is handled within the policy class
                # if depth.shape[:2] != image_size:
                #     depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

                # Ensure depth is 2D or 3D with shape (H, W) or (H, W, 1)
                if depth.ndim == 2:
                    # Add channel dimension: (H, W) -> (1, H, W)
                    depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
                elif depth.ndim == 3 and depth.shape[2] == 1:
                    # Convert (H, W, 1) -> (1, H, W)
                    depth_tensor = torch.from_numpy(depth).float().permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected depth shape: {depth.shape}. Expected (H, W) or (H, W, 1)")

                lerobot_obs["observation.depth"] = depth_tensor
            else:
                # Already a tensor
                lerobot_obs["observation.depth"] = depth

        # Pass through task description if present
        if "task" in obs:
            lerobot_obs["task"] = obs["task"]

        return lerobot_obs

    def _transform_action_to_quat_format(
        self,
        action: np.ndarray,
        current_qpos: np.ndarray
    ) -> np.ndarray:
        """
        Transform action from r6 representation to quaternion format.
        Note that this assumes the input action is a numerical subtraction instead of a relative transform.

        Input action format (from model, denormalized):
            20D: [left_pos(3), left_r6(6), left_gripper(1),
                  right_pos(3), right_r6(6), right_gripper(1)]

        Output action format:
            16D: [left_quat_wxyz(4), left_pos(3),
                  right_quat_wxyz(4), right_pos(3),
                  left_gripper(1), right_gripper(1)]

        Args:
            action: Action in r6 format, shape (20,) or (B, 20)
            current_qpos: Current state, shape (20,) or (B, 20)

        Returns:
            Action in quaternion format, shape (16,) or (B, 16)
        """
        single_action = len(action.shape) == 1
        single_qpos = len(current_qpos.shape) == 1
        if single_action:
            action = action[None, :]
        if single_qpos:
            current_qpos = current_qpos[None, :]

        assert action.shape[0] == current_qpos.shape[0], \
            "Batch size of action and current_qpos must match."

        batch_size = action.shape[0]
        ac_quat = np.zeros((batch_size, 16), dtype=np.float32)

        # Convert left arm rotation (r6 indices 3:9) to quaternion
        if self.is_delta_action:
            left_quat = self.r6_to_quat(action[:, 3:9], current_qpos[:, :10])
        else:
            left_quat = self.r6_absolute_to_quat(action[:, 3:9])
        
        ac_quat[:, :4] = left_quat  # left quat (wxyz)

        # Left position (indices 0:3)
        ac_quat[:, 4:7] = action[:, 0:3]  # left pos

        # Convert right arm rotation (r6 indices 13:19) to quaternion
        if self.is_delta_action:
            right_quat = self.r6_to_quat(action[:, 13:19], current_qpos[:, 10:])
        else:
            right_quat = self.r6_absolute_to_quat(action[:, 13:19])

        ac_quat[:, 7:11] = right_quat  # right quat (wxyz)

        # Right position (indices 10:13)
        ac_quat[:, 11:14] = action[:, 10:13]  # right pos

        # Grippers
        ac_quat[:, 14] = action[:, 9]   # left gripper
        ac_quat[:, 15] = action[:, 19]  # right gripper

        return ac_quat.squeeze() if single_action else ac_quat
    
    def _transform_rel_transform_to_quat_format(
        self,
        action: np.ndarray,
        # current_qpos: np.ndarray
    ) -> np.ndarray:
        """
        Transform action from r6 representation to quaternion format.
        Note that this assumes the input action is a valid relative transform.

        Input action format (from model, denormalized):
            20D: [left_pos(3), left_r6(6), left_gripper(1),
                  right_pos(3), right_r6(6), right_gripper(1)]

        Output action format:
            16D: [left_quat_wxyz(4), left_pos(3),
                  right_quat_wxyz(4), right_pos(3),
                  left_gripper(1), right_gripper(1)]

        Args:
            action: Action in r6 format, shape (20,) or (B, 20)
            current_qpos: Current state, shape (20,) or (B, 20)

        Returns:
            Action in quaternion format, shape (16,) or (B, 16)
        """
        single_action = len(action.shape) == 1
        if single_action:
            action = action[None, :]

        batch_size = action.shape[0]
        ac_quat = np.zeros((batch_size, 16), dtype=np.float32)

        print(f"[DEBUG] left r6: {action[:, 3:9]}")

        left_quat = self.r6_absolute_to_quat(action[:, 3:9])
        
        ac_quat[:, :4] = left_quat  # left quat (wxyz)

        # Left position (indices 0:3)
        ac_quat[:, 4:7] = action[:, 0:3]  # left pos

        right_quat = self.r6_absolute_to_quat(action[:, 13:19])

        ac_quat[:, 7:11] = right_quat  # right quat (wxyz)

        # Right position (indices 10:13)
        ac_quat[:, 11:14] = action[:, 10:13]  # right pos

        # Grippers
        ac_quat[:, 14] = action[:, 9]   # left gripper
        ac_quat[:, 15] = action[:, 19]  # right gripper

        return ac_quat.squeeze() if single_action else ac_quat

    def forward(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Forward pass matching the sample policy interface.

        This method expects raw observation format with "images" and "qpos",
        and returns actions in quaternion format (16D).

        Input format:
            {
                "images": {
                    "cam_high": np.array (H, W, 3),
                    ...
                },
                "qpos": np.array (20,)
            }

        Output format:
            16D: [left_quat_wxyz(4), left_pos(3),
                  right_quat_wxyz(4), right_pos(3),
                  left_gripper(1), right_gripper(1)]

        Args:
            obs: Raw observation dictionary

        Returns:
            Action in quaternion format as numpy array (16,)
        """
        # Store current qpos for transformation
        current_qpos = obs["qpos"]

        # Transform observation to LeRobot format
        lerobot_obs = self._transform_observation_to_lerobot_format(obs)

        # Run inference through preprocessor -> model -> postprocessor
        processed_obs = self.preprocessor(lerobot_obs)

        with torch.inference_mode():
            action = self.policy.select_action(processed_obs)

        action = self.postprocessor({"action": action})

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, dict) and "action" in action:
            action = action["action"]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

        # Transform from r6 to quaternion format
        # action_quat = self._transform_action_to_quat_format(action, current_qpos)
        # IMPORTANT for policies that predict an actual delta transform, use the other function
        action_quat = self._transform_rel_transform_to_quat_format(action)

        # left_min_z, right_min_z = None, None

        # if self.first_inference:
        #     current_left_z = current_qpos[2]
        #     print(f"DEBUG: Current left z: {current_left_z:.4f}", flush=True)
        #     current_right_z = current_qpos[12]
        #     print(f"DEBUG: Current right z: {current_right_z:.4f}", flush=True)
        #     left_min_z, right_min_z = self.get_lowest_z_in_queue(current_left_z, current_right_z)
        # if left_min_z is not None and right_min_z is not None:
        #     print(f"DEBUG: Lowest left z in action queue after postprocessing: {left_min_z:.4f}", flush=True)
        #     print(f"DEBUG: Lowest right z in action queue after postprocessing: {right_min_z:.4f}", flush=True)

        self.first_inference = False

        return action_quat

    def predict(
        self,
        observation: dict[str, Any],
        return_dict: bool = False,
        transform_to_quat: bool = False
    ) -> np.ndarray | dict:
        """
        Generate action from observation.

        This method supports both raw format and LeRobot format observations.

        Raw format (detected by presence of "images" and "qpos"):
            {
                "images": {"cam_name": np.array, ...},
                "qpos": np.array (20,)
            }

        LeRobot format:
            {
                "observation.state": array or tensor,
                "observation.images.cam_name": array or tensor,
                "task": str (optional)
            }

        Args:
            observation: Dictionary containing observation data
            return_dict: If True, return full action dictionary instead of just action array
            transform_to_quat: If True and raw format detected, transform output to quaternion format (16D)

        Returns:
            Action as numpy array, or full action dictionary if return_dict=True
        """
        # Detect if this is raw format (has "images" and "qpos")
        is_raw_format = "images" in observation and "qpos" in observation

        if is_raw_format and transform_to_quat:
            # Use forward method for full transformation pipeline
            return self.forward(observation)

        # Transform observation if needed
        if is_raw_format:
            observation = self._transform_observation_to_lerobot_format(observation)

        # Preprocess observation
        processed_obs = self.preprocessor(observation)

        # Run inference
        with torch.inference_mode():
            action = self.policy.select_action(processed_obs)

        # Postprocess action
        action = self.postprocessor({"action": action})

        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, dict):
            if return_dict:
                # Convert all tensors in dict to numpy
                return {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in action.items()
                }
            else:
                # Extract action array
                if "action" in action:
                    action = action["action"]
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()

        # Transform to quaternion format if requested and we have current state
        if transform_to_quat:
            action = self._transform_rel_transform_to_quat_format(action)

        return action

    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        return_dict: bool = False,
        transform_to_quat: bool = False
    ) -> np.ndarray | dict:
        """
        Generate a chunk of actions from observation.

        This returns multiple timesteps of actions at once (chunk_size actions).

        Args:
            observation: Dictionary containing observation data (raw or LeRobot format)
            return_dict: If True, return full action dictionary instead of just action array
            transform_to_quat: If True and raw format detected, transform output to quaternion format

        Returns:
            Action chunk as numpy array of shape (chunk_size, action_dim),
            or full action dictionary if return_dict=True
        """
        # Detect if this is raw format (has "images" and "qpos")
        is_raw_format = "images" in observation and "qpos" in observation

        # Transform observation if needed
        if is_raw_format:
            observation = self._transform_observation_to_lerobot_format(observation)

        # Preprocess observation
        processed_obs = self.preprocessor(observation)

        # Run inference for action chunk
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(processed_obs)

        # Postprocess action
        action_chunk = self.postprocessor({"action": action_chunk})

        # Convert to numpy if needed
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        elif isinstance(action_chunk, dict):
            if return_dict:
                # Convert all tensors in dict to numpy
                return {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in action_chunk.items()
                }
            else:
                # Extract action array
                if "action" in action_chunk:
                    action_chunk = action_chunk["action"]
                    if isinstance(action_chunk, torch.Tensor):
                        action_chunk = action_chunk.cpu().numpy()

        # Transform to quaternion format if requested and we have current state
        if transform_to_quat:
            action_chunk = self._transform_rel_transform_to_quat_format(action_chunk)

        return action_chunk

    @property
    def expected_image_keys(self) -> list[str]:
        """Return expected image observation keys."""
        return self._get_expected_image_keys()

    @property
    def expected_state_dim(self) -> int:
        """Return expected state dimension."""
        if not hasattr(self.config, "input_features"):
            return 0
        state_meta = self.config.input_features.get("observation.state", None)
        if state_meta is None:
            return 0
        state_shapes = state_meta.shape
        return state_shapes[0] if isinstance(state_shapes, (list, tuple)) else state_shapes

    @property
    def action_dim(self) -> int:
        """Return action dimension."""
        if hasattr(self.config, "output_features") and "action" in self.config.output_features:
            return self.config.output_features["action"]["shape"][0]
        return 0

    @property
    def chunk_size(self) -> int:
        """Return action chunk size."""
        return getattr(self.config, "chunk_size", 1)

    def _compute_rtc_prefix(self, execution_horizon) -> tuple[torch.Tensor | None, int]:
        """
        Compute RTC prefix from previous chunk based on RTC paper formulation.

        According to the RTC paper (Fig. 1):
        - s = execution_horizon: actions executed between inferences
        - d = inference_delay: timesteps during which inference runs
        - The prefix consists of d actions from previous chunk at indices [s, s+d)
        - These are the actions executed DURING inference (RED in diagram)
        - Constraint: d ≤ H - s (so YELLOW actions still remain in buffer)

        Args:
            execution_horizon: Number of actions executed between inferences (s)

        Returns:
            Tuple of (prefix_tensor, prefix_length):
                - prefix_tensor: Tensor of shape (batch_size, prefix_len, action_dim) or None
                - prefix_length: Number of prefix actions (inference delay d)
        """
        if self.prev_chunk is None:
            return None, 0  # First chunk, no prefix

        if execution_horizon >= self.chunk_size:
            return None, 0

        # Use configured inference delay
        s = execution_horizon
        d = self.inference_delay

        # Extract prefix: actions at indices [s, s+d) from previous chunk
        # These are the actions that will be/were executed during inference
        prefix = self.prev_chunk[:, s:s+d, :]

        return prefix, d

    def predict_action_chunk_async(
        self,
        observation: dict[str, Any],
        current_time: float | None = None,
        transform_to_quat: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Generate action chunk with async RTC prefix conditioning.

        This method implements the RTC paper's async inference strategy:
        1. Computes RTC prefix: d actions from previous chunk executed during inference (RED)
        2. Generates new chunk conditioned on the prefix
        3. Returns full chunk (H actions) to send to robot
        4. Robot-side manages buffer overlap when new chunk arrives

        Key parameters:
        - s (execution_horizon): Action steps between inferences
        - d (prefix_length): Inference delay in action steps
        - H (chunk_size): Total prediction horizon in action steps

        Timeline:
        - Robot executes first s actions before next inference starts
        - During inference, robot executes next d actions (RED prefix)
        - Robot still has (H - s - d) actions in buffer (YELLOW) when new chunk arrives

        Args:
            observation: Dictionary containing observation data (raw or LeRobot format)
            current_time: Current timestamp in seconds. If None, uses time.time()
            transform_to_quat: If True and raw format detected, transform output to quaternion format

        Returns:
            Tuple of (actions_to_execute, metadata):
                - actions_to_execute: Full action chunk to send to robot, shape (H, action_dim)
                - metadata: Dict with 'chunk_idx', 'send_time', 'prefix_length', etc.
        """
        import time as time_module

        if self.prev_chunk is None:
            # First chunk, no prefix
            execution_horizon = 0
        else:
            execution_horizon = self.chunk_size

        # Detect if this is raw format (has "images" and "qpos")
        is_raw_format = "images" in observation and "qpos" in observation

        # Transform observation if needed
        if is_raw_format:
            observation = self._transform_observation_to_lerobot_format(observation)

        # Preprocess observation
        processed_obs = self.preprocessor(observation)

        # Compute RTC prefix from previous chunk
        # Prefix = d actions at indices [s, s+d) from previous chunk
        prefix, prefix_length = self._compute_rtc_prefix(execution_horizon)

        # Move prefix to device if it exists
        if prefix is not None:
            prefix = prefix.to(self.device)

        # Run inference with RTC conditioning
        # The policy's sample_actions method should accept prev_chunk_left_over and inference_delay
        with torch.inference_mode():
            if prefix is not None and prefix_length > 0:
                # Call with RTC parameters
                raw_chunk = self.policy.predict_action_chunk(
                    processed_obs,
                    prev_chunk_left_over=prefix,
                    inference_delay=prefix_length
                )
            else:
                # No prefix, standard inference
                raw_chunk = self.policy.predict_action_chunk(processed_obs)

        # Store raw chunk for next RTC prefix computation (before postprocessing)
        # Store as tensor on device (batch_size, chunk_size, action_dim)
        if isinstance(raw_chunk, torch.Tensor):
            self.prev_chunk = raw_chunk.detach()
        else:
            self.prev_chunk = torch.from_numpy(raw_chunk).to(self.device)

        # Ensure batch dimension
        if self.prev_chunk.ndim == 2:
            self.prev_chunk = self.prev_chunk.unsqueeze(0)

        # Postprocess action chunk for output
        action_chunk = self.postprocessor({"action": raw_chunk})

        # Convert to numpy if needed
        if isinstance(action_chunk, torch.Tensor):
            action_chunk_np = action_chunk.cpu().numpy()
        elif isinstance(action_chunk, dict):
            if "action" in action_chunk:
                action_chunk_tensor = action_chunk["action"]
                if isinstance(action_chunk_tensor, torch.Tensor):
                    action_chunk_np = action_chunk_tensor.cpu().numpy()
                else:
                    action_chunk_np = action_chunk_tensor
            else:
                raise ValueError("Postprocessor output dict does not contain 'action' key")
        else:
            action_chunk_np = action_chunk

        if current_time is None:
            current_time = time_module.time()
        self.prev_chunk_send_time = current_time
        self.chunk_idx += 1

        # Send entire chunk (or at least s + d actions) to robot
        # The robot will execute s actions, then d actions (prefix) during next inference,
        # and still have (H - s - d) actions in buffer when new chunk arrives.
        # Robot-side buffer management will handle overlapping actions when new chunk arrives.
        actions_to_execute = action_chunk_np

        # Squeeze batch dimension if present
        if actions_to_execute.ndim == 3 and actions_to_execute.shape[0] == 1:
            actions_to_execute = actions_to_execute.squeeze(0)

        # Transform to quaternion format if requested
        if transform_to_quat:
            actions_to_execute = self._transform_rel_transform_to_quat_format(actions_to_execute)

        # Create metadata
        metadata = {
            'chunk_idx': self.chunk_idx,
            'send_time': current_time,
            'prefix_length': prefix_length,  # d: inference delay in action steps
            'execution_horizon': execution_horizon,  # s: action steps between inferences
            'full_chunk_size': self.chunk_size,  # H: total prediction horizon in action steps
            'actions_sent': len(actions_to_execute),  # Number of actions sent to robot
        }

        return actions_to_execute, metadata
    
    def reset_action_queue(self):
        """Reset the action queue in the policy model."""
        if hasattr(self.policy, "_action_queue") and isinstance(self.policy._action_queue, deque):
            self.policy._action_queue.clear()
        else:
            self.policy._action_queue = deque(maxlen=self.chunk_size)

    # testing only
    def get_lowest_z_in_queue(self, current_left_z, current_right_z) -> float | None:
        """Get the lowest z value from the action queue for testing purposes."""
        if hasattr(self.policy, "_action_queue") and isinstance(self.policy._action_queue, deque):
            if not self.policy._action_queue:
                return None, None
            postprocessed_actions = self.postprocessor(
                {"action": torch.stack(list(self.policy._action_queue))}
            )["action"]
            postprocessed_action_list = postprocessed_actions.cpu().numpy().tolist()
            left_delta_z_values = [action[0][2] for action in postprocessed_action_list]
            right_delta_z_values = [action[0][12] for action in postprocessed_action_list]
            # cumsum to get absolute z positions
            left_z_positions = np.cumsum(left_delta_z_values) + current_left_z
            right_z_positions = np.cumsum(right_delta_z_values) + current_right_z
            lowest_left_z = np.min(left_z_positions)
            lowest_right_z = np.min(right_z_positions)
            # save as fig
            import matplotlib.pyplot as plt

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot left z trajectory
            ax1.plot(left_z_positions, marker='o', linestyle='-', linewidth=2, markersize=4)
            ax1.axhline(y=current_left_z, color='r', linestyle='--', label='Current Z')
            ax1.axhline(y=lowest_left_z, color='g', linestyle='--', label='Lowest Z')
            ax1.set_xlabel('Action Step')
            ax1.set_ylabel('Z Position')
            ax1.set_title('Left Arm Z Trajectory')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot right z trajectory
            ax2.plot(right_z_positions, marker='o', linestyle='-', linewidth=2, markersize=4)
            ax2.axhline(y=current_right_z, color='r', linestyle='--', label='Current Z')
            ax2.axhline(y=lowest_right_z, color='g', linestyle='--', label='Lowest Z')
            ax2.set_xlabel('Action Step')
            ax2.set_ylabel('Z Position')
            ax2.set_title('Right Arm Z Trajectory')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig('z_trajectories.png', dpi=150, bbox_inches='tight')
            plt.close()
            return lowest_left_z, lowest_right_z

        return None, None


def main():
    """Example usage and testing of Pi05InferencePolicy."""
    import argparse

    parser = argparse.ArgumentParser(description="Pi0.5 Inference Example")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test inference with dummy data",
    )
    args = parser.parse_args()

    # Load policy
    print("=" * 60)
    print("Loading Pi0.5 Policy")
    print("=" * 60)

    policy = Pi05InferencePolicy(args.checkpoint, device=args.device)

    print(f"\nModel Properties:")
    print(f"  Expected image keys: {policy.expected_image_keys}")
    print(f"  Expected state dim: {policy.expected_state_dim}")
    print(f"  Action dim: {policy.action_dim}")
    print(f"  Chunk size: {policy.chunk_size}")

    # Run warmup to trigger torch.compile
    policy.warmup()

    if args.test:
        print("\n" + "=" * 60)
        print("Running Test Inference")
        print("=" * 60)

        # Create dummy observation for testing
        dummy_obs = {
            "task": "fold the cloth",
        }

        # Add dummy state if expected
        if policy.expected_state_dim > 0:
            dummy_obs["observation.state"] = torch.randn(policy.expected_state_dim)
            print(f"Created dummy state with shape: {dummy_obs['observation.state'].shape}")

        # Add dummy images for each expected camera
        if policy.expected_image_keys:
            for img_key in policy.expected_image_keys:
                # Assuming 224x224 RGB images (Pi0.5 default)
                dummy_obs[img_key] = torch.randn(3, 224, 224)
                print(f"Created dummy image '{img_key}' with shape: {dummy_obs[img_key].shape}")

        print("\nRunning single action prediction...")
        try:
            action = policy.predict(dummy_obs)
            print(f"✓ Predicted action shape: {action.shape}")
            print(f"  Action preview: {action}")
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()

        print("\nRunning action chunk prediction...")
        try:
            action_chunk = policy.predict_action_chunk(dummy_obs)
            print(f"✓ Predicted action chunk shape: {action_chunk.shape}")
            print(f"  Action chunk preview (first 3): {action_chunk[:3]}")
        except Exception as e:
            print(f"✗ Chunk prediction failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()