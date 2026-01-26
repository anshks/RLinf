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

"""WorldGym dataset for loading PNG initial frames and instructions.

This dataset is designed for world model-based RL training using PNG initial frames
from the WorldGym benchmark, adapted from SimpleVLA-RL's implementation.
"""

import numpy as np
import torch
from typing import List, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from rlinf.utils.worldmodel import discover_trials, load_png_to_tensor


class WorldGymDataset(torch.utils.data.Dataset):
    """Dataset for WorldGym PNG initial frames and task instructions.

    This dataset discovers and loads WorldGym trials (PNG files with JSON metadata)
    and provides them in the format expected by RLinf's world model infrastructure.

    For cloth folding dataset: uses PNG files only with fixed instruction.

    Args:
        data_dir: Directory containing WorldGym trials (PNG files + JSON metadata or PNG only)
        camera_names: List of camera names (e.g., ["image"])
        action_dim: Action dimensionality (e.g., 20 for Pi0.5 cloth folding)
        task_filter: Optional list of task names to filter (default: None = use all)
        use_cloth_fold: If True, uses cloth folding mode (PNG only, fixed instruction)
        cloth_fold_instruction: Instruction to use for cloth folding (default: "Fold the shirt on the table")
    """

    def __init__(
        self,
        data_dir: str,
        camera_names: List[str],
        action_dim: int,
        task_filter: Optional[List[str]] = None,
        use_cloth_fold: bool = False,
        cloth_fold_instruction: str = "Fold the shirt on the table",
    ):
        """Initialize WorldGym dataset.

        Args:
            data_dir: Path to directory containing trial PNG files (and optionally JSON metadata)
            camera_names: List of camera view names
            action_dim: Dimension of action space
            task_filter: Optional list of task names to include (None = all tasks)
            use_cloth_fold: If True, load PNG files directly without JSON metadata
            cloth_fold_instruction: Fixed instruction for cloth folding tasks
        """
        self.data_dir = data_dir
        self.camera_names = camera_names
        self.action_dim = action_dim
        self.task_filter = task_filter
        self.use_cloth_fold = use_cloth_fold
        self.cloth_fold_instruction = cloth_fold_instruction

        if self.use_cloth_fold:
            # Cloth folding mode: load PNG files directly
            self._load_cloth_fold_trials()
        else:
            # Standard WorldGym mode: use discover_trials for PNG+JSON
            self.trials = discover_trials(data_dir)

            # Filter by task if specified
            if task_filter is not None:
                self.trials = [
                    trial for trial in self.trials
                    if trial["task_key"] in task_filter
                ]

            print(f"[WorldGymDataset] Found {len(self.trials)} trials in {data_dir}")
            if task_filter:
                print(f"[WorldGymDataset] Filtered to tasks: {task_filter}")

            # Group trials by instruction for consistent batching
            self._group_by_instruction()

    def _load_cloth_fold_trials(self):
        """Load cloth folding trials (PNG files only, no JSON)."""
        data_path = Path(self.data_dir)

        # Find all PNG files
        png_files = sorted(list(data_path.glob("*.png")))

        if len(png_files) == 0:
            raise ValueError(f"No PNG files found in {self.data_dir}")

        # Create trial entries with fixed instruction
        self.trials = []
        for png_file in png_files:
            trial = {
                "trial_png": str(png_file),
                "instruction": self.cloth_fold_instruction,
                "task_key": "cloth_fold",
                "task_display": "Cloth Fold",
            }
            self.trials.append(trial)

        print(f"[WorldGymDataset] Cloth fold mode: Found {len(self.trials)} PNG files in {data_path}")
        print(f"[WorldGymDataset] Using instruction: '{self.cloth_fold_instruction}'")

        # All trials have the same instruction, so grouping is simple
        self.sorted_instructions = [self.cloth_fold_instruction]
        self.instruction_groups = {self.cloth_fold_instruction: self.trials}

    def _group_by_instruction(self):
        """Group trials by instruction (same as SimpleVLA-RL).

        This ensures batches have the same instruction, avoiding padding issues.
        """
        instruction_groups = {}
        for trial in self.trials:
            instruction = trial["instruction"]
            if instruction not in instruction_groups:
                instruction_groups[instruction] = []
            instruction_groups[instruction].append(trial)

        # Sort instructions for deterministic ordering
        self.sorted_instructions = sorted(instruction_groups.keys())
        self.instruction_groups = instruction_groups

        # print(f"[WorldGymDataset] Grouped into {len(self.sorted_instructions)} unique instructions")
        for i, instruction in enumerate(self.sorted_instructions[:5]):  # Show first 5
            count = len(instruction_groups[instruction])
            print(f"  [{i}] '{instruction[:60]}...' ({count} trials)")
        if len(self.sorted_instructions) > 5:
            print(f"  ... and {len(self.sorted_instructions) - 5} more instructions")

    def __len__(self) -> int:
        """Return total number of trials."""
        return len(self.trials)

    def _create_default_initial_state(self) -> np.ndarray:
        """Create a default initial state for dual-arm robot.

        For Pi0.5 with dual-arm robots, the state is:
        [left_pos(3), left_r6(6), left_gripper(1), right_pos(3), right_r6(6), right_gripper(1)] = 20D

        This provides a reasonable default starting pose for cloth folding tasks.

        Returns:
            Default initial state as numpy array [action_dim]
        """
        state = np.zeros(self.action_dim, dtype=np.float32)

        # Default left arm pose (from Pi0.5 cloth folding setup)
        left_pos = np.array([0.35646688, 0.0382356, 0.92677665])
        left_quat_wxyz = np.array([-0.08122765, 0.70717267, 0.69836195, -0.07482959])
        left_quat_xyzw = np.array([left_quat_wxyz[1], left_quat_wxyz[2], left_quat_wxyz[3], left_quat_wxyz[0]])
        left_rot = R.from_quat(left_quat_xyzw)
        left_rot_matrix = left_rot.as_matrix()
        left_r6 = left_rot_matrix[:, :2].T.flatten()

        state[0:3] = left_pos       # Left EE position
        state[3:9] = left_r6        # Left EE rotation (r6 representation)
        state[9] = 1.0              # Left gripper open

        # Default right arm pose (from Pi0.5 cloth folding setup)
        right_pos = np.array([0.35906339, -0.45805741, 0.92072002])
        right_quat_xyzw = np.array([0.70221996, 0.70186009, -0.07412596, -0.09372766])
        right_rot = R.from_quat(right_quat_xyzw)
        right_rot_matrix = right_rot.as_matrix()
        right_r6 = right_rot_matrix[:, :2].T.flatten()

        state[10:13] = right_pos    # Right EE position
        state[13:19] = right_r6     # Right EE rotation (r6 representation)
        state[19] = 1.0             # Right gripper open

        return state

    def __getitem__(self, idx: int) -> dict:
        """Get a single trial by index.

        Args:
            idx: Index of the trial

        Returns:
            Dictionary containing:
                - start_items: List of initial frame dicts (camera images + state + task)
                - task: Task instruction string
                - trial_png: Path to PNG file
                - instruction: Task instruction (same as task)
                - task_key: Task identifier
        """
        trial = self.trials[idx]

        # Load initial PNG frame
        png_path = trial["trial_png"]
        frame_tensor = load_png_to_tensor(png_path, target_size=256)  # (H, W, C) [0, 1]

        # Convert to uint8 [0, 255]
        frame_uint8 = (frame_tensor * 255).byte()

        # Create start item dict matching RLinf format
        start_item = {}
        for camera_name in self.camera_names:
            start_item[camera_name] = frame_uint8

        # Add initial robot state (dual-arm default pose for cloth folding)
        initial_state = self._create_default_initial_state()
        start_item["observation.state"] = torch.from_numpy(initial_state)

        # # Debug: Print initial state for first few trials
        # if idx < 3:  # Only print for first 3 trials to avoid spam
        #     print(f"[WorldGymDataset] Trial {idx}: Created initial state")
        #     print(f"  Left arm pos: {initial_state[0:3]}")
        #     print(f"  Left gripper: {initial_state[9]:.2f}")
        #     print(f"  Right arm pos: {initial_state[10:13]}")
        #     print(f"  Right gripper: {initial_state[19]:.2f}")
        #     print(f"  All zeros? {np.allclose(initial_state, 0.0)}")

        # Add task instruction
        start_item["task"] = trial["instruction"]

        # Return in format expected by BaseFakeModelInference
        return {
            "start_items": [start_item],  # List with single initial frame
            "task": trial["instruction"],
            "trial_png": trial["trial_png"],
            "instruction": trial["instruction"],
            "task_key": trial["task_key"],
            "initial_state": initial_state,  # Add initial state to episode dict
        }
