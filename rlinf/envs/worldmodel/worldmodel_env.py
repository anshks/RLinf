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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import gym
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.utils import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from rlinf.envs.worldmodel.dataset import LeRobotDatasetWrapper
from rlinf.envs.worldgym.dataset import WorldGymDataset
from rlinf.models.worldmodel.base_fake_model import BaseFakeModelInference
from rlinf.models.worldmodel.dit_world_model import DiTWorldModelInference
from rlinf.utils.worldmodel import predict


class WorldModelEnv(gym.Env):
    """A Gym environment that wraps a world model for reinforcement learning.

    This environment provides a Gym-compatible interface for interacting with a learned
    world model that simulates the dynamics of a real environment. It handles environment
    resets, step execution, reward calculation, and metrics tracking while supporting
    both single-step and chunked action execution.

    Key features:
    - Supports fixed or random reset state selection
    - Handles relative or absolute reward computation
    - Provides video recording capabilities
    - Tracks episode success/failure metrics
    - Supports chunked action execution for efficiency
    - Auto-reset functionality for continuous episodes

    Args:
        cfg: The configuration object containing environment, dataset, and video settings.
        seed_offset (int): An offset added to the base seed for creating different
            environment instances with reproducible randomness.
        total_num_processes (int): The total number of parallel processes for distributed training.
        record_metrics (bool, optional): Whether to track and log episode metrics like
            success rates and returns. Defaults to True.
    """

    def __init__(
        self, cfg, num_envs, seed_offset: int, total_num_processes, record_metrics=True
    ):
        """Initializes the WorldModelEnv with configuration and setup parameters.

        This method sets up the world model environment by:
        - Initializing configuration parameters from the config object
        - Setting up the dataset wrapper for state management
        - Creating the world model backend for simulation
        - Initializing metrics tracking and video recording capabilities
        - Setting up reset state ID generation for episode initialization

        Args:
            cfg: The configuration object containing all environment settings including
                dataset configuration, backend settings, and video recording options.
            seed_offset (int): An offset added to the base seed to ensure different
                environment instances have different random seeds for reproducibility.
            total_num_processes (int): The total number of parallel processes for
                distributed training scenarios.
            record_metrics (bool, optional): Whether to track episode-level metrics
                such as success rates, failure rates, and returns. Defaults to True.
        """

        self.cfg = cfg

        # Load basic configuration information
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = num_envs
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.gen_num_image_each_step = cfg.backend_cfg.gen_num_image_each_step
        print(f"[WorldModelEnv] Initialized with gen_num_image_each_step={self.gen_num_image_each_step}")

        # Determine dataset type based on task_suite_name or env_type
        dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg, resolve=True)
        task_suite_name = cfg.get("task_suite_name", "")

        if "worldgym" in task_suite_name or cfg.env_type == "worldgym":
            # Use WorldGym dataset for world model-based training
            self.task_dataset = WorldGymDataset(**dataset_cfg)
            self.is_worldgym = True
        else:
            # Use LeRobot dataset wrapper for standard datasets
            self.task_dataset = LeRobotDatasetWrapper(**dataset_cfg)
            self.is_worldgym = False

        self.total_num_group_envs = len(self.task_dataset)
        self.camera_names = self.task_dataset.camera_names

        self.device = "cuda"
        env_cfg = OmegaConf.to_container(cfg.backend_cfg, resolve=True)

        # Use DiT backend for WorldGym, otherwise use base backend
        if self.is_worldgym:
            self.env = DiTWorldModelInference(env_cfg, self.task_dataset, self.device)
        else:
            self.env = BaseFakeModelInference(env_cfg, self.task_dataset, self.device)

        self._is_start = True
        self._init_reset_state_ids()
        self._init_metrics()
        self.record_metrics = record_metrics
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._rollout_horizon = int(
            cfg.get("max_steps_per_rollout_epoch", cfg.max_episode_steps)
        )
        self._gpt_enabled = self.is_worldgym and bool(
            getattr(cfg, "use_gpt_rewards", False)
        )
        self._gpt_parallel_workers = max(
            1, int(getattr(cfg, "gpt_parallel_workers", 1) or 1)
        )
        self._gpt_votes = int(getattr(cfg, "gpt_votes", 5) or 5)
        self._episode_frames = [[] for _ in range(self.num_envs)]
        self._episode_instructions = [None for _ in range(self.num_envs)]
        self._gpt_evaluated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = {camera_name: [] for camera_name in self.camera_names}
        if self._gpt_enabled and predict is None:
            raise RuntimeError(
                "GPT rewards enabled, but world_model_eval is unavailable."
            )
        if self._gpt_enabled and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("GPT rewards enabled, but OPENAI_API_KEY is not set.")

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _init_reset_state_ids(self):
        """Initializes the random generator for reset state ID selection."""
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _init_metrics(self):
        """Initializes episode tracking metrics for all environments."""
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_gpt_buffers(self, env_idx=None):
        if not self._gpt_enabled:
            return
        if env_idx is None:
            self._episode_frames = [[] for _ in range(self.num_envs)]
            self._episode_instructions = [None for _ in range(self.num_envs)]
            self._gpt_evaluated[:] = False
            return
        env_indices = (
            env_idx.tolist()
            if torch.is_tensor(env_idx)
            else list(env_idx)
        )
        for idx in env_indices:
            self._episode_frames[idx] = []
            self._episode_instructions[idx] = None
        self._gpt_evaluated[env_indices] = False

    def _update_episode_instructions(self, options=None):
        if not self._gpt_enabled:
            return
        if options and "env_idx" in options:
            env_idx = options["env_idx"]
            env_indices = (
                env_idx.tolist()
                if torch.is_tensor(env_idx)
                else list(env_idx)
            )
            for idx in env_indices:
                self._episode_instructions[idx] = self.env.episodes[idx].get("task")
        else:
            self._episode_instructions = [
                episode.get("task") for episode in self.env.episodes
            ]

    def _append_gpt_frames(self, obs, env_idx=None):
        if not self._gpt_enabled or obs is None:
            return
        obs_list = obs if isinstance(obs, list) else [obs]
        if not self.camera_names:
            return
        camera_name = self.camera_names[0]
        env_indices = None
        if env_idx is not None:
            env_indices = (
                env_idx.tolist()
                if torch.is_tensor(env_idx)
                else list(env_idx)
            )
        for frame_obs in obs_list:
            if not isinstance(frame_obs, dict):
                continue
            images_and_states = frame_obs.get("images_and_states")
            if not images_and_states or camera_name not in images_and_states:
                continue
            frame_batch = images_and_states[camera_name]
            for env_id in (
                env_indices if env_indices is not None else range(self.num_envs)
            ):
                frame = frame_batch[env_id]
                frame_np = common.to_numpy(frame).copy()
                if frame_np.dtype != np.uint8:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                self._episode_frames[env_id].append(frame_np)

    def _compute_gpt_rewards(self, pending_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._gpt_enabled:
            print("[WorldModelEnv] GPT reward computation skipped: GPT not enabled", flush=True)
            return scores
        if predict is None:
            raise RuntimeError("GPT rewards enabled, but world_model_eval is unavailable.")
        env_indices = pending_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if not env_indices:
            # print("[WorldModelEnv] No pending envs for GPT evaluation (all cached?)", flush=True)
            return scores

        frame_counts = [len(self._episode_frames[idx]) for idx in env_indices]
        print(
            f"[WorldModelEnv] GPT scoring envs={env_indices} frames={frame_counts}",
            flush=True,
        )

        max_workers = min(self._gpt_parallel_workers, len(env_indices))

        def _evaluate_single(idx: int) -> float:
            frames = self._episode_frames[idx]
            if not frames:
                print(
                    f"[WorldModelEnv] GPT skip env {idx}: no frames",
                    flush=True,
                )
                return 0.0
            video = np.stack(frames, axis=0)
            if video.dtype != np.uint8:
                video = np.clip(video * 255, 0, 255).astype(np.uint8)
            trial = {
                "instruction": self._episode_instructions[idx] or "",
                "partial_criteria": None,
            }
            # print(f"[WorldModelEnv] Calling GPT for env {idx} with {self._gpt_votes} votes", flush=True)
            score = float(predict(video, trial, n=self._gpt_votes))
            # print(f"[WorldModelEnv] GPT env {idx} voting result: score={score}", flush=True)
            return score

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_single, idx): idx for idx in env_indices
            }
            for future in as_completed(futures):
                idx = futures[future]
                scores[idx] = future.result()
                self._episode_frames[idx] = []

        print(f"[WorldModelEnv] GPT scores (raw, before coef): {[scores[idx].item() for idx in env_indices]}", flush=True)
        return scores

    def _select_latest_obs(self, obs):
        """Selects the latest observation from a list or returns the observation directly."""
        if isinstance(obs, list):
            assert len(obs) >= 1, "obs length must bigger than 0."
            obs = obs[-1]
        return self._convert_obs(obs)

    def _convert_obs(self, obs):
        """Convert world model observations to the expected env observation format."""
        if obs is None:
            return obs
        if not isinstance(obs, dict):
            return obs
        if "images_and_states" not in obs:
            return obs

        images_and_states = obs["images_and_states"]
        main_images = None
        if self.camera_names:
            main_images = images_and_states.get(self.camera_names[0])
        if main_images is None:
            for key, value in images_and_states.items():
                if key != "state":
                    main_images = value
                    break

        return {
            "main_images": main_images,
            "states": images_and_states.get("state"),
            "task_descriptions": obs.get("task_descriptions"),
        }

    def _reset_metrics(self, env_idx=None):
        """Resets episode metrics for specified environments or all environments.

        Args:
            env_idx: Optional tensor or list of environment indices to reset.
                If None, resets metrics for all environments.
        """

        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self._elapsed_steps[mask] = 0
            self._reset_gpt_buffers(env_idx)
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self._elapsed_steps[:] = 0
            self._reset_gpt_buffers()
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_rewards, infos):
        """Records episode metrics including success, failure, and return information.

        Args:
            step_rewards: Tensor of rewards for the current step
            infos: List of info dictionaries for each environment

        Returns:
            list: Updated info dictionaries with episode metrics added
        """

        for i, step_reward in enumerate(step_rewards):
            episode_info = {}
            self.returns += step_reward
            if "success" in infos[i]:
                self.success_once = self.success_once | infos[i]["success"]
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos[i]:
                self.fail_once = self.fail_once | infos[i]["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            infos[i]["episode"] = episode_info
        return infos

    def _calc_step_reward(self, rewards, terminations):
        """(Initial implementation) Calculates step rewards based on termination states and configuration.

        This method computes rewards for each environment based on termination signals
        and the configured reward calculation mode. It supports both relative rewards
        (difference from previous step) and absolute rewards.

        Args:
            rewards: Raw reward values from the world model backend
            terminations: Boolean tensor indicating which environments have terminated

        Returns:
            list: A list of calculated rewards for each environment, where each reward
                is either the absolute reward or the difference from the previous step
                based on the use_rel_reward configuration.
        """

        terminations_list = (
            terminations
            if isinstance(terminations, (list, tuple))
            else [terminations]
        )
        return_rewards = []
        final_mask = self._elapsed_steps >= self._rollout_horizon
        pending_mask = final_mask & ~self._gpt_evaluated
        gpt_scores = None
        if self._gpt_enabled and pending_mask.any():
            print(f"[WorldModelEnv] Evaluating GPT rewards for episodes: {pending_mask.nonzero(as_tuple=False).squeeze(-1).tolist()}", flush=True)
            gpt_scores = self._compute_gpt_rewards(pending_mask)
            print(f"[WorldModelEnv] Raw GPT scores (before reward_coef): {gpt_scores.tolist()}", flush=True)
            self._gpt_evaluated[pending_mask] = True

        last_idx = len(terminations_list) - 1
        for i, termination in enumerate(terminations_list):
            reward = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            if self._gpt_enabled:
                if gpt_scores is not None and i == last_idx:
                    reward = gpt_scores
            # Print before applying reward_coef
            if self._gpt_enabled and gpt_scores is not None and i == last_idx:
                print(f"[WorldModelEnv] (Step {i}) Raw GPT scores before coef: {reward.tolist()}", flush=True)
            reward = reward * float(self.cfg.reward_coef)
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward

            if self.use_rel_reward:
                return_rewards.append(reward_diff)
            else:
                return_rewards.append(reward)
        return return_rewards

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        return_converted: bool = True,
    ):
        """Resets the environment to initial states and returns observations.

        Args:
            seed: Optional seed or list of seeds for deterministic resets.
                If provided, ensures reproducible environment initialization.
            options: Optional dictionary containing reset options.
                Can include 'env_idx' to reset specific environments,
                or 'episode_id' for fixed reset state selection.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observations from the reset environment
                - info: Additional information about the reset state
        """

        obs, info = self.env.reset(seed=seed, options=options)
        if self._is_start:
            self._is_start = False
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            env_idx = None
            self._reset_metrics()
        if self._gpt_enabled:
            self._update_episode_instructions(options)
            self._append_gpt_frames(obs, env_idx=env_idx)
        latest_obs = obs[-1] if isinstance(obs, list) else obs
        if return_converted:
            return self._convert_obs(latest_obs), info
        return latest_obs, info

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        """Executes one environment step with the given actions.

        Args:
            actions: The actions to execute in the environment. Can be None only
                for the first step after reset, where a reset operation is performed.
            auto_reset: Whether to automatically reset environments that reach
                termination or truncation states. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - obs: The observations after executing the actions
                - rewards: The calculated rewards for this step
                - terminations: Boolean tensor indicating episode terminations
                - truncations: Boolean tensor indicating episode truncations
                - info: Dictionary containing additional information including
                    episode metrics and final observations for auto-reset episodes

        Note:
            For the initial step (is_start=True), this method performs a reset
            operation and returns zero rewards with no terminations/truncations.
        """

        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            extracted_obs, infos = self.reset(
                seed=self.seed,
                options={"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {},
                return_converted=False,
            )
            self._is_start = False
            terminations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            truncations = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if self.video_cfg.save_video:
                self.add_new_frames(extracted_obs, infos)
            return (
                self._select_latest_obs(extracted_obs),
                torch.zeros(self.num_envs, dtype=torch.float32).to(self.device),
                terminations,
                truncations,
                infos,
            )

        if actions is not None and actions.ndim == 3:
            # World model expects [batch, action_dim]; chunk_step slices add a singleton.
            actions = actions[:, 0, :]

        new_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._elapsed_steps += 1
        if self._gpt_enabled:
            self._append_gpt_frames(new_obs)

        step_rewards = self._calc_step_reward(rewards, terminations)
        infos = self._record_metrics(step_rewards, infos)

        if self.ignore_terminations:
            for termination, info in zip(terminations, infos):
                termination[:] = False
                if self.record_metrics:
                    if "success" in info:
                        info["episode"]["success_at_end"] = info["success"].clone()
                    if "fail" in info:
                        info["episode"]["fail_at_end"] = info["fail"].clone()

        # only consider the last step output
        dones = torch.logical_or(terminations[-1], truncations[-1])
        _auto_reset = auto_reset and self.auto_reset

        if dones.any() and _auto_reset:
            new_obs[-1], infos[-1] = self._handle_auto_reset(
                dones, new_obs[-1], infos[-1]
            )

        if self.video_cfg.save_video:
            self.add_new_frames(new_obs, infos)

        return (
            self._select_latest_obs(new_obs),
            step_rewards,
            terminations,
            truncations,
            infos[-1],
        )

    def chunk_step(self, chunk_actions):
        """Executes multiple environment steps with chunked actions for efficiency.

        Args:
            chunk_actions: A tensor of shape [num_envs, chunk_step, action_dim] containing
                the sequence of actions to execute for each environment.

        Returns:
            tuple: A tuple containing:
                - extracted_obs: The observations after executing all chunked actions
                - chunk_rewards: Tensor of rewards for each step in the chunk [num_envs, chunk_steps]
                - chunk_terminations: Boolean tensor indicating terminations [num_envs, chunk_steps]
                - chunk_truncations: Boolean tensor indicating truncations [num_envs, chunk_steps]
                - info: Dictionary containing information about the final state

        Note:
            The chunk_step must be divisible by gen_num_image_each_step for proper processing.
            Terminations and truncations are only marked at the end of the chunk sequence.
        """

        chunk_step = chunk_actions.shape[1]
        print(f"[WorldModelEnv] chunk_step called with shape={chunk_actions.shape}, gen_num_image_each_step={self.gen_num_image_each_step}, computed chunk_size={chunk_step // self.gen_num_image_each_step} from chunk_step: {chunk_step}")
        assert chunk_step % self.gen_num_image_each_step == 0, (
            "chunk_step must be divisible by gen_num_image_each_step"
        )
        chunk_size = chunk_step // self.gen_num_image_each_step
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        
        # Accumulate actions for state update after chunk completes
        accumulated_actions = []
        
        for i in range(chunk_size):
            actions = chunk_actions[
                :,
                i * self.gen_num_image_each_step : (i + 1)
                * self.gen_num_image_each_step,
                :,
            ]
            # Accumulate actions for state update (squeeze time dimension for state update)
            accumulated_actions.append(actions[:, 0, :] if actions.ndim == 3 else actions)
            
            extracted_obs, step_rewards, terminations, truncations, info = self.step(
                actions, auto_reset=False
            )
            if not isinstance(step_rewards, (list, tuple)):
                step_rewards = [step_rewards]
            if not isinstance(terminations, (list, tuple)):
                terminations = [terminations]
            if not isinstance(truncations, (list, tuple)):
                truncations = [truncations]
            step_rewards = [
                reward if torch.is_tensor(reward) else torch.as_tensor(reward)
                for reward in step_rewards
            ]
            terminations = [
                termination
                if torch.is_tensor(termination)
                else torch.as_tensor(termination)
                for termination in terminations
            ]
            truncations = [
                truncation
                if torch.is_tensor(truncation)
                else torch.as_tensor(truncation)
                for truncation in truncations
            ]

            chunk_rewards.extend(step_rewards)
            raw_chunk_terminations.extend(terminations)
            raw_chunk_truncations.extend(truncations)

        # Update states with all actions in the chunk (after all frames generated)
        # This matches pi05_rollout_rtc.py pattern: generate all frames first, then update state
        if self.is_worldgym and hasattr(self.env, '_update_states_with_actions'):
            for action in accumulated_actions:
                self.env._update_states_with_actions(action)
            print(f"[WorldModelEnv] Updated states for {len(accumulated_actions)} actions in chunk")

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, info = self._handle_auto_reset(
                past_dones, extracted_obs, info
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            info,
        )

    def _handle_auto_reset(self, dones, extracted_obs, info):
        """Handles automatic reset of environments that have reached terminal states.

        Args:
            dones: Boolean tensor indicating which environments need reset
            extracted_obs: Current observations before reset
            info: Current information dictionary before reset

        Returns:
            tuple: Updated observations and info after reset, with final state information preserved
        """

        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(info)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, info = self.reset(options=options)
        info["final_observation"] = final_obs
        info["final_info"] = final_info
        info["_final_info"] = dones
        info["_final_observation"] = dones
        info["_elapsed_steps"] = dones
        return extracted_obs, info

    def run(self):
        """Runs a test execution of the environment with random actions.

        Note:
            This is primarily used for testing and debugging the environment setup.
        """

        obs, _ = self.reset()
        self.reset(
            options={"env_idx": torch.arange(0, self.num_envs - 4, device=self.device)}
        )
        self.step()
        for step in range(10):
            base = (
                torch.tensor(self.env.action_space.sample(), dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            actions = base.unsqueeze(1).repeat(1, self.gen_num_image_each_step * 2, 1)
            obs, reward, terminations, truncations, info = self.chunk_step(
                actions.cpu().numpy()
            )
            print(
                f"Step {step}: obs={obs.keys()}, reward={reward.mean()}, \
terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )
        self.flush_video()

    def add_new_frames(self, extracted_obs, infos):
        """Adds new frames to the video recording buffer.

        Args:
            extracted_obs: Dictionary containing observations with image data for each camera
            infos: Information dictionary that may contain metadata to overlay on frames

        Note:
            Frames are processed for each camera view and tiled into a grid layout
            for efficient video recording across multiple parallel environments.
        """

        is_info_on_video = self.video_cfg.info_on_video
        if isinstance(infos, dict) and infos == {}:
            is_info_on_video = False
        else:
            infos = common.to_numpy(infos)

        if isinstance(extracted_obs, dict):
            extracted_obs = [extracted_obs]

        for frame_idx, frame_obs in enumerate(extracted_obs):
            images = {camera_name: [] for camera_name in self.camera_names}
            for env_id in range(self.num_envs):
                for camera_name in self.camera_names:
                    image = frame_obs["images_and_states"][camera_name][env_id, :]
                    image = common.to_numpy(image)
                    if is_info_on_video:
                        info = infos[frame_idx]
                        info_item = {
                            k: v if np.size(v) == 1 else "" for k, v in info.items()
                        }
                        image = put_info_on_image(image, info_item)
                    images[camera_name].append(image)
            for camera_name in self.camera_names:
                full_image = tile_images(
                    images[camera_name], nrows=int(np.sqrt(self.num_envs))
                )
                self.render_images[camera_name].append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Generates and saves video files from accumulated frames.

        Args:
            video_sub_dir: Optional subdirectory name within the base video directory.
                If provided, videos will be saved in a subdirectory of the seed-specific folder.

        Note:
            Videos are saved with unique names based on a counter and camera name.
            The frame buffer is cleared after successful video generation.
        """

        for camera_name in self.camera_names:
            print(
                f"total {len(self.render_images[camera_name])} frames for {camera_name}"
            )

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        for camera_name in self.camera_names:
            images_to_video(
                self.render_images[camera_name],
                output_dir=output_dir,
                video_name=f"{self.video_cnt}_{camera_name}",
            )
        self.video_cnt += 1
        self.render_images = {camera_name: [] for camera_name in self.camera_names}
