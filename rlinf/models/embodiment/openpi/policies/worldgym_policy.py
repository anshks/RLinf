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
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_worldgym_example() -> dict:
    """Creates a random input example for the WorldGym policy."""
    return {
        "observation/state": np.random.rand(20),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "Fold the shirt on the table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _pad_to_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    array = np.asarray(array)
    current_dim = array.shape[-1]
    if current_dim == target_dim:
        return array
    if current_dim > target_dim:
        return array[..., :target_dim]
    pad_width = [(0, 0)] * (array.ndim - 1) + [(0, target_dim - current_dim)]
    return np.pad(array, pad_width, mode="constant")


@dataclasses.dataclass(frozen=True)
class WorldGymInputs(transforms.DataTransformFn):
    """WorldGym transforms for single-view inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class WorldGymPadStateActions(transforms.DataTransformFn):
    """Pad state/actions to the model action dimension after normalization."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            data["state"] = _pad_to_dim(data["state"], self.action_dim)
        if "actions" in data:
            data["actions"] = _pad_to_dim(data["actions"], self.action_dim)
        return data


@dataclasses.dataclass(frozen=True)
class WorldGymOutputs(transforms.DataTransformFn):
    """WorldGym output parser (20D actions)."""

    def __call__(self, data: dict) -> dict:
        # WorldGym cloth folding uses 20D actions in the checkpoint stats.
        return {"actions": np.asarray(data["actions"][:, :20])}
