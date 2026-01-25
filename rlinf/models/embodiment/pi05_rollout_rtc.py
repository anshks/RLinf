import fire
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import imageio
import logging
import einops
from scipy.spatial.transform import Rotation as R
from PIL import Image

# Adjust sys.path to ensure local imports work if needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add src/ to path for package imports

from world_model_eval.world_model import WorldModel
from world_model_eval.pi05_inference_rtc import Pi05InferencePolicy


def r6_to_rotation(r6: np.ndarray) -> R:
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


def rotation_to_r6(rot: R) -> np.ndarray:
    """Convert scipy Rotation to r6 representation."""
    matrix = rot.as_matrix()
    return matrix[:, :2].T.flatten()


def load_image_as_tensor(image_path: Path, input_h: int, input_w: int) -> torch.Tensor:
    """Load a PNG image and convert to tensor with shape (H, W, C) and values in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_w, input_h), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
    return torch.from_numpy(img_np)


def main(
    image_dir: str,
    output_dir: str,
    world_model_checkpoint: str,
    policy_checkpoint: str,
    task: str,
    action_dim: int = 21,
    input_h: int = 256,
    input_w: int = 256,
    n_frames: int = 50,
    seed: int = 42,
    camera_name: str = "cam_high",
    device: str = "cuda",
    use_policy_quat: bool = False,
):
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(seed)
    np.random.seed(seed)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    world_model_ckpt_path = Path(world_model_checkpoint)
    policy_ckpt_path = Path(policy_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNG images in the image directory
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {image_dir}")
    logging.info(f"Found {len(image_paths)} PNG images in {image_dir}")

    # Initialize World Model
    if world_model_ckpt_path.is_dir():
        ckpts = sorted(world_model_ckpt_path.glob("ckpt_*.pt"))
        if not ckpts:
            raise ValueError(f"No checkpoints found in {world_model_ckpt_path}")
        latest_ckpt = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
        logging.info(f"Using world model checkpoint: {latest_ckpt}")
        world_model_path = str(latest_ckpt)
    else:
        world_model_path = str(world_model_ckpt_path)

    world_model_random_actions = WorldModel(world_model_path, action_dim=action_dim)
    world_model_policy = WorldModel(world_model_path, action_dim=action_dim)
    logging.info(f"World Model initialized with action_dim={action_dim}")

    # Initialize Policy
    logging.info(f"Loading Pi0.5 policy from {policy_ckpt_path}")
    policy = Pi05InferencePolicy(
        checkpoint_path=policy_ckpt_path,
        device=device,
        primary_camera=camera_name,
        verbose=True
    )

    # Run rollouts for each image
    for image_path in tqdm(image_paths, desc="Generating rollouts"):
        # Load the initial frame from PNG
        start_frame = load_image_as_tensor(image_path, input_h, input_w)
        random_actions = torch.randn(n_frames - 1, action_dim).to(world_model_random_actions.device).clip(-1, 1)

        world_model_random_actions.reset(start_frame.to(world_model_random_actions.device))
        world_model_policy.reset(start_frame.to(world_model_policy.device))
        
        # Track generated frames
        current_frame_np = start_frame.numpy()  # (H, W, C) float [0, 1]
        generated_frames_policy = [current_frame_np]
        generated_frames_random_actions = [current_frame_np]
        state_dim = policy.expected_state_dim
        
        # Initial state initialization with provided EE pose/quat values
        if state_dim > 0:
            current_state = np.zeros(state_dim, dtype=np.float32)
            
            # Left EE
            left_pos = np.array([0.35646688, 0.0382356, 0.92677665])
            # User provided quat: -0.08122765  0.70717267  0.69836195 -0.07482959 (wxyz)
            left_quat_wxyz = np.array([-0.08122765, 0.70717267, 0.69836195, -0.07482959])
            left_quat_xyzw = np.array([left_quat_wxyz[1], left_quat_wxyz[2], left_quat_wxyz[3], left_quat_wxyz[0]])
            
            left_rot = R.from_quat(left_quat_xyzw) 
            left_rot_matrix = left_rot.as_matrix() # 3x3
            # Extract r6 (first two columns flattened)
            left_r6 = left_rot_matrix[:, :2].T.flatten() # (2, 3) flattened -> (6,)
            
            current_state[0:3] = left_pos
            current_state[3:9] = left_r6
            current_state[9] = 1.0 # Left gripper
            
            # Right EE
            right_pos = np.array([0.35906339, -0.45805741, 0.92072002])
            right_quat_xyzw = np.array([0.70221996, 0.70186009, -0.07412596, -0.09372766])
            
            right_rot = R.from_quat(right_quat_xyzw)
            right_rot_matrix = right_rot.as_matrix()
            right_r6 = right_rot_matrix[:, :2].T.flatten()
            
            current_state[10:13] = right_pos
            current_state[13:19] = right_r6
            current_state[19] = 1.0 # Right gripper
        else:
            current_state = None
        
        # Generate subsequent frames
        print(f"Starting generation loop for {n_frames - 1} frames...")

        frames_generated = 0
        chunk_idx = 0

        with torch.no_grad():
            while frames_generated < n_frames - 1:
                # 1. Get action chunk from policy
                obs = {
                    "images": {
                        camera_name: current_frame_np
                    },
                    "task": task
                }
                if current_state is not None:
                    obs["qpos"] = current_state

                try:
                    # Predict action chunk (chunk_size, action_dim)
                    actions = policy.predict_action_chunk(obs, transform_to_quat=use_policy_quat)
                    actions = actions.squeeze(0)
                    chunk_size = actions.shape[0]
                    print(f"Chunk {chunk_idx}: Predicted action chunk shape: {actions.shape}")

                    # 2. Process actions for World Model
                    new_actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
                    min_dim = min(actions.shape[1], action_dim)
                    new_actions[:, :min_dim] = actions[:, :min_dim]

                    # 3. Generate frames one at a time (world model chunk_size=1)
                    actions_used = 0
                    for i in range(chunk_size):
                        if frames_generated >= n_frames - 1:
                            break

                        # Random action frame
                        random_action = random_actions[frames_generated:frames_generated + 1].unsqueeze(0).to(world_model_random_actions.device)  # (1, 1, D)
                        for _, frame_tensor in world_model_random_actions.generate_chunk(random_action):
                            frame_np = frame_tensor.squeeze().cpu().numpy()
                            generated_frames_random_actions.append(frame_np)

                        # Policy action frame
                        action_tensor = torch.from_numpy(new_actions[i:i+1]).float().unsqueeze(0).to(world_model_policy.device)  # (1, 1, D)
                        for _, frame_tensor in world_model_policy.generate_chunk(action_tensor):
                            frame_np = frame_tensor.squeeze().cpu().numpy()
                            generated_frames_policy.append(frame_np)
                            current_frame_np = frame_np

                        frames_generated += 1
                        actions_used += 1

                    # 4. Update current_state for each action in the chunk
                    if current_state is not None and actions_used > 0:
                        for a in range(actions_used):
                            action = new_actions[a]
                            
                            # Left arm position: add delta
                            current_state[0:3] += action[0:3]
                            
                            # Left arm rotation: R_new = R_current @ R_delta
                            left_rot_current = r6_to_rotation(current_state[3:9])
                            left_rot_delta = r6_to_rotation(action[3:9])
                            left_rot_new = left_rot_current * left_rot_delta
                            current_state[3:9] = rotation_to_r6(left_rot_new)
                            
                            # Left gripper: absolute value
                            current_state[9] = action[9]
                            
                            # Right arm position: add delta
                            current_state[10:13] += action[10:13]
                            
                            # Right arm rotation: R_new = R_current @ R_delta
                            right_rot_current = r6_to_rotation(current_state[13:19])
                            right_rot_delta = r6_to_rotation(action[13:19])
                            right_rot_new = right_rot_current * right_rot_delta
                            current_state[13:19] = rotation_to_r6(right_rot_new)
                            
                            # Right gripper: absolute value
                            current_state[19] = action[19]

                    chunk_idx += 1

                except Exception as e:
                    print(f"CRITICAL ERROR at chunk {chunk_idx}: {e}")
                    logging.error(f"Error at chunk {chunk_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        # Stack frames
        gen_video_policy = np.array(generated_frames_policy)  # (T, H, W, C)
        gen_video_random_actions = np.array(generated_frames_random_actions)  # (T, H, W, C)

        # Ensure same length
        min_len = min(len(gen_video_policy), len(gen_video_random_actions))
        gen_video_policy = gen_video_policy[:min_len]
        gen_video_random_actions = gen_video_random_actions[:min_len]

        # Concatenate side by side: policy actions | random actions
        # viz_video = np.concatenate([gen_video_policy, gen_video_random_actions], axis=2)
        viz_video = gen_video_policy

        # Save video using the image filename as base
        image_name = image_path.stem
        save_path = output_dir / f"rollout_{image_name}_policy.gif"
        viz_video_uint8 = (viz_video * 255).clip(0, 255).astype(np.uint8)
        imageio.mimsave(save_path, viz_video_uint8, fps=8)
        logging.info(f"Saved rollout for {image_name} to {save_path}")

    logging.info(f"Saved {len(image_paths)} rollouts to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)