import numpy as np

def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Args:
        d6: (..., 6) array (first two columns of the rotation matrix)
    Returns:
        matrix: (..., 3, 3) rotation matrix
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    
    # Normalize the first vector
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    
    # Orthogonalize the second vector relative to the first
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    
    # Compute the third vector via cross product
    b3 = np.cross(b1, b2, axis=-1)
    
    return np.stack([b1, b2, b3], axis=-1)

def matrix_to_rotation_6d(matrix):
    """
    Converts 3x3 rotation matrix to 6D representation.
    Args:
        matrix: (..., 3, 3) rotation matrix
    Returns:
        d6: (..., 6) array (first two columns concatenated)
    """
    # Take the first two columns
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)

def aggregate_actions(actions, factor=3):
    """
    Aggregates a sequence of VLA actions to simulate a higher downsample factor.
    
    Args:
        actions: (T, 20) array of actions from downsample_factor=1.
                 Format: [L_dpos(3), L_drot(6), L_grip(1), R_dpos(3), R_drot(6), R_grip(1)]
        factor: The number of steps to aggregate (default 3).
        
    Returns:
        aggregated_actions: (T // factor, 20) array of aggregated actions.
    """
    T = actions.shape[0]
    n_chunks = T // factor
    
    # Truncate to make divisible by factor
    actions_trimmed = actions[:n_chunks * factor]
    
    # Reshape to (N, factor, 20) to process chunks
    chunks = actions_trimmed.reshape(n_chunks, factor, 20)
    
    # --- 1. Aggregate Positions (Sum of deltas) ---
    # Left Position (Indices 0-3)
    agg_left_pos = np.sum(chunks[..., 0:3], axis=1)
    # Right Position (Indices 10-13)
    agg_right_pos = np.sum(chunks[..., 10:13], axis=1)
    
    # --- 2. Aggregate Grippers (Take the first state) ---
    # We take index 0 because the action at t=0 contains the gripper state at t=0.
    # Left Gripper (Index 9)
    # agg_left_gripper = chunks[:, 0, 9:10]
    agg_left_gripper = chunks[:, 2, 9:10]  # Use the last gripper state in the chunk
    # Right Gripper (Index 19)
    # agg_right_gripper = chunks[:, 0, 19:20]
    agg_right_gripper = chunks[:, 2, 19:20]  # Use the last gripper state in the chunk
    
    # --- 3. Aggregate Rotations (Matrix Composition) ---
    # Helper to process rotation for a specific arm (indices start at `start_idx`)
    def aggregate_arm_rotation(start_idx):
        rot_6d_chunks = chunks[..., start_idx : start_idx + 6] # (N, factor, 6)
        
        # Convert all 6D vectors to matrices: (N, factor, 3, 3)
        rot_mats = rotation_6d_to_matrix(rot_6d_chunks)
        
        # Multiply matrices sequentially: R_total = R_last @ ... @ R_first
        # Note: Since these are global deltas applied sequentially:
        # P_next = Delta * P_curr.
        # R_1 = D_0 * R_0
        # R_2 = D_1 * R_1 = D_1 * (D_0 * R_0)
        # So we left-multiply the next delta.
        
        running_rot = rot_mats[:, 0] # Start with first delta
        for k in range(1, factor):
            # Apply next delta (k) to the left of the accumulated delta
            running_rot = rot_mats[:, k] @ running_rot
            
        return matrix_to_rotation_6d(running_rot)

    # Left Rotation (Indices 3-9)
    agg_left_rot = aggregate_arm_rotation(3)
    # Right Rotation (Indices 13-19)
    agg_right_rot = aggregate_arm_rotation(13)
    
    # --- 4. Reassemble ---
    aggregated_actions = np.concatenate([
        agg_left_pos,       # 3
        agg_left_rot,       # 6
        agg_left_gripper,   # 1
        agg_right_pos,      # 3
        agg_right_rot,      # 6
        agg_right_gripper   # 1
    ], axis=1)
    
    return aggregated_actions

# Example Usage:
# aggregated_chunk = aggregate_actions(action_chunk, factor=3)