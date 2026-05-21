from __future__ import annotations

import numpy as np


def forward_kinematics(
    joints: list[object],
    local_rotations: list[np.ndarray],
    root_offset: np.ndarray,
    topological_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Student part-1 implementation.

    Expected inputs:
    - joints: each joint has `.parent` and `.translation`
    - local_rotations: one 3x3 local rotation matrix per joint
    - root_offset: global translation applied to the root
    - topological_order: optional parent-before-child traversal order

    Expected outputs:
    - world_rotations: shape (J, 3, 3)
    - world_positions: shape (J, 3)
    """
    joint_count = len(joints)
    
    # Initialize
    world_positions = np.zeros((joint_count, 3), dtype=np.float32)
    world_rotations = np.zeros((joint_count, 3, 3), dtype=np.float32)

    # Check if world roataion is already computed
    computed = [False] * joint_count

    # Ensure parent is calculated before child
    def parent_before_child_joint(i):
        if computed[i]:
            return
        parent_index = joints[i].parent

        # For root joint
        if parent_index == -1:
            # R_world(root) = R_local(root)
            # p_world(root) = t_root + root_offset
            world_rotations[i] = local_rotations[i]
            world_positions[i] = joints[i].translation + root_offset

        # For child joint
        else:
            # Calculate its parent joint
            parent_before_child_joint(parent_index)
            # R_world(i) = R_world(p(i)) R_local(i)
            # p_world(i) = p_world(p(i)) + R_world(p(i)) t_local(i)
            world_rotations[i] = world_rotations[parent_index] @ local_rotations[i]
            world_positions[i] = world_positions[parent_index] + world_rotations[parent_index] @ joints[i].translation
        
        computed[i] = True

    if topological_order is not None:
        order = topological_order
    else:
        order = range(joint_count)

    for i in order:
        parent_before_child_joint(i)
    
    return world_rotations, world_positions
