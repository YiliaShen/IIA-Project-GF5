from __future__ import annotations

import numpy as np

# Inputs for skinning:
# model_data.rest_vertices: mesh vertices in the default pose, shape (V, 3)
# model_data.rest_joints: joint centres in the default pose, shape (J, 3)
# model_data.ground_translation: shift that puts the rest body on the floor
# world_rotations: posed joint rotations from FK, shape (J, 3, 3)
# world_positions: posed joint centres from FK, shape (J, 3)
# model_data.skinning_weights: vertex-to-joint weights, shape (V, J)


# Rest Vertices And Rest Joints:
# The rest pose is the character before animation.
# rest_vertices[i]: where mesh vertex i starts in rest pose
# rest_joints[j]: where joint j starts in rest pose
# add ground_translation so both are in viewer world coordinates
# rest_vertices[i] - rest_joints[j]: vertex i as seen from joint j


# The Skinning Transform
# Each joint gives vertex i one possible posed location.
# start from the vertex's rest-pose offset from that joint
# rotate that offset using the joint's posed world rotation
# place the rotated offset at the joint's posed world position
# combine the joint proposals using the skinning weights


# One-Hot Skinning
# One-hot skinning is the simplest baseline:
# find the joint with the largest weight for each vertex
# set that joint's weight to 1
# set all other weights to 0
# This creates rigid piecewise motion. It is easy to implement and easy to interpret, which is why it is a good first baseline.


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    return np.zeros_like(np.asarray(weights, dtype=np.float32))


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    rest_vertices = np.asarray(model_data.rest_vertices, dtype=np.float32)
    return np.zeros_like(rest_vertices)
