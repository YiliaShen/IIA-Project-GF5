from __future__ import annotations

import numpy as np

# Inputs for skinning:
# model_data.rest_vertices: mesh vertices in the default pose, shape (V, 3)
# model_data.rest_joints: joint centres in the default pose, shape (J, 3)
# model_data.ground_translation: shift that puts the rest body on the floor
# world_rotations: posed joint rotations from FK, shape (J, 3, 3)
# world_positions: posed joint centres from FK, shape (J, 3)
# model_data.skinning_weights: vertex-to-joint weights, shape (V, J)


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    w = np.asarray(weights, dtype=np.float32)
    
    # set all weights to 0
    one_hot = np.zeros_like(np.asarray(weights, dtype=np.float32))

    # find the joint with the largest weight for each vertex (which bone impacts most significantly for each point)
    max_i = np.argmax(w, axis=1)
    
    # set that joint's weight to 1
    row = np.arange(len(weights)) # the vertex
    one_hot[row, max_i] = 1.0

    return one_hot


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    # Rest Vertices And Rest Joints:
    # The rest pose is the character before animation.
    # rest_vertices[i]: where mesh vertex i starts in rest pose
    # rest_joints[j]: where joint j starts in rest pose
    # add ground_translation so both are in viewer world coordinates
    # rest_vertices[i] - rest_joints[j]: vertex i as seen from joint j

    rest_vertices = np.asarray(model_data.rest_vertices, dtype=np.float32)
    rest_joints = np.asarray(model_data.rest_joints, dtype=np.float32)

    if use_blended_weights == True:
        weights = model_data.skinning_weights
    else:
        weights = model_data.one_hot_skinning_weights

    # Initialize the result matrix
    num_vertices = rest_vertices.shape[0]
    smpl_vertices = np.zeros_like(rest_vertices)
    
    # The Skinning Transform
    # Each joint gives vertex i one possible posed location.
    # start from the vertex's rest-pose offset from that joint
    # rotate that offset using the joint's posed world rotation
    # place the rotated offset at the joint's posed world position
    # combine the joint proposals using the skinning weights#
        
    num_joints = len(world_rotations)

    for i in range(num_joints):
        # transform vertex to local coordinate
        offset = rest_vertices - rest_joints[i:i+1, :]

        R = world_rotations[i] # rotation matrix (3,3)
        t = world_positions[i] # translation vector (3,)
        
        # T_j v_i: the local influence of a bone to all vertices = rest vertices * rotation + translation
        bone_weights = (offset @ R.T) + t

        # v'_i = sum_j w_ij T_j v_i: a vertex is influenced by many bones with different bone_weights
        smpl_vertices += bone_weights * weights[:, i:i+1]

    return smpl_vertices
