import os
import argparse
import numpy as np
import trimesh
import json


def _get_wall_alignment_target(forward_vector, angle_tolerance=30.0):
    """
    Map object's forward direction to nearest horizontal cardinal axis.
    Returns (align_vector, target_axis) or (None, None) if outside tolerance.
    """
    yaw_deg = np.degrees(np.arctan2(forward_vector[1], forward_vector[0]))
    yaw_candidates = [
        (0.0, np.array([1.0, 0.0, 0.0]), 'x'),
        (90.0, np.array([0.0, 1.0, 0.0]), 'y'),
        (-90.0, np.array([0.0, -1.0, 0.0]), 'y'),
        (180.0, np.array([-1.0, 0.0, 0.0]), 'x'),
        (-180.0, np.array([-1.0, 0.0, 0.0]), 'x'),
    ]

    best = None
    best_dist = float("inf")
    for target_yaw, align_vec, axis in yaw_candidates:
        dist = abs(((yaw_deg - target_yaw + 180.0) % 360.0) - 180.0)
        if dist < best_dist:
            best_dist = dist
            best = (align_vec, axis)

    if best is None or best_dist > angle_tolerance:
        return None, None
    return best


def _select_closest_wall(walls_info, axis, center, span_margin=0.3):
    """
    Select nearest wall on a given axis. Prefer walls whose span covers object center.
    Returns (wall_info, distance) or (None, inf).
    """
    candidates = [w for w in walls_info if w.get('axis') == axis]
    if len(candidates) == 0:
        return None, float("inf")

    other_axis_idx = 1 if axis == 'x' else 0
    axis_idx = 0 if axis == 'x' else 1
    center_other = center[other_axis_idx]

    in_span = []
    for wall in candidates:
        span_start, span_end = wall.get('span', (-float("inf"), float("inf")))
        lo = min(span_start, span_end) - span_margin
        hi = max(span_start, span_end) + span_margin
        if lo <= center_other <= hi:
            in_span.append(wall)

    lookup = in_span if len(in_span) > 0 else candidates
    best_wall = min(lookup, key=lambda w: abs(w['position'] - center[axis_idx]))
    best_dist = abs(best_wall['position'] - center[axis_idx])
    return best_wall, best_dist

def refine_supported_by_floor_object(object_info):
    '''
    This function refines the "supported_by_floor" relationship by align the axis and bottom of the object to the floor.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
    Returns: The same dict with a different "T" to align the object to the floor.
    '''
    transform_matrix = object_info["T"]
    upper_real_vector = np.array([0, 0, 1])
    # Note that glb is y-up, but ours format is z-up.
    upper_transformed_vector = transform_matrix[:3,1] / np.linalg.norm(transform_matrix[:3,1])
    theta_gravity = np.arccos(np.clip(np.dot(upper_real_vector, upper_transformed_vector), -1.0, 1.0)) / np.pi * 180
    # if the upper_transformed_vector is close to the gravity direction, we consider it as supported_by_floor and align it to the floor. Otherwise, we keep the original transformation.
    if theta_gravity < 10.0 or theta_gravity > 170.0:
        if theta_gravity < 90.0:
            upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, upper_real_vector)
        else:
            upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, -upper_real_vector)
    else:
        upper_align_matrix = np.eye(4)
    # align the axis of the object to the floor, we only align the rotation part and keep the translation part unchanged to preserve the original position of the object.
    transform_matrix[:3, :3] = upper_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    # align the bottom of the object to the floor
    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]
    # if the bottom of the object is close to the floor (z=0), we consider it as supported_by_floor and align it to the floor. Otherwise, we keep the original transformation.
    if abs(z_min) < 0.3:
        translation_vector = np.array([0, 0, -z_min])
        translation_matrix = trimesh.transformations.translation_matrix(translation_vector)
    else:
        translation_matrix = np.eye(4)
    # align the bottom of the object to the floor
    transform_matrix = translation_matrix @ transform_matrix
    object_info["T"] = transform_matrix
    return object_info


def refine_embedded_in_wall_object(object_info, walls_info):
    '''
        This function refines the "embedded_in_wall" relationship by aligning the object horizontal axis
        to a wall direction and snapping the object center to the nearest valid wall plane.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
      walls_info: list returned by get_walls_info().
            camera_pos: optional camera position, kept for interface consistency with attached-to-wall refinement.
    Returns: The same dict with refined "T".
    '''
    if not walls_info:
        return object_info

    transform_matrix = object_info["T"].copy()
    forward_vector = transform_matrix[:3, 2]
    forward_norm = np.linalg.norm(forward_vector)
    if forward_norm < 1e-8:
        return object_info
    forward_vector = forward_vector / forward_norm

    align_vector, wall_axis = _get_wall_alignment_target(forward_vector, angle_tolerance=30.0)
    if align_vector is None:
        return object_info

    # align object horizontal axis to wall axis; only update rotation and keep translation unchanged
    horizontal_align_matrix = trimesh.geometry.align_vectors(transform_matrix[:3, 2], align_vector)
    transform_matrix[:3, :3] = horizontal_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    center = transform_matrix[:3, 3]
    # snap the object center to the nearest wall plane if distance is small enough
    nearest_wall, min_dist = _select_closest_wall(walls_info, wall_axis, center, span_margin=0.3)
    if nearest_wall is not None and min_dist <= 0.3:
        axis_idx = 0 if wall_axis == 'x' else 1
        offset = nearest_wall['position'] - center[axis_idx]
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[axis_idx] = offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    # keep object above floor plane z=0
    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]
    if z_min < 0.0:
        transform_matrix = trimesh.transformations.translation_matrix(np.array([0.0, 0.0, -z_min])) @ transform_matrix

    object_info["T"] = transform_matrix
    return object_info


def refine_attached_to_wall_object(object_info, walls_info, camera_pos=None):
    '''
        This function refines the "attached_to_wall" relationship by aligning the object horizontal axis,
        and snapping the back surface to the nearest wall plane.
    Args: A dict containing:
        - original_mesh: a trimesh.Trimesh object representing the 3D mesh of the object instance
        - T: the transformation matrix to align the mesh to the world coordinate system
      walls_info: list returned by get_walls_info().
            camera_pos: optional camera position used to determine which side of the object should touch the wall.
    Returns: The same dict with refined "T".
    '''
    if not walls_info:
        return object_info

    transform_matrix = object_info["T"].copy()
    forward_vector = transform_matrix[:3, 2]
    forward_norm = np.linalg.norm(forward_vector)
    if forward_norm < 1e-8:
        return object_info
    forward_vector = forward_vector / forward_norm

    align_vector, wall_axis = _get_wall_alignment_target(forward_vector, angle_tolerance=20.0)
    if align_vector is None:
        return object_info

    # align object horizontal axis to wall axis; only update rotation and keep translation unchanged
    horizontal_align_matrix = trimesh.geometry.align_vectors(transform_matrix[:3, 2], align_vector)
    transform_matrix[:3, :3] = horizontal_align_matrix[:3, :3] @ transform_matrix[:3, :3]

    center = transform_matrix[:3, 3]
    nearest_wall, min_dist = _select_closest_wall(walls_info, wall_axis, center, span_margin=0.3)
    if nearest_wall is not None and min_dist <= 0.3:
        axis_idx = 0 if wall_axis == 'x' else 1

        # use back face as contact surface. camera_pos determines which side is considered as back side.
        transformed_vertices = trimesh.transformations.transform_points(
            object_info["original_mesh"].vertices.copy(),
            transform_matrix,
        )
        if camera_pos is not None:
            if camera_pos[axis_idx] > center[axis_idx]:
                contact_val = transformed_vertices[:, axis_idx].min()
            else:
                contact_val = transformed_vertices[:, axis_idx].max()
        else:
            # fallback to aligned direction when camera position is unavailable
            if align_vector[axis_idx] > 0:
                contact_val = transformed_vertices[:, axis_idx].min()
            else:
                contact_val = transformed_vertices[:, axis_idx].max()

        snap_offset = nearest_wall['position'] - contact_val
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[axis_idx] = snap_offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    # keep object above floor plane z=0
    transformed_mesh = object_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    z_min = transformed_mesh.bounds[0, 2]
    if z_min < 0.0:
        transform_matrix = trimesh.transformations.translation_matrix(np.array([0.0, 0.0, -z_min])) @ transform_matrix

    object_info["T"] = transform_matrix
    return object_info
