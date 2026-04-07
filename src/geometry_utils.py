import numpy as np
import trimesh
import matplotlib
from scipy.spatial import Delaunay

def compute_surface_area_from_pointmap(pointmap, mask, max_triangle_size = 2e-4):
    """
    Compute the surface area of an object given its pointmap and mask using Delaunay triangulation.
    
    Args:
        pointmap: HxWx3 array where each element is a 3D coordinate (X, Y, Z).
        mask: HxW binary array where True indicates the object and False is background.
        max_triangle_size: Maximum allowed area for a triangle to be considered valid (to filter outliers).
    
    Returns:
        The computed surface area of the object.
    """
    H, W, _ = pointmap.shape
    
    # Extract points from pointmap based on the mask
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 3:
        return 0.0
    
    points_3d = pointmap[y_coords, x_coords]
    
    # Ensure there are at least 3 valid points
    if points_3d.shape[0] < 3:
        return 0.0
    
    pixel_coords = np.column_stack([x_coords, y_coords])

    try:
        tri = Delaunay(pixel_coords)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return 0.0
    
    simplices = tri.simplices
    triangles_3d = points_3d[simplices]
    
    # Calculate vectors AB and AC for each triangle
    AB = triangles_3d[:, 1] - triangles_3d[:, 0]
    AC = triangles_3d[:, 2] - triangles_3d[:, 0]

    # Calculate cross product and triangle areas
    cross_product = np.cross(AB, AC)
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    # Filter out invalid triangles
    valid_triangle_mask = (triangle_areas > 0) & (triangle_areas < max_triangle_size)
    valid_areas = triangle_areas[valid_triangle_mask]
    
    total_area = np.sum(valid_areas)
    
    return total_area

def predictions_to_pcd(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    prediction_mode="Predicted Depthmap",
) -> trimesh.Scene:
    """
    Copied from vggt/visual_util.py
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    return point_cloud_data

def get_plane_info(pointmap, mask):
    '''
    Compute the plane parameters from a pointmap and binary mask using PCA.
    
    Args:
    - pointmap: HxWx3 numpy array where each element is a 3D coordinate (X, Y, Z).
    - mask: HxW binary numpy array where True indicates the object and False is background.
    Returns:
    - A dictionary
        {
            'normal': np.ndarray of shape (3,), the normal vector of the plane,
            'd': float, the distance from the plane to the origin,
            'area': float, the surface area of the plane,
            'centroid': np.ndarray of shape (3,), the centroid of the plane
            'mean_distance': float, the mean distance of points to the fitted plane
        }
    '''
    masked_points = pointmap[mask]  # shape (N, 3)
    centroid = np.mean(masked_points, axis=0)  # shape (3,)
    centered_points = masked_points - centroid
    cov_matrix = np.dot(centered_points.T, centered_points)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    except Exception as e:
        return {
            'normal': np.array([0, 0, 1]),  # Default normal vector
            'd': -centroid[2],  # Distance from the plane to the origin
            'area': 0,  # Area based on the number of points
            'centroid': centroid,  # Centroid of the points
            'mean_distance': 1e6  # Large mean distance to indicate poor fit
        }
    min_eigenvalue_idx = np.argmin(eigenvalues)
    normal = eigenvectors[:, min_eigenvalue_idx]
    normal = -normal if normal[0] < 0 else normal
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    d = -np.dot(normal, centroid)
    distance_numerator = np.abs(np.dot(masked_points, normal) + d)
    distance_denominator = np.linalg.norm(normal)
    point_distances = distance_numerator / distance_denominator
    mean_distance = np.mean(point_distances)
    area = compute_surface_area_from_pointmap(pointmap, mask)

    return {
        'normal': normal,
        'd': d,
        'area': area,
        'centroid': centroid,
        'mean_distance': mean_distance
    }

def align_to_room_coordinate_system(world_points, wall_masks, floor_masks, wall_mean_distance_thres=0.02, floor_mean_distance_thres=0.02):
    '''
    Align the scene to a room coordinate system based on the detected wall and floor masks.
    
    Args:
    - world_points: numpy array of shape (T, H, W, 3) representing the 3D coordinates of each pixel in each frame.
    - wall_masks: list of dictionaries containing 'frame_id' and 'mask' for detected walls.
    - floor_masks: list of dictionaries containing 'frame_id' and 'mask' for detected floors.
    
    Returns:
    - R: numpy array of shape (3, 3) representing the rotation matrix to align the scene to the room coordinate system.
    - t: numpy array of shape (3,) representing the translation vector to align the scene to the room coordinate system.
    '''
    wall_plane_infos = []
    floor_plane_infos = []
    for wall_mask in wall_masks:
        frame_id = wall_mask['frame_id']
        mask = wall_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < wall_mean_distance_thres:
            wall_plane_infos.append(plane_info)
    for floor_mask in floor_masks:
        frame_id = floor_mask['frame_id']
        mask = floor_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < floor_mean_distance_thres:
            floor_plane_infos.append(plane_info)
    if len(floor_plane_infos) == 0:
        return np.eye(3), np.zeros(3)
    # choose the floor plane with the largest area and normal vector close to mean floor normal vector (in case wrong floor segmentation)
    mean_floor_normal = np.mean([info['normal'] for info in floor_plane_infos], axis=0)
    mean_floor_normal = mean_floor_normal / np.linalg.norm(mean_floor_normal)
    vaild_floor_plane_infos = [info for info in floor_plane_infos if abs(np.dot(info['normal'], mean_floor_normal)) > np.cos(np.radians(30))]
    floor_plane_info = max(vaild_floor_plane_infos, key=lambda x: x['area'])
    floor_normal = floor_plane_info['normal']
    # choose the wall plane with the largest area and orthogonal (within 5 degrees) to the floor
    orthogonal_wall_plane_infos = [info for info in wall_plane_infos if abs(np.dot(info['normal'], floor_normal)) < np.cos(np.radians(85))]
    if len(orthogonal_wall_plane_infos) == 0:
        return np.eye(3), np.zeros(3)
    wall_plane_info = max(orthogonal_wall_plane_infos, key=lambda x: x['area'])
    wall_normal_1 = wall_plane_info['normal']
    # the floor normal should be upward, use the wall centroid to determine the direction of the wall normal
    floor_to_wall_vector = wall_plane_info['centroid'] - floor_plane_info['centroid']
    if np.dot(floor_to_wall_vector, floor_normal) < 0:
        floor_normal = -floor_normal
    # get the third axis by cross product and refine the wall normal by cross product to ensure orthogonality
    wall_normal_2 = np.cross(floor_normal, wall_normal_1)
    wall_normal_2 = wall_normal_2 / np.linalg.norm(wall_normal_2)
    wall_normal_1 = np.cross(wall_normal_2, floor_normal)
    wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)
    R = np.stack([wall_normal_1, wall_normal_2, floor_normal], axis=0)
    # use the floor plane to determine the translation, set the floor plane to be at z=0
    floor_centroid = floor_plane_info['centroid']
    rotated_floor_centroid = floor_centroid @ R.T
    current_floor_z = rotated_floor_centroid[2]
    t = np.zeros(3)
    t[2] = -current_floor_z
    # set the origin to the center of the scene bbox
    all_points = world_points.reshape(-1, 3)
    rotated_points = all_points @ R.T
    min_coords = np.min(rotated_points, axis=0)
    max_coords = np.max(rotated_points, axis=0)
    center = (min_coords + max_coords) / 2
    t[:2] = -center[:2]
    return R, t

def align_vggt_predictions(predictions, R, t):
    '''
    Align the VGGt predictions to the room coordinate system using the given rotation and translation.
    
    Args:
    - predictions: dictionary containing VGGt predictions.
    - R: numpy array of shape (3, 3) representing the rotation matrix to align the scene to the room coordinate system.
    - t: numpy array of shape (3,) representing the translation vector to align the scene to the room coordinate system.
    Returns:
    - predictions: dictionary containing the aligned VGGt predictions.
    '''

    # Update extrinsic matrices in predictions
    c2w_old = predictions["extrinsics"]  # shape: (N, 4, 4)
    R_c2w_old = c2w_old[:, :3, :3]      # shape: (N, 3, 3)
    t_c2w_old = c2w_old[:, :3, 3]       # shape: (N, 3)
    R_c2w_new = R_c2w_old @ R.T           # shape: (N, 3, 3)
    t_c2w_new = t_c2w_old - (R_c2w_new @ t)  # shape: (N, 3)
    predictions["extrinsics"][:, :3, :3] = R_c2w_new
    predictions["extrinsics"][:, :3, 3] = t_c2w_new

    # update world points in predictions
    predictions['world_points'] = predictions['world_points'] @ R.T + t

    # update pcd
    predictions['point_cloud_data'].apply_transform(np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]]))
    return predictions

def get_optimal_view_frame_id(world_points, instance_masks):
    '''
    Get the optimal view frame id for each instance based on 3D surface area
    Args:
        world_points: numpy array of shape (T, H, W, 3) representing the 3D coordinates of each pixel in each frame.
        instance_masks: list of dictionaries containing 'frame_id' and 'mask' for each instance.
    Returns:
        A integer representing the optimal view frame id for the instance.
    '''
    optimal_frame_id = -1
    max_area = 0
    for instance_mask in instance_masks:
        frame_id = instance_mask['frame_id']
        mask = instance_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        area = compute_surface_area_from_pointmap(pointmap, mask)
        if area > max_area:
            max_area = area
            optimal_frame_id = frame_id
    return optimal_frame_id
