import os
import cv2
import numpy as np
import trimesh
import open3d as o3d

class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

def get_overlap_ratio(source_pts, target_pts):
    """
    Compute overlap ratio of source in target.
    Args:
        source_pts: numpy array of shape (N1, 3)
        target_pts: numpy array of shape (N2, 3)
    Returns:
        Overlap_ratio of source pcd in target pcd
    """
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return 0.0
    
    # bbox judgement
    s_min = source_pcd.get_min_bound()
    s_max = source_pcd.get_max_bound()
    t_min = target_pcd.get_min_bound()
    t_max = target_pcd.get_max_bound()
    
    if (s_max[0] < t_min[0] or s_min[0] > t_max[0] or  # X
        s_max[1] < t_min[1] or s_min[1] > t_max[1] or  # Y
        s_max[2] < t_min[2] or s_min[2] > t_max[2]):   # Z
        return 0.0
    
    # Further computation
    threshold = np.mean(source_pcd.compute_nearest_neighbor_distance()) * 3.0
    
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    dists = np.asarray(dists)

    overlap_count = np.sum(dists < threshold)
    return overlap_count / len(source_pcd.points)

def self_category_deduplicate(category_masks, world_points, world_points_conf, conf_k=50, overlap_thre=0.3):
    '''
    Deduplication inside a single category
    Args:
        category_masks: A list of segmented masks returned by segment_and_track function
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: Only keep top conf_k percent confidence 3D points
        overlap_thre: Merge instance if overlap ratio exceeds overlap_thre
    Returns:
        deduplicated_category_masks: A list of deduplicated instance masks
    '''
    if not category_masks:
        return []
    print(f"Starting self-deduplication for category with {len(category_masks)} raw instances...")

    # back-project instance masks into pointcloud 
    instance_point_arrays = [] 
    valid_indices = [] 

    for idx, instance_frames in enumerate(category_masks):
        obj_pts_list = []
        obj_conf_list = []

        for frame_data in instance_frames:
            frame_id = frame_data['frame_id']
            mask = frame_data['mask']
            
            # 边界检查
            if frame_id >= world_points.shape[0]:
                continue
                
            pts_frame = world_points[frame_id]
            conf_frame = world_points_conf[frame_id]
            
            valid_pixels = mask > 0
            if not np.any(valid_pixels):
                continue
                
            v_pts = pts_frame[valid_pixels]
            v_conf = conf_frame[valid_pixels]
            
            if len(v_conf) > 0:
                obj_pts_list.append(v_pts)
                obj_conf_list.append(v_conf)
        
        # Global Filtering per Instance
        if obj_pts_list:
            full_pts = np.concatenate(obj_pts_list, axis=0)
            full_conf = np.concatenate(obj_conf_list, axis=0)
            
            # confidence threshold
            thresh = 0.0
            if conf_k > 0:
                thresh = np.percentile(full_conf, conf_k)
                
            conf_mask = (full_conf >= thresh) & (full_conf > 1e-5)
            
            if np.any(conf_mask):
                final_pts = full_pts[conf_mask]
                # point num filter
                if len(final_pts) >= 50: 
                    instance_point_arrays.append(final_pts)
                    valid_indices.append(idx)

    if len(instance_point_arrays) <= 1:
        return category_masks

    # Union-Find Clustering
    uf = UnionFind(range(len(instance_point_arrays)))
    n = len(instance_point_arrays)
    
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            
            ov1 = get_overlap_ratio(instance_point_arrays[i], instance_point_arrays[j])
            ov2 = get_overlap_ratio(instance_point_arrays[j], instance_point_arrays[i])
            
            if ov1 >= overlap_thre or ov2 >= overlap_thre:
                uf.union(i, j)
                
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(valid_indices[i])

    # get deduplicated category masks
    deduplicated_category_masks = []
    for group_indices in groups.values():
        merged_frames = {}
        for orig_idx in group_indices:
            original_instance = category_masks[orig_idx]
            for frame_data in original_instance:
                fid = frame_data['frame_id']
                msk = frame_data['mask']
                if fid not in merged_frames:
                    merged_frames[fid] = np.zeros_like(msk)
                merged_frames[fid] = np.logical_or(merged_frames[fid], msk)
        
        merged_instance_list = []
        for fid in sorted(merged_frames.keys()):
            merged_instance_list.append({
                'frame_id': fid,
                'mask': merged_frames[fid]
            })
            
        deduplicated_category_masks.append(merged_instance_list)

    print(f"  -> Reduced {len(category_masks)} instances to {len(deduplicated_category_masks)} instances.")
    return deduplicated_category_masks

def cross_category_deduplicate(all_masks, world_points, world_points_conf, conf_k=50, overlap_thre=0.5):
    '''
    Deduplication across all categories
    Args:
        all_masks: A dict, key is the category name and value is a list of segmented masks
                   (list of list of dicts with frame_id and mask)
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: Only keep top conf_k percent confidence 3D points
        overlap_thre: Merge instance if overlap ratio exceeds overlap_thre
    Returns:
        deduplicated_all_masks: A dict (same format as input) containing deduplicated instances
    '''
    print(f"Starting cross-category deduplication...")
    
    all_candidates = []
    
    for cat_name, instances_list in all_masks.items():
        for inst_idx, instance_frames in enumerate(instances_list):
            obj_pts_list = []
            obj_conf_list = []
            
            for frame_data in instance_frames:
                frame_id = frame_data['frame_id']
                mask = frame_data['mask']
                
                if frame_id >= world_points.shape[0]:
                    continue
                    
                pts_frame = world_points[frame_id]
                conf_frame = world_points_conf[frame_id]
                
                valid_pixels = mask > 0
                if not np.any(valid_pixels):
                    continue
                
                v_pts = pts_frame[valid_pixels]
                v_conf = conf_frame[valid_pixels]
                
                if len(v_conf) > 0:
                    obj_pts_list.append(v_pts)
                    obj_conf_list.append(v_conf)

            if obj_pts_list:
                full_pts = np.concatenate(obj_pts_list, axis=0)
                full_conf = np.concatenate(obj_conf_list, axis=0)
                
                thresh = 0.0
                if conf_k > 0:
                    thresh = np.percentile(full_conf, conf_k)
                conf_mask = (full_conf >= thresh) & (full_conf > 1e-5)
                
                if np.any(conf_mask):
                    final_pts = full_pts[conf_mask]
                    
                    if len(final_pts) >= 50:
                        all_candidates.append({
                            "category": cat_name,
                            "points": final_pts,
                            "original_masks": instance_frames 
                        })

    if len(all_candidates) <= 1:
        return all_masks

    N = len(all_candidates)
    overlap_matrix = {} 
    print(f"  Comparing {N} instances globally...")
    uf = UnionFind(range(N))
    
    for i in range(N):
        for j in range(i + 1, N):
            if uf.find(i) == uf.find(j): 
                continue
            pts_i = all_candidates[i]["points"]
            pts_j = all_candidates[j]["points"]
            
            ov1 = get_overlap_ratio(pts_i, pts_j) 
            ov2 = get_overlap_ratio(pts_j, pts_i)
            overlap_matrix[(i, j)] = (ov1, ov2)
            
            if ov1 >= overlap_thre or ov2 >= overlap_thre:
                uf.union(i, j)
    final_groups = {}
    for i in range(N):
        root = uf.find(i)
        if root not in final_groups: 
            final_groups[root] = []
        final_groups[root].append(i) 
        
    final_objects = []
    
    for group_indices in final_groups.values():
        if len(group_indices) == 1:
            final_objects.append(all_candidates[group_indices[0]])
        else:
            target_idx = group_indices[0]
            min_avg_overlap = float('inf')
            
            for i in group_indices:
                total_overlap = 0.0
                count = 0
                for j in group_indices:
                    if i == j: continue
                    key = (min(i, j), max(i, j))
                    ov1, ov2 = overlap_matrix[key]
                    overlap_val = ov1 if i < j else ov2
                    
                    total_overlap += overlap_val
                    count += 1
                
                avg_overlap = total_overlap / count if count > 0 else 0
                
                if avg_overlap < min_avg_overlap:
                    min_avg_overlap = avg_overlap
                    target_idx = i
            
            target = all_candidates[target_idx]
            
            all_pts = [all_candidates[idx]["points"] for idx in group_indices]
            merged_pts = np.concatenate(all_pts, axis=0)
            
            merged_masks_dict = {}
            for idx in group_indices:
                for frame_data in all_candidates[idx]["original_masks"]:
                    fid = frame_data['frame_id']
                    msk = frame_data['mask']
                    if fid not in merged_masks_dict:
                        merged_masks_dict[fid] = np.zeros_like(msk)
                    merged_masks_dict[fid] = np.logical_or(merged_masks_dict[fid], msk)
            
            merged_masks_list = []
            for fid in sorted(merged_masks_dict.keys()):
                merged_masks_list.append({
                    'frame_id': fid,
                    'mask': merged_masks_dict[fid]
                })

            final_objects.append({
                "category": target["category"], 
                "points": merged_pts,
                "original_masks": merged_masks_list
            })

    deduplicated_all_masks = {}
    for obj in final_objects:
        if len(obj["original_masks"]) < 3:
            continue
        cat = obj["category"]
        if cat not in deduplicated_all_masks:
            deduplicated_all_masks[cat] = []
        deduplicated_all_masks[cat].append(obj["original_masks"])

    filtered_object_count = sum(len(instances) for instances in deduplicated_all_masks.values())
    print(f"  -> Reduced total instances to {filtered_object_count} global objects.")
    return deduplicated_all_masks