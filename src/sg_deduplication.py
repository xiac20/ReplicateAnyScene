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

def get_overlap_ratio(source_pcd, target_pcd):
    """
    Compute overlap ratio of source in target
    Args:
        source_pcd: o3d.geometry.PointCloud object
        target_pcd: o3d.geometry.PointCloud object
    Returns:
        Overlap_ratio of source pcd in target pcd
    """
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

def self_category_deduplicate():
    '''
    Deduplication inside a single category
    Args:
        category_masks: A list of segmented masks returned by segment_and_track function
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: Only keep top conf_k confidence 3D points
        overlap_thre: Merge instance if overlap ratio exceeds overlap_thre
    Returns:
        deduplicated_category_masks: A list of deduplicated instance masks
    '''
    pass