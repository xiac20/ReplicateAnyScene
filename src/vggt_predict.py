from html import parser
import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import argparse
import random
import numpy as np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import cv2
from PIL import Image
from src.geometry_utils import predictions_to_pcd

def vggt_predict(images, model):
    '''
    Predict depth maps and camera poses for a sequence of input frames using the VGGT model.
    Args:
        images: loaded frames from utils.load_video_frames
    Returns:        
        A dictionary containing:
        - point_cloud_data: a trimesh.PointCloud object representing the predicted point cloud
        - colors: numpy array of shape (S, H, W, 3)
        - depths: numpy array of shape (S, H, W)
        - extrinsics: numpy array of shape (S, 4, 4)
        - world_points: numpy array of shape (S, H, W, 3)
        - world_points_conf: numpy array of shape (S, H, W)
        - intrinsic: numpy array of shape (3, 3)
    '''
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16 
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions['images'] = images.cpu().numpy()

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    point_cloud_data = predictions_to_pcd(
            predictions,
            conf_thres=50.0,  # 默认值
            filter_by_frames="All",
            mask_black_bg=False,
            mask_white_bg=False,
            prediction_mode="Depthmap and Camera Branch",
        )
    
    colors = (predictions['images'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions['depth'].squeeze(-1)
    extrinsics = np.pad(predictions['extrinsic'], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    extrinsics[:, 3, 3] = 1
    world_points = predictions['world_points_from_depth'].copy()
    world_points_conf = predictions['world_points_conf'].copy()
    intrinsic = np.mean(predictions['intrinsic'], axis=0)

    return {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "world_points": world_points,
        "world_points_conf": world_points_conf,
        "intrinsic": intrinsic,
    }